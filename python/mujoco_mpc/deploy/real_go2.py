import time
import mujoco
import mujoco.viewer
import numpy as np
import argparse
import matplotlib.pyplot as plt
import scienceplots
from scipy.spatial.transform import Rotation as R
import time
from loop_rate_limiters import RateLimiter

# Shared Memory
from multiprocessing import shared_memory
import struct

# Unitree SDK2
import sys
from unitree_sdk2py.core.channel import (
    ChannelPublisher,
    ChannelSubscriber,
    ChannelFactoryInitialize,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.idl.default import (
    unitree_hg_msg_dds__LowCmd_,
    unitree_go_msg_dds__LowCmd_,
    unitree_go_msg_dds__LowState_,
    unitree_hg_msg_dds__LowState_,
    unitree_hg_msg_dds__MotorCmd_,
)
from unitree_sdk2py.utils.crc import CRC

from config import G1Config, Go2Config
from utils import unpack_control_data, pack_state_data, unpack_mocap_data,pack_control_data, read_motor_gears

controller_mode_mapping = {
    "position": 0,
    "torque": 1
}

class Real:
    def __init__(self, robot_name="g1"):
        self.control_mode = controller_mode_mapping["torque"]
        if robot_name == "g1":
            self.config = G1Config()
        elif robot_name == "go2":
            self.config = Go2Config()
        else:
            raise ValueError(f"Robot {robot_name} not supported")
        self.firstRun = True
        # Mujoco related
        self.mj_model = mujoco.MjModel.from_xml_path(self.config.xml_path_sim)
        self.mj_model.opt.timestep = self.config.dt_real
        self.mj_data = mujoco.MjData(self.mj_model)
        self.q = self.mj_model.key_qpos[0].copy()
        self.qd = self.mj_model.key_qvel[0].copy()
        print(f"xml_path_sim: {self.config.xml_path_sim}")
        self.motor_gears = read_motor_gears(self.config.xml_path_ctrl)
        # exit()
        # self.motor_gears = self._read_motor_gears(self.config.xml_path_ctrl)
        self.gear_array = np.ones(self.mj_model.nu) * 23.7

        # Update the array with gear values at the corresponding joint indices
        for motor in self.motor_gears.values():
            joint_index = motor['joint_index'] - 1
            gear = motor['gear']
            print(f"Joint index: {joint_index}, Gear: {gear}")
            self.gear_array[joint_index] = gear
        print(f"gear_array: {self.gear_array}")
        # sanity check
        assert (
            len(self.config.kp_real) == len(self.config.kd_real) == (self.config.nq_real-7)
        ), f"kp ({len(self.config.kp_real)}) and kd ({len(self.config.kd_real)}) must have the same length as nq_robot ({self.config.nq_real})"
        assert (
            (self.mj_model.nq + len(self.config.locked_joint_idx))
            == self.config.nq_real
        ), f"nq ({self.mj_model.nq}) + len(locked_joint_idx) ({len(self.config.locked_joint_idx)}) must equal nq_robot ({self.config.nq_real})"
        self.kp = self.config.kp_real
        self.kd = self.config.kd_real

        # Control message (send to robot)
        self.ctrl_msg = unitree_go_msg_dds__LowCmd_()
        self.ctrl_msg.mode_pr = 0
        self.ctrl_msg.mode_machine = 3

        # Shared Memory for control inputs (read from controller)
        self.ctrl_shm_size = 8 + self.config.nu_real * 8  # time + tau + q + qd
        
        # try to create the shared memory, if it already exists, delete it and create a new one
        try:
            self.ctrl_shm = shared_memory.SharedMemory(
                name="ctrl_shm", create=True, size=self.ctrl_shm_size
            )
            self.ctrl_buffer = self.ctrl_shm.buf
            self.ctrl_buffer[:] = pack_control_data(
                        self.ctrl_buffer, 0, np.zeros(self.config.nu_real)
                        )
        except FileExistsError:
            print("Shared memory already exists, deleting and creating a new one")
            self.ctrl_shm = shared_memory.SharedMemory(
                name="ctrl_shm", create=False, size=self.ctrl_shm_size
            )
            self.ctrl_shm.close()
            self.ctrl_shm.unlink()
            self.ctrl_shm = shared_memory.SharedMemory(
                name="ctrl_shm", create=True, size=self.ctrl_shm_size
            )
            self.ctrl_buffer = self.ctrl_shm.buf
            self.ctrl_buffer[:] = pack_control_data(
                        self.ctrl_buffer, 0, np.zeros(self.config.nu_real)
                        )
        # Shared Memory for state variables (send to controller)
        # Try to create the shared memory, if it already exists, delete it and create a new one
        self.state_shm_size = (
                8 + self.config.nq_real * 8 + self.config.nqd_real * 8
                )  # time + q + qd
        try:
            
            self.state_shm = shared_memory.SharedMemory(
                name="state_shm", create=True, size=self.state_shm_size
            )
            self.state_buffer = self.state_shm.buf
        except FileExistsError:
            print("Shared memory already exists, deleting and creating a new one")
            self.state_shm = shared_memory.SharedMemory(
                name="state_shm", create=False, size=self.state_shm_size
            )
            self.state_shm.close()
            self.state_shm.unlink()
            self.state_shm = shared_memory.SharedMemory(
                name="state_shm", create=True, size=self.state_shm_size
            )
            self.state_buffer = self.state_shm.buf

        # Shared Memory for mocap state (read from Vicon)
        self.shared_mem_name = "mocap_state_shm"
        self.shared_mem_size = (
            8 + 13 * 8
        )  # 8 bytes for utime (int64), 13 float64s (13*8 bytes)
        try:
            self.mocap_shm = shared_memory.SharedMemory(
                name=self.shared_mem_name, create=False, size=self.shared_mem_size
            )
        except FileNotFoundError:
            self.mocap_shm = shared_memory.SharedMemory(
                name=self.shared_mem_name, create=False, size=self.shared_mem_size
            )
            self.mocap_shm.close()
            self.mocap_shm.unlink()
            self.mocap_shm = shared_memory.SharedMemory(
                name=self.shared_mem_name, create=True, size=self.shared_mem_size
            )
        self.mocap_buffer = self.mocap_shm.buf
        self.low_cmd_msg = unitree_go_msg_dds__LowCmd_()
        self.low_cmd_msg.mode_pr = 0
        self.low_cmd_msg.mode_machine = 3
        self.crc = CRC()
        # Initialize Unitree SDK2
        ChannelFactoryInitialize(0, "en0")
        # self.InitLowCmd()
        self.low_state_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        print("init low state subscriber")
        self.low_state_subscriber.Init(self.low_state_handler, 10)
        print("init low state subscriber done")
        self.low_cmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.low_cmd_publisher.Init()

        
   

    def low_state_handler(self, msg: LowState_):
        # print("Received low state message")
        # print(msg.imu_state.gyroscope)
      
        self.low_state = msg
        for i in range(self.config.nq_real-7):
            self.q[7 + i] = msg.motor_state[i].q
            self.qd[6 + i] = msg.motor_state[i].dq
        # print imu message to check the communication
        
        if not self.config.use_mocap_ang_vel:
            omega = np.array([msg.imu_state.gyroscope]).flatten()
            self.qd[6:9] = omega
            
    def init_stand(self):
        while self.config.percent_3<1:
            time.sleep(0.002)
            self.low_cmd_msg.head[0]=0xFE
            self.low_cmd_msg.head[1]=0xEF
            self.low_cmd_msg.level_flag = 0xFF
            self.low_cmd_msg.gpio = 0
            for i in range(20):
                self.low_cmd_msg.motor_cmd[i].mode = 0x01
            if self.firstRun:
                for i in range(self.config.nu_real):
                    self.config.startPos[i] = self.low_state.motor_state[i].q
                self.firstRun = False

            self.config.percent_1 += 1.0 / self.config.duration_1
            self.config.percent_1 = min(self.config.percent_1, 1)
            if self.config.percent_1 < 1:
                for i in range(self.config.nu_real):
                    self.low_cmd_msg.motor_cmd[i].q = (1 - self.config.percent_1) * self.config.startPos[i] + self.config.percent_1 * self.config._targetPos_1[i]
                    self.low_cmd_msg.motor_cmd[i].dq = 0
                    self.low_cmd_msg.motor_cmd[i].kp = self.kp[i]
                    self.low_cmd_msg.motor_cmd[i].kd = self.kd[i]
                    self.low_cmd_msg.motor_cmd[i].tau = 0

            if (self.config.percent_1 == 1) and (self.config.percent_2 <= 1):
                self.config.percent_2 += 1.0 / self.config.duration_2
                self.config.percent_2 = min(self.config.percent_2, 1)
                for i in range(self.config.nu_real):
                    self.low_cmd_msg.motor_cmd[i].q = (1 - self.config.percent_2) * self.config._targetPos_1[i] + self.config.percent_2 * self.config._targetPos_2[i]
                    self.low_cmd_msg.motor_cmd[i].dq = 0
                    self.low_cmd_msg.motor_cmd[i].kp = self.kp[i]
                    self.low_cmd_msg.motor_cmd[i].kd = self.kd[i]
                    self.low_cmd_msg.motor_cmd[i].tau = 0

            if (self.config.percent_1 == 1) and (self.config.percent_2 == 1) and (self.config.percent_3 < 1):
                self.config.percent_3 += 1.0 / self.config.duration_3
                self.config.percent_3 = min(self.config.percent_3, 1)
                for i in range(self.config.nu_real):
                    self.low_cmd_msg.motor_cmd[i].q = self.config._targetPos_2[i] 
                    self.low_cmd_msg.motor_cmd[i].dq = 0
                    self.low_cmd_msg.motor_cmd[i].kp = self.kp[i]
                    self.low_cmd_msg.motor_cmd[i].kd = self.kd[i]
                    self.low_cmd_msg.motor_cmd[i].tau = 0

            
            
            self.low_cmd_msg.crc = self.crc.Crc(self.low_cmd_msg)
            self.low_cmd_publisher.Write(self.low_cmd_msg)
            # print('standing',self.config.percent_3)
        print("Stand up complete")

    def fix_position_as_init(self):
        # First loop: Check if the control input is approximately zero
        while True:
            print("Checking control signal")
            
            _,ctrl = unpack_control_data(self.ctrl_buffer, self.config.nu_real)
            # print(f"ctrl: {ctrl}")
            if np.allclose(ctrl, np.zeros(self.config.nu_real), atol=1e-3):
                for i in range(self.config.nu_real):
                    self.low_cmd_msg.motor_cmd[i].q = self.config._targetPos_2[i] 
                    self.low_cmd_msg.motor_cmd[i].dq = 0
                    self.low_cmd_msg.motor_cmd[i].kp = self.kp[i]
                    self.low_cmd_msg.motor_cmd[i].kd = self.kd[i]
                    self.low_cmd_msg.motor_cmd[i].tau = 0
                self.low_cmd_msg.crc = self.crc.Crc(self.low_cmd_msg)
                self.low_cmd_publisher.Write(self.low_cmd_msg)
                continue
            else:
                print("Initial control signal detected")
                initial_ctrl = ctrl
                print(f"initial_ctrl: {initial_ctrl}")
                for i in range(self.config.nu_real):
                    self.low_cmd_msg.motor_cmd[i].q = self.config._targetPos_2[i] 
                    self.low_cmd_msg.motor_cmd[i].dq = 0
                    self.low_cmd_msg.motor_cmd[i].kp = self.kp[i]
                    self.low_cmd_msg.motor_cmd[i].kd = self.kd[i]
                    self.low_cmd_msg.motor_cmd[i].tau = 0
                self.low_cmd_msg.crc = self.crc.Crc(self.low_cmd_msg)
                self.low_cmd_publisher.Write(self.low_cmd_msg)
                
                break
            time.sleep(0.002)

        while True:
            _, ctrl = unpack_control_data(self.ctrl_buffer, self.config.nu_real)
            if np.allclose(ctrl, initial_ctrl, atol=1e-3):
                for i in range(self.config.nu_real):
                    self.low_cmd_msg.motor_cmd[i].q = self.config._targetPos_2[i] 
                    self.low_cmd_msg.motor_cmd[i].dq = 0
                    self.low_cmd_msg.motor_cmd[i].kp = self.kp[i]
                    self.low_cmd_msg.motor_cmd[i].kd = self.kd[i]
                    self.low_cmd_msg.motor_cmd[i].tau = 0
                self.low_cmd_msg.crc = self.crc.Crc(self.low_cmd_msg)
                self.low_cmd_publisher.Write(self.low_cmd_msg)
                
                continue
            else:
                print("Control signal changed, starting MPC")
                print(f"ctrl: {ctrl}")
                for i in range(self.config.nu_real):
                    self.low_cmd_msg.motor_cmd[i].q = self.config._targetPos_2[i] 
                    self.low_cmd_msg.motor_cmd[i].dq = 0
                    self.low_cmd_msg.motor_cmd[i].kp = self.kp[i]
                    self.low_cmd_msg.motor_cmd[i].kd = self.kd[i]
                    self.low_cmd_msg.motor_cmd[i].tau = 0
                self.low_cmd_msg.crc = self.crc.Crc(self.low_cmd_msg)
                self.low_cmd_publisher.Write(self.low_cmd_msg)
                
                break
            time.sleep(0.002)

        

    def finish_stand(self):
        # Store current position as start position
        for i in range(self.config.nu_real):
            self.config.startPos[i] = self.low_state.motor_state[i].q
            
        # Single transition percentage
        percent = 1.0
        
        while percent > 0:
            time.sleep(0.002)
            # Setup command header
            self.low_cmd_msg.head[0] = 0xFE
            self.low_cmd_msg.head[1] = 0xEF
            self.low_cmd_msg.level_flag = 0xFF
            self.low_cmd_msg.gpio = 0

            # Update transition percentage
            percent -= 1.0 / self.config.duration_2  # Using duration_2 as the transition time
            percent = max(percent, 0)
            
            # Direct interpolation from current position to final target
            for i in range(self.config.nu_real):
                self.low_cmd_msg.motor_cmd[i].mode = 0x01
                self.low_cmd_msg.motor_cmd[i].q = (1 - percent) * self.config._targetPos_1[i] + percent * self.config.startPos[i]
                self.low_cmd_msg.motor_cmd[i].dq = 0
                self.low_cmd_msg.motor_cmd[i].kp = self.kp[i]
                self.low_cmd_msg.motor_cmd[i].kd = self.kd[i]
                self.low_cmd_msg.motor_cmd[i].tau = 0

            # Send command
            self.low_cmd_msg.crc = self.crc.Crc(self.low_cmd_msg)
            self.low_cmd_publisher.Write(self.low_cmd_msg)

    def main_loop(self):
        First_signal_flag = False
        t0 = time.time()
        rate_limiter = RateLimiter(frequency=1 / self.config.ctrl_dt)
        print("Starting main loop")
        try:
            with mujoco.viewer.launch_passive(
                self.mj_model, self.mj_data, show_left_ui=True, show_right_ui=False
            ) as viewer:
                # self.fix_position_as_init()
                while viewer.is_running():
                    # Read control inputs from shared memory
                    _, ctrl = unpack_control_data(self.ctrl_buffer, self.config.nu_real)
                    # print the ctrl
                    
                    # Read mocap state from shared memory
                    q_mocap, qd_mocap = unpack_mocap_data(self.mocap_buffer)
                    
                    self.q[:7] = q_mocap[:7]
                    if self.config.use_mocap_ang_vel:
                        self.qd[:6] = qd_mocap[:6]
                    else:
                        self.qd[:3] = qd_mocap[:3]
                    

                    self.low_cmd_msg.crc = self.crc.Crc(self.low_cmd_msg)
                    self.low_cmd_publisher.Write(self.low_cmd_msg)

                    # Write the state to shared memory
                    t_real = time.time() - t0
                    self.state_buffer[:] = pack_state_data(
                        self.state_buffer, t_real, self.q, self.qd
                    )

                    self.mj_data.qpos = self.q
                    self.mj_data.qvel = self.qd
                    mujoco.mj_kinematics(self.mj_model, self.mj_data)

                    # Update the viewer
                    viewer.sync()
                    if np.allclose(ctrl, np.zeros(self.config.nu_real), atol=1e-3):
                        for i in range(self.config.nu_real):
                            self.low_cmd_msg.motor_cmd[i].q = self.config._targetPos_2[i] 
                            self.low_cmd_msg.motor_cmd[i].dq = 0
                            self.low_cmd_msg.motor_cmd[i].kp = self.kp[i]
                            self.low_cmd_msg.motor_cmd[i].kd = self.kd[i]
                            self.low_cmd_msg.motor_cmd[i].tau = 0
                        self.low_cmd_msg.crc = self.crc.Crc(self.low_cmd_msg)
                        self.low_cmd_publisher.Write(self.low_cmd_msg)
                        # skip the rest of the loop
                        continue
                    if not First_signal_flag:
                        First_signal_flag = True
                        initial_ctrl = ctrl
                    if np.allclose(ctrl, initial_ctrl, atol=1e-3):
                        print("Control signal is approximately initial")
                        for i in range(self.config.nu_real):
                            self.low_cmd_msg.motor_cmd[i].q = self.config._targetPos_2[i] 
                            self.low_cmd_msg.motor_cmd[i].dq = 0
                            self.low_cmd_msg.motor_cmd[i].kp = self.kp[i]
                            self.low_cmd_msg.motor_cmd[i].kd = self.kd[i]
                            self.low_cmd_msg.motor_cmd[i].tau = 0
                        self.low_cmd_msg.crc = self.crc.Crc(self.low_cmd_msg)
                        self.low_cmd_publisher.Write(self.low_cmd_msg)
                        continue
                    print(f"ctrl: {ctrl}")
                    
                    ctrl_with_gear = ctrl * self.gear_array
                    # Publish control via Unitree SDK2
                    print(f"ctrl_with_gear: {ctrl_with_gear}")
                    print(f"gear_array: {self.gear_array}")
                    for idx in range(self.config.nu_real):
                        self.low_cmd_msg.motor_cmd[idx].mode = 0x01  # Set appropriate mode
                        self.low_cmd_msg.motor_cmd[idx].q = 0.0
                        self.low_cmd_msg.motor_cmd[idx].dq = 0.0
                        self.low_cmd_msg.motor_cmd[idx].tau = ctrl_with_gear[idx] 
                        self.low_cmd_msg.motor_cmd[idx].kp = 0.0
                        self.low_cmd_msg.motor_cmd[idx].kd = 3.0

                    # rate_limiter.sleep()
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
            
        finally:
            # Clean up shared memory
            self.finish_stand()
            self.state_shm.close()
            self.state_shm.unlink()
            self.ctrl_shm.close()
            self.ctrl_shm.unlink()
            self.mocap_shm.close()
            self.mocap_shm.unlink()


if __name__ == "__main__":
    viz = Real(robot_name="go2")
    viz.init_stand()
    # viz.fix_position_as_init()
    viz.main_loop()
    viz.finish_stand()