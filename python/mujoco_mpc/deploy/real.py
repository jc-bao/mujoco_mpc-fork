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
import xml.etree.ElementTree as ET
# Unitree SDK2
import sys
from unitree_sdk2py.core.channel import (
    ChannelPublisher,
    ChannelSubscriber,
    ChannelFactoryInitialize,
)

from unitree_sdk2py.idl.default import (
    unitree_hg_msg_dds__LowCmd_,
)
from unitree_sdk2py.idl.default import(
    unitree_go_msg_dds__LowCmd_,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmd_hg
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmd_go
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowState_hg
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowState_go
from unitree_sdk2py.utils.crc import CRC

from config import G1Config, Go2Config
from utils import unpack_control_data, pack_state_data, unpack_mocap_data

class Real:
    def __init__(self, robot_name="g1"):
        print(robot_name)
        if robot_name == "g1":
            self.config = G1Config()
            self.ctrl_msg = unitree_hg_msg_dds__LowCmd_()
            self.low_cmd_msg = unitree_hg_msg_dds__LowCmd_()
            print('imported G1Config')
        elif robot_name == "go2":
            self.config = Go2Config()
            self.ctrl_msg = unitree_go_msg_dds__LowCmd_()
            self.low_cmd_msg = unitree_go_msg_dds__LowCmd_()
            print('imported Go2Config')
        else:
            raise ValueError(f"Robot {robot_name} not supported")

        # Mujoco related
        self.mj_model = mujoco.MjModel.from_xml_path(self.config.xml_path_sim)
        self.mj_model.opt.timestep = self.config.dt_real
        self.mj_data = mujoco.MjData(self.mj_model)
        self.q = self.mj_model.key_qpos[0].copy()
        self.qd = self.mj_model.key_qvel[0].copy()
        self.motor_gears = self._read_motor_gears(self.config.xml_path_ctrl)
        self.gear_array = np.ones(self.mj_model.nu)

        # Update the array with gear values at the corresponding joint indices
        for motor in self.motor_gears.values():
            joint_index = motor['joint_index'] - 1
            gear = motor['gear']
            print(f"Joint index: {joint_index}, Gear: {gear}")
            self.gear_array[joint_index] *= gear

        # Control message (send to robot)
        self.ctrl_msg.mode_pr = 0
        self.ctrl_msg.mode_machine = 3

        # Shared Memory for control inputs (read from controller)
        self.ctrl_shm_size = 8 + self.config.nu_real * 8  # time + tau + q + qd
        self.ctrl_shm = shared_memory.SharedMemory(
            name="ctrl_shm", create=True, size=self.ctrl_shm_size
        )
        self.ctrl_buffer = self.ctrl_shm.buf

        # Shared Memory for state variables (send to controller)
        self.state_shm_size = (
            8 + self.config.nq_real * 8 + self.config.nqd_real * 8
        )  # time + q + qd
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

        # Initialize Unitree SDK2
        ChannelFactoryInitialize(0, "en0")
        if robot_name == "g1":
            self.low_state_subscriber = ChannelSubscriber("rt/lowstate", LowState_hg)
        elif robot_name == "go2":
            self.low_state_subscriber = ChannelSubscriber("rt/lowstate", LowState_go)
        self.low_state_subscriber.Init(self.low_state_handler, 10)

        if robot_name == "g1":
            self.low_cmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_hg)
        elif robot_name == "go2":
            self.low_cmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_go)
        self.low_cmd_publisher.Init()
        self.low_cmd_msg.mode_pr = 0
        self.low_cmd_msg.mode_machine = 3
        self.crc = CRC()

        # Proprioception subscriber
        if robot_name == "g1":
            self.low_sub = ChannelSubscriber("rt/lowstate", LowState_hg)
        elif robot_name == "go2":
            self.low_sub = ChannelSubscriber("rt/lowstate", LowState_go)
        self.low_sub.Init(self.low_state_handler, 1)

    def _read_motor_gears(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        motor_gears = {}
        for motor in root.findall('.//motor'):
            name = motor.get('name')
            gear = motor.get('gear')
            joint_name = motor.get('joint')
            if name and gear and joint_name:
                # Find the joint index in the MuJoCo model
                joint_index = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                motor_gears[name] = {
                    'gear': float(gear),
                    'joint_index': joint_index
                }
        return motor_gears

    def low_state_handler(self, msg: LowState_):
        for i in range(self.config.nq_real-7):
            self.q[7 + i] = msg.motor_state[i].q
            self.qd[6 + i] = msg.motor_state[i].dq
        if not self.config.use_mocap_ang_vel:
            omega = np.array([msg.imu_state.gyroscope]).flatten()
            self.qd[6:9] = omega

    def main_loop(self):
        t0 = time.time()
        rate_limiter = RateLimiter(frequency=1 / self.config.ctrl_dt)
        
        try:
            with mujoco.viewer.launch_passive(
                self.mj_model, self.mj_data, show_left_ui=True, show_right_ui=False
            ) as viewer:
                while viewer.is_running():
                    # Read control inputs from shared memory
                    _, ctrl = unpack_control_data(self.ctrl_buffer, self.config.nu_real)

                    # Read mocap state from shared memory
                    q_mocap, qd_mocap = unpack_mocap_data(self.mocap_buffer)
                    self.q[:7] = q_mocap[:7]
                    if self.config.use_mocap_ang_vel:
                        self.qd[:6] = qd_mocap[:6]
                    else:
                        self.qd[:3] = qd_mocap[:3]
                    
                    print('gear', self.gear_array)
                    ctrl = ctrl * self.gear_array
                    print('control', ctrl)
                    # Publish control via Unitree SDK2
                    for idx in range(self.config.nu_real):
                        self.low_cmd_msg.motor_cmd[
                            idx
                      ].mode = 1  # Set appropriate mode
                        self.low_cmd_msg.motor_cmd[idx].q = 0.0
                        self.low_cmd_msg.motor_cmd[idx].dq = 0.0
                        self.low_cmd_msg.motor_cmd[idx].tau = ctrl[idx]
                        self.low_cmd_msg.motor_cmd[idx].kp = 0.0
                        self.low_cmd_msg.motor_cmd[idx].kd =0.0
                    print('locked joint idx', self.config.locked_joint_idx)
                        
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

                    rate_limiter.sleep()
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
        finally:
            # Clean up shared memory
            self.state_shm.close()
            self.state_shm.unlink()
            self.ctrl_shm.close()
            self.ctrl_shm.unlink()
            self.mocap_shm.close()
            self.mocap_shm.unlink()


if __name__ == "__main__":
    viz = Real(robot_name="go2")
    viz.main_loop()
