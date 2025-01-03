import numpy as np
import struct
from multiprocessing import shared_memory
from loop_rate_limiters import RateLimiter
import pathlib
import time
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import mujoco
from mujoco_mpc import agent as agent_lib

from config import G1PositionConfig, Go2PositionConfig, QuadrupedConfig, G1FixedConfig, H1_2PositionConfig
from utils import (
    pack_control_data,
    unpack_mocap_data,
    unpack_state_data,
    ctrl_sim2real,
    state_real2sim,
    create_bar
)

# Unitree SDK2
import sys
from unitree_sdk2py.core.channel import (
    ChannelPublisher,
    ChannelSubscriber,
    ChannelFactoryInitialize,
)

# from unitree_sdk2py.idl.default import (
#     unitree_hg_msg_dds__LowCmd_,
#     unitree_hg_msg_dds__LowState_,
#     unitree_hg_msg_dds__MotorCmd_,
# )
# from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ 
# from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.go2.robot_state.robot_state_client import RobotStateClient


class Controller:
    def __init__(self, robot_name="g1"):
        self.act_time = time.time()
        self.firstRun = True
        self.robot_name = robot_name
        if robot_name == "g1":
            self.config = G1PositionConfig()
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ 
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        elif robot_name == "g1_fixed":
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ 
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
            self.config = G1FixedConfig()
        elif robot_name == "go2":
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ 
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
            from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
            self.config = Go2PositionConfig()
        elif robot_name == "quadruped":
            self.config = QuadrupedConfig()
        elif robot_name == "h1_2":
            self.config = H1_2PositionConfig()
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ 
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        else:
            raise ValueError(f"Robot {robot_name} not supported")

        # Mocap variables read from real robot
        try:
            self.mocap_shm = shared_memory.SharedMemory(name="mocap_state_shm")
            self.mocap_buffer = self.mocap_shm.buf
        except FileNotFoundError:
            print("Could not create mocap shared memory 'mocap_state_shm'.")
            exit()

        # filter variables
        self.low_pass_filter_gamma = 1.0
        self.low_pass_filter_window_size = 1
        self.qd_buffer = np.zeros((self.low_pass_filter_window_size, self.config.nqd_ctrl))

        # Initialize state variables
        self.global_ctrl_scale = 0.0
        self.global_kp_scale = 0.0
        self.global_kd_scale = 1.0
        self.global_z_offset = 0.01
        self.global_rpy_offset = np.array([0.017, -0.008, 0.0])
        self.q = np.zeros(self.config.nq_real)
        self.qd = np.zeros(self.config.nqd_real)
        # Initialize control variables
        self.last_plan_time = 0.0
        self.state = None  # Will be initialized in main_loop
        if robot_name == "g1" or robot_name == "g1_fixed":
            self.low_cmd_msg = unitree_hg_msg_dds__LowCmd_()
        elif robot_name == "go2":
            self.low_cmd_msg = unitree_go_msg_dds__LowCmd_()
        elif robot_name == "h1_2":
            self.low_cmd_msg = unitree_hg_msg_dds__LowCmd_()
        self.low_cmd_msg.mode_pr = 0
        self.low_cmd_msg.mode_machine = 6
        self.crc = CRC()
        # Initialize Unitree SDK2
        ChannelFactoryInitialize(0, "en0")
        # disable sport mode for go2
        if robot_name == "go2":
            rsc = RobotStateClient()
            rsc.SetTimeout(3.0)
            rsc.Init()
            code = rsc.ServiceSwitch("sport_mode", False) 
            if code != 0:
                print("[ERROR] service stop sport_mode error. code:", code)
            else:
                print("[INFO] service stop sport_mode success. code:", code)
        # self.InitLowCmd()
        self.low_state_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        print("init low state subscriber")
        self.low_state_subscriber.Init(self.low_state_handler, 10)
        print("init low state subscriber done")
        self.low_cmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.low_cmd_publisher.Init()


    def low_state_handler(self, msg):
        self.low_state = msg
        for i in range(self.config.nq_real - 7):
            self.q[7 + i] = msg.motor_state[i].q
            self.qd[6 + i] = msg.motor_state[i].dq
        if not self.config.use_mocap_ang_vel:
            omega = np.array([msg.imu_state.gyroscope]).flatten()
            self.qd[6:9] = omega

    def init_stand_g1(self):
        rate_limiter = RateLimiter(frequency=1 / 0.02)
        self.config.startPos = np.zeros(self.config.nu_real)
        for i in range(self.config.nu_real):
            self.config.startPos[i] = self.q[7 + i]
        self.config.percent_1 = 0
        while self.config.percent_1 < 1:

            print("percent_1", self.config.percent_1)
            self.config.percent_1 += 1.0 / self.config.duration_1
            self.config.percent_1 = min(self.config.percent_1, 1)
            if self.config.percent_1 < 1:
                for i in range(self.config.nu_real):
                    self.low_cmd_msg.mode_pr = 0
                    self.low_cmd_msg.mode_machine = 6
                    self.low_cmd_msg.motor_cmd[i].mode = 1  # 1:Enab
                    self.low_cmd_msg.motor_cmd[i].q = (
                        1 - self.config.percent_1
                    ) * self.config.startPos[
                        i
                    ] + self.config.percent_1 * self.config._targetPos_1[
                        i
                    ]
                    self.low_cmd_msg.motor_cmd[i].dq = 0
                    self.low_cmd_msg.motor_cmd[i].kp = self.config.kp_real[i]
                    self.low_cmd_msg.motor_cmd[i].kd = self.config.kd_real[i] 
                    self.low_cmd_msg.motor_cmd[i].tau = 0
                self.low_cmd_msg.crc = self.crc.Crc(self.low_cmd_msg)
                self.low_cmd_publisher.Write(self.low_cmd_msg)
            rate_limiter.sleep()
        print("Stand up complete")

    def init_stand_h1_2(self):
        rate_limiter = RateLimiter(frequency=1 / 0.02)
        self.config.startPos = np.zeros(self.config.nu_real)
        for i in range(self.config.nu_real):
            self.config.startPos[i] = self.q[7 + i]
        self.config.percent_1 = 0
        while self.config.percent_1 < 1:

            print("percent_1", self.config.percent_1)
            self.config.percent_1 += 1.0 / self.config.duration_1
            self.config.percent_1 = min(self.config.percent_1, 1)
            if self.config.percent_1 < 1:
                for i in range(self.config.nu_real):
                    self.low_cmd_msg.mode_pr = 0
                    self.low_cmd_msg.mode_machine = 6
                    self.low_cmd_msg.motor_cmd[i].mode = 1  # 1:Enab
                    self.low_cmd_msg.motor_cmd[i].q = (
                        1 - self.config.percent_1
                    ) * self.config.startPos[
                        i
                    ] + self.config.percent_1 * self.config._targetPos_1[
                        i
                    ]
                    self.low_cmd_msg.motor_cmd[i].dq = 0
                    self.low_cmd_msg.motor_cmd[i].kp = self.config.kp_real[i]
                    self.low_cmd_msg.motor_cmd[i].kd = self.config.kd_real[i] 
                    self.low_cmd_msg.motor_cmd[i].tau = 0
                self.low_cmd_msg.crc = self.crc.Crc(self.low_cmd_msg)
                self.low_cmd_publisher.Write(self.low_cmd_msg)
            rate_limiter.sleep()
        print("Stand up complete")



    def init_stand_go2(self):
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
                    self.low_cmd_msg.motor_cmd[i].kp = self.config.kp_real[i]
                    self.low_cmd_msg.motor_cmd[i].kd = self.config.kd_real[i]
                    self.low_cmd_msg.motor_cmd[i].tau = 0

            if (self.config.percent_1 == 1) and (self.config.percent_2 <= 1):
                self.config.percent_2 += 1.0 / self.config.duration_2
                self.config.percent_2 = min(self.config.percent_2, 1)
                for i in range(self.config.nu_real):
                    self.low_cmd_msg.motor_cmd[i].q = (1 - self.config.percent_2) * self.config._targetPos_1[i] + self.config.percent_2 * self.config._targetPos_2[i]
                    self.low_cmd_msg.motor_cmd[i].dq = 0
                    self.low_cmd_msg.motor_cmd[i].kp = self.config.kp_real[i]
                    self.low_cmd_msg.motor_cmd[i].kd = self.config.kd_real[i]
                    self.low_cmd_msg.motor_cmd[i].tau = 0

            if self.config.control_mode == "torque":
                # laydown in torque mode
                if (self.config.percent_1 == 1) and (self.config.percent_2 == 1) and (self.config.percent_3 < 1):
                    self.config.percent_3 += 1.0 / self.config.duration_3
                    self.config.percent_3 = min(self.config.percent_3, 1)
                    for i in range(self.config.nu_real):
                        self.low_cmd_msg.motor_cmd[i].q = self.config._targetPos_2[i] 
                        self.low_cmd_msg.motor_cmd[i].dq = 0
                        self.low_cmd_msg.motor_cmd[i].kp = self.config.kp_real[i]
                        self.low_cmd_msg.motor_cmd[i].kd = self.config.kd_real[i]
                        self.low_cmd_msg.motor_cmd[i].tau = 0

            
            
            self.low_cmd_msg.crc = self.crc.Crc(self.low_cmd_msg)
            self.low_cmd_publisher.Write(self.low_cmd_msg)
            # print('standing',self.config.percent_3)

            if self.config.control_mode == "position" and np.isclose(self.config.percent_1, 1):
                break
        print("Stand up complete")

    def set_action(self, ctrl):
        if self.config.control_mode == "torque":
            tau = ctrl * self.global_ctrl_scale
            q_des = np.zeros(self.config.nu_real)
        elif self.config.control_mode == "position":
            tau = np.zeros(self.config.nu_real)
            q_des = ctrl * self.global_ctrl_scale + (1 - self.global_ctrl_scale) * self.config.q_default
        self.low_cmd_msg.mode_pr = 0
        self.low_cmd_msg.mode_machine = 6
        for i in range(self.config.nu_real):
            self.low_cmd_msg.motor_cmd[i].mode = 1  # 1:Enab
            self.low_cmd_msg.motor_cmd[i].dq = 0
        if np.allclose(ctrl, np.zeros(self.config.nu_real), atol=1e-3):
            print("Control disabled")
            ctrl = np.zeros(self.config.nu_real)
            for i in range(self.config.nu_real):
                self.low_cmd_msg.motor_cmd[i].q = self.config.q_default[i]
                self.low_cmd_msg.motor_cmd[i].kp = self.config.kp_real[i]
                self.low_cmd_msg.motor_cmd[i].kd = self.config.kd_real[i] 
                self.low_cmd_msg.motor_cmd[i].tau = 0
        else:
            for i in range(self.config.nu_real):
                if i in self.config.locked_joint_idx:
                    self.low_cmd_msg.motor_cmd[i].q = self.config._targetPos_1[i]
                    self.low_cmd_msg.motor_cmd[i].tau = 0.0
                    self.low_cmd_msg.motor_cmd[i].kp = self.config.kp_real[i]
                    self.low_cmd_msg.motor_cmd[i].kd = self.config.kd_real[i]
                else:
                    self.low_cmd_msg.motor_cmd[i].q = q_des[i]
                    self.low_cmd_msg.motor_cmd[i].tau = tau[i]
                    self.low_cmd_msg.motor_cmd[i].kp = self.config.kp_real[i] * self.global_kp_scale
                    self.low_cmd_msg.motor_cmd[i].kd = self.config.kd_real[i] * self.global_kd_scale
        self.low_cmd_msg.crc = self.crc.Crc(self.low_cmd_msg)
        self.low_cmd_publisher.Write(self.low_cmd_msg)
        # print the time in ms
        action_duration = (time.time() - self.act_time) * 1000
        print(f"Action duration: {action_duration:.2f} ms")
        self.act_time = time.time()

    def get_state(self):
        q_mocap, qd_mocap = unpack_mocap_data(self.mocap_buffer)
        self.q[:7] = q_mocap
        if self.config.use_mocap_ang_vel:
            self.qd[:6] = qd_mocap
        else:
            self.qd[:3] = qd_mocap[:3]
        q_sim, qd_sim = state_real2sim(
            self.q,
            self.qd,
            self.config.locked_joint_idx,
            self.config.nq_ctrl,
            self.config.nqd_ctrl,
            self.config.nq_real,
            self.config.nqd_real,
        )
        # add z offset
        q_sim[2] += self.global_z_offset
        # add rpy offset
        # check if q_sim is zero norm, if so, set it to identity
        if np.linalg.norm(q_sim[3:7]) < 1e-6:
            q_sim[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
            print("[WARNING] q_sim is zero norm, setting it to identity")
        rot_robot = R.from_quat(q_sim[3:7], scalar_first=True)
        rot_body = R.from_euler("xyz", self.global_rpy_offset, degrees=False)
        rot_robot_new = rot_robot * rot_body
        q_sim[3:7] = rot_robot_new.as_quat(scalar_first=True)
        return q_sim, qd_sim

    def main_loop(self):
        # Create the GUI elements
        params_dict = {
            "ctrl_scale": {"lower": 0, "upper": 1.0, "step": 0.01, "default": 0.0},
            "kp_scale": {"lower": 0.0, "upper": 2.0, "step": 0.01, "default": 1.0},
            "kd_scale": {"lower": 0.5, "upper": 2.0, "step": 0.01, "default": 1.0},
            "z_offset": {"lower": -0.1, "upper": 0.1, "step": 0.001, "default": 0.005},
            "roll_offset": {"lower": -0.1, "upper": 0.1, "step": 0.001, "default": 0.013},
            "pitch_offset": {"lower": -0.1, "upper": 0.1, "step": 0.001, "default": -0.009},
        }
        root, bars, update_gui = create_bar(params_dict)
        # Controller
        model = mujoco.MjModel.from_xml_path(self.config.xml_path_ctrl)
        rate_limiter = RateLimiter(frequency=1 / self.config.dt_ctrl)
        try:
            with agent_lib.Agent(
                server_binary_path=pathlib.Path(agent_lib.__file__).parent
                / "mjpc"
                / "ui_agent_server",
                task_id=self.config.task_id,
                model=model,
            ) as agent:
                first_ctrl = agent.get_action()
                print("first_ctrl", first_ctrl)
                if self.robot_name == "h1_2":
                    self.init_stand_h1_2()
                while True:
                    update_gui()
                    self.global_ctrl_scale = bars["ctrl_scale"].get()
                    self.global_kp_scale = bars["kp_scale"].get()
                    self.global_kd_scale = bars["kd_scale"].get()
                    self.global_z_offset = bars["z_offset"].get()
                    self.global_rpy_offset = np.array([bars["roll_offset"].get(), bars["pitch_offset"].get(), 0.0])
                    q_sim, qd_sim = self.get_state()

                    # DEBUG: test fixed base mode
                    if self.config.task_id == "G1 Fixed":
                        q_sim = q_sim[7:]
                        qd_sim = qd_sim[6:]
                    

                    # low pass filter
                    qd_sim_lp = self.low_pass_filter_gamma * qd_sim + (1 - self.low_pass_filter_gamma) * self.qd_buffer[-1]
                    self.qd_buffer = np.roll(self.qd_buffer, -1, axis=0)
                    self.qd_buffer[-1, :] = qd_sim_lp
                    self.qd_buffer[-1, :] = np.mean(self.qd_buffer, axis=0)
                    qd_sim = self.qd_buffer[-1, :]

                    agent.set_state(qpos=q_sim, qvel=qd_sim)
                    ctrl = agent.get_action()
                    # print("ctrl", ctrl)
                    if np.allclose(ctrl, first_ctrl, atol=1e-3):
                        print("Control disabled")
                        ctrl = np.zeros_like(first_ctrl)
                    ctrl_real = ctrl_sim2real(
                        ctrl * self.config.gear_real,
                        self.config.locked_joint_idx,
                        self.config.nu_real,
                    )
                    self.set_action(ctrl_real) 

        except KeyboardInterrupt:
            print("Keyboard interrupt detected. Exiting...")
        finally:
            self.mocap_shm.close()
            root.destroy()

if __name__ == "__main__":
    controller = Controller(robot_name="h1_2")
    # controller.init_stand_go2() 
    # controller.init_stand_h1_2()
    controller.main_loop()