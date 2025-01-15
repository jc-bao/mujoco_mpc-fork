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
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from config import G1PositionConfig, Go2PositionConfig, QuadrupedConfig, G1FixedConfig, H1_2PositionConfig, H1_2_simpleConfig, H1Config
from utils import (
    pack_control_data,
    unpack_mocap_data,
    unpack_state_data,
    ctrl_sim2real,
    state_real2sim,
    create_bar,
    minimize_z_difference, 
    h1_joint_remapping
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
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_

from unitree_sdk2py.utils.crc import CRC



class Controller:
    def __init__(self, robot_name="g1", open_loop_mode=False):
        self.max_delta_ctrl = 0.05
        self.open_loop_mode = open_loop_mode
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
            from unitree_sdk2py.go2.robot_state.robot_state_client import RobotStateClient
            self.config = Go2PositionConfig()
        elif robot_name == "quadruped":
            self.config = QuadrupedConfig()
        elif robot_name == "h1_2":
            self.config = H1_2PositionConfig()
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ 
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        elif robot_name == "h1_2_simple":
            self.config = H1_2_simpleConfig()
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ 
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        elif robot_name == "h1":
            self.config = H1Config()
            from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
            from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
            from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
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
        elif robot_name == "go2" or robot_name == "h1":
            self.low_cmd_msg = unitree_go_msg_dds__LowCmd_()
        elif robot_name == "h1_2" or robot_name == "h1_2_simple":
            self.low_cmd_msg = unitree_hg_msg_dds__LowCmd_()
        self.low_cmd_msg.mode_pr = 0
        self.low_cmd_msg.mode_machine = 6
        self.crc = CRC()
        # Initialize Unitree SDK2
        ChannelFactoryInitialize(0, "enp2s0")
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
        if robot_name == "h1":
            self.disable_sport_mode()

        
        # self.InitLowCmd()
        self.low_state_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        print("init low state subscriber")
        self.low_state_subscriber.Init(self.low_state_handler, 10)
        print("init low state subscriber done")
        self.low_cmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.low_cmd_publisher.Init()

        # mujoco 

    def disable_sport_mode(self):
        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(10.0)
        self.msc.Init()

        status, result = self.msc.CheckMode()
        while result['name']:
            self.msc.ReleaseMode()
            status, result = self.msc.CheckMode()
            
            time.sleep(1)


    def low_state_handler(self, msg: LowState_):
        self.low_state = msg
        
            

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
        
        

    def init_stand_h1(self):
        percent = 0
        start_pos = np.zeros(self.config.nu_real)
        for i in range(self.config.nu_real):
            start_pos[i] = self.low_state.motor_state[i].q
        start_pos_sim = h1_joint_remapping(self.config.motor_order, start_pos, direction="real2sim")
        print("start_pos_sim", start_pos_sim)
        self.low_cmd_msg.head[0] = 0xFE
        self.low_cmd_msg.head[1] = 0xEF
        self.low_cmd_msg.level_flag = 0xFF
        self.low_cmd_msg.gpio = 0
        for i in range(self.config.nu_real):
            # find which motor index corresponds to the joint index
            # if i not in the list then just skip
            if i not in self.config.motor_order:
                continue
            else:
                j = np.where(self.config.motor_order == i)[0][0]
            if j in self.config.weak_motor_idx:
                self.low_cmd_msg.motor_cmd[i].mode = 0x01
            else:
                self.low_cmd_msg.motor_cmd[i].mode = 0x0A
        while percent<1:
            percent += 1.0 / self.config.duration_1
            percent = min(percent, 1)
            cur_pos = percent * self.config.q_default + (1 - percent) * start_pos_sim
            for i in range(self.config.nu_real):
                if i not in self.config.motor_order:
                    continue
                else:
                    j = np.where(self.config.motor_order == i)[0][0]
                
                self.low_cmd_msg.motor_cmd[i].q = cur_pos[j]
                self.low_cmd_msg.motor_cmd[i].tau = 0.0
                self.low_cmd_msg.motor_cmd[i].kp = self.config.kp_real[j] 
                self.low_cmd_msg.motor_cmd[i].kd = self.config.kd_real[j] 
                
            self.low_cmd_msg.crc = self.crc.Crc(self.low_cmd_msg)
            self.low_cmd_publisher.Write(self.low_cmd_msg)
            time.sleep(0.02)
        print("Stand up complete")
        

    def set_action_h1(self, ctrl):
        if self.config.control_mode == "torque":
            tau = ctrl * self.global_ctrl_scale
            q_des = np.zeros(self.config.nu_real)
        elif self.config.control_mode == "position":
            tau = np.zeros(self.config.nu_real)
            q_des = ctrl * self.global_ctrl_scale + (1 - self.global_ctrl_scale) * self.config.q_default
        self.low_cmd_msg.head[0] = 0xFE
        self.low_cmd_msg.head[1] = 0xEF
        self.low_cmd_msg.level_flag = 0xFF
        self.low_cmd_msg.gpio = 0
        for i in range(self.config.nu_real):
            # find which motor index corresponds to the joint index
            # if i not in the list then just skip
            if i not in self.config.motor_order:
                continue
            else:
                j = np.where(self.config.motor_order == i)[0][0]
            if j in self.config.weak_motor_idx:
                self.low_cmd_msg.motor_cmd[i].mode = 0x01
            else:
                self.low_cmd_msg.motor_cmd[i].mode = 0x0A
        if np.allclose(ctrl, np.zeros(self.config.nu_real-1), atol=1e-3):
            for i in range(self.config.nu_real):
                if i not in self.config.motor_order:
                    continue
                else:
                    j = np.where(self.config.motor_order == i)[0][0]
                self.low_cmd_msg.motor_cmd[i].q = self.config.q_default[j]
                self.low_cmd_msg.motor_cmd[i].tau = 0.0
                self.low_cmd_msg.motor_cmd[i].kp = self.config.kp_real[j]
                self.low_cmd_msg.motor_cmd[i].kd = self.config.kd_real[j]
        else:
            for i in range(self.config.nu_real):
                if i not in self.config.motor_order:
                    continue
                else:
                    j = np.where(self.config.motor_order == i)[0][0]
                if i in self.config.locked_joint_idx:
                    self.low_cmd_msg.motor_cmd[i].q = self.config.q_default[j]
                    self.low_cmd_msg.motor_cmd[i].tau = tau[j]
                    self.low_cmd_msg.motor_cmd[i].kp = self.config.kp_real[j] * self.global_kp_scale
                    self.low_cmd_msg.motor_cmd[i].kd = self.config.kd_real[j] * self.global_kd_scale
                else:
                    self.low_cmd_msg.motor_cmd[i].q = q_des[j]
                    self.low_cmd_msg.motor_cmd[i].tau = tau[j]
                    self.low_cmd_msg.motor_cmd[i].kp = self.config.kp_real[j] * self.global_kp_scale
                    self.low_cmd_msg.motor_cmd[i].kd = self.config.kd_real[j] * self.global_kd_scale
        self.low_cmd_msg.crc = self.crc.Crc(self.low_cmd_msg)
        self.low_cmd_publisher.Write(self.low_cmd_msg)

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
                    #fix the print format
                    print(f"Control Command sent to motor index {i}: q_des={q_des[i]}, tau={tau[i]}, kp={self.config.kp_real[i]}, kd={self.config.kd_real[i]}")
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
        q = []
        qd = []
        for i in range(self.config.nu_real):
            # print("q[i]", self.low_state.motor_state[i].q, i)
            q.append(self.low_state.motor_state[i].q)
            qd.append(self.low_state.motor_state[i].dq)
        q = np.array(q)
        qd = np.array(qd)
        if not self.config.use_mocap_ang_vel:
            omega = np.array([self.low_state.imu_state.gyroscope]).flatten()
            self.qd[6:9] = omega
        # if self.robot_name == "h1":
        #     q = h1_joint_remapping(self.config.motor_order, q, direction="real2sim")
        #     qd = h1_joint_remapping(self.config.motor_order, qd, direction="real2sim")
        self.q[7:] = q
        self.qd[6:] = qd
        q_mocap, qd_mocap = unpack_mocap_data(self.mocap_buffer)
        self.q[:7] = q_mocap
        if self.config.use_mocap_ang_vel:
            self.qd[:6] = qd_mocap
        else:
            self.qd[:3] = qd_mocap[:3]
        if self.robot_name=="h1":
            q_jnt = h1_joint_remapping(self.config.motor_order, self.q[7:], direction="real2sim")
            # print("q_jnt", q_jnt)
            qd_jnt = h1_joint_remapping(self.config.motor_order, self.qd[6:], direction="real2sim")
            q = np.concatenate([self.q[:7], q_jnt]).flatten()
            qd = np.concatenate([self.qd[:6], qd_jnt]).flatten()
            q_sim, qd_sim = state_real2sim(
                q,
                qd,
                self.config.locked_joint_idx,
                self.config.nq_ctrl,
                self.config.nqd_ctrl,
                self.config.nq_real - 1,
                self.config.nqd_real - 1,
            )   
        else:

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
        R_world_marker = R.from_quat(q_sim[3:7], scalar_first=True)
        R_marker_body = R.from_euler("xyz", self.global_rpy_offset, degrees=False)
        R_world_body = R_world_marker * R_marker_body
        q_sim[3:7] = R_world_body.as_quat(scalar_first=True)
        # apply change to qd_sim
        world_v_marker = qd_sim[:3]
        world_v_body = R_marker_body.inv().as_matrix() @ world_v_marker
        qd_sim[:3] = world_v_body

        return q_sim, qd_sim

    def main_loop(self):
        # Create the GUI elements
        params_dict = {
            "ctrl_scale": {"lower": 0, "upper": 1.0, "step": 0.01, "default": 0.0},
            "kp_scale": {"lower": 0.0, "upper": 2.0, "step": 0.01, "default": 1.0},
            "kd_scale": {"lower": 0.5, "upper": 2.0, "step": 0.01, "default": 1.0},
            "z_offset": {"lower": -0.1, "upper": 0.1, "step": 0.001, "default": 0.0},
            "roll_offset": {"lower": -0.1, "upper": 0.1, "step": 0.001, "default": 0.0},
            "pitch_offset": {"lower": -0.1, "upper": 0.1, "step": 0.001, "default": 0.0},
            "yaw_offset": {"lower": -3.14, "upper": 3.14, "step": 0.001, "default": 0.0},
            "calibrate": {"lower": 0, "upper": 1, "step": 1.0, "default": 0.0},
        }
        root, bars, update_gui = create_bar(params_dict)
        # Controller
        model = mujoco.MjModel.from_xml_path(self.config.xml_path_ctrl)
        data = mujoco.MjData(model)
        if self.robot_name == "h1_2" or self.robot_name == "h1_2_simple":   
            left_foot_geom_names = ["left_heel_left", "left_heel_right", "left_toe_left", "left_toe_right"]
            right_foot_geom_names = ["right_heel_left", "right_heel_right", "right_toe_left", "right_toe_right"]
            left_foot_geom_idx = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in left_foot_geom_names]
            right_foot_geom_idx = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in right_foot_geom_names]
        elif self.robot_name == "h1":
            foot_geom_names = ["left_heel_left", "left_toe_left", "right_heel_left", "right_toe_left"]
            foot_geom_idx = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in foot_geom_names]
            print("======")
            print("foot_geom_idx", foot_geom_idx)
        elif self.robot_name == "go2":
            foot_geom_names = ["FR", "FL", "HR", "HL"]
            foot_geom_idx = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in foot_geom_names]
        rate_limiter = RateLimiter(frequency=1 / self.config.dt_ctrl)
        counter = 0
        # check if open loop mode
        if self.open_loop_mode:
            # load dumped data
            data = np.load(f"{self.robot_name}_data.npz")
            q_buffer = data["q"]
            qd_buffer = data["qd"]
            ctrl_buffer = data["ctrl"]
            q_real_buffer = np.zeros_like(q_buffer)
            qd_real_buffer = np.zeros_like(qd_buffer)
            # create a matplot figure to update in the real time
            n_ctrl = len(ctrl_buffer)
            n_plot_per_roll = np.sqrt(n_ctrl).astype(int) + 1
            fig, axs = plt.subplots(n_plot_per_roll, n_plot_per_roll)
            axs = axs.flatten()
            # plot q_real_buffer and qd_real_buffer 
            plt.ion()
            lines_real = []
            for i in range(n_ctrl):
                line_real, = axs[i].plot(q_real_buffer[:, 7:])
                axs[i].plot(qd_rq_buffereal_buffer[:, 6:], '--')
                lines_real.append(line_real)
        try:
            with agent_lib.Agent(
                server_binary_path=pathlib.Path(agent_lib.__file__).parent
                / "mjpc"
                / "ui_agent_server",
                task_id=self.config.task_id,
                model=model,
            ) as agent:
                first_ctrl = agent.get_action()
                last_ctrl = first_ctrl
                print("first_ctrl", first_ctrl)
                if self.robot_name == "h1_2" or self.robot_name == "h1_2_simple":
                    self.init_stand_h1_2()
                elif self.robot_name == "h1":
                    self.init_stand_h1()
                while True:
                    update_gui()
                    self.global_ctrl_scale = bars["ctrl_scale"].get()
                    self.global_kp_scale = bars["kp_scale"].get()
                    self.global_kd_scale = bars["kd_scale"].get()
                    self.global_z_offset = bars["z_offset"].get()
                    self.global_rpy_offset = np.array([bars["roll_offset"].get(), bars["pitch_offset"].get(), bars["yaw_offset"].get()])
                    
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

                    # calibration
                    if bars["calibrate"].get() > 0.5:
                        print(f"Calibrating")

                        if self.robot_name == "h1_2" or self.robot_name == "h1_2_simple":
                            data.qpos = q_sim
                            data.qvel = qd_sim
                            data.ctrl = np.zeros(model.nu)
                            mujoco.mj_forward(model, data)

                            left_foot_geom_pos = [data.geom_xpos[idx] for idx in left_foot_geom_idx]
                            left_foot_geom_mat = [data.geom_xmat[idx] for idx in left_foot_geom_idx]
                            right_foot_geom_pos = [data.geom_xpos[idx] for idx in right_foot_geom_idx]
                            right_foot_geom_mat = [data.geom_xmat[idx] for idx in right_foot_geom_idx]

                            left_foot_pos = np.mean(left_foot_geom_pos, axis=0)
                            right_foot_pos = np.mean(right_foot_geom_pos, axis=0)
                            left_foot_mat = left_foot_geom_mat[0]
                            right_foot_mat = right_foot_geom_mat[0]

                            # get mean of xmat for left and right foot
                            foot_mat_mean = (left_foot_mat + right_foot_mat) / 2
                            # create rotation from it
                            foot_rot = R.from_matrix(foot_mat_mean.reshape(3, 3))
                            rot_body_origin = R.from_euler("xyz", self.global_rpy_offset, degrees=False)
                            foot_rot_origin = foot_rot * rot_body_origin.inv()
                            # get rpy with inverse of foot_rot
                            self.global_rpy_offset = foot_rot_origin.inv().as_euler("xyz", degrees=False)
                            # get z offset as negative of the mean of left and right foot up vector
                            foot_pos_mean = (left_foot_pos + right_foot_pos) / 2
                            foot_z = foot_pos_mean[2]
                            foot_z_origin = foot_z - self.global_z_offset
                            self.global_z_offset = -foot_z_origin + 0.005
                            # set bars
                            bars["z_offset"].set(self.global_z_offset)
                            bars["roll_offset"].set(self.global_rpy_offset[0])
                            bars["pitch_offset"].set(self.global_rpy_offset[1])
                            bars["yaw_offset"].set(self.global_rpy_offset[2])
                        elif self.robot_name == "go2" or self.robot_name == "h1":
                            q_sim_marker = q_sim.copy()
                            qd_sim_marker = qd_sim.copy()
                            # undo z offset
                            q_sim_marker[2] -= self.global_z_offset
                            # undo rpy offset 
                            R_world_body = R.from_quat(q_sim[3:7], scalar_first=True)
                            R_marker_body = R.from_euler("xyz", self.global_rpy_offset, degrees=False)
                            R_world_marker = R_world_body * R_marker_body.inv()
                            q_sim_marker[3:7] = R_world_marker.as_quat(scalar_first=True)
                            # undo qd
                            v_world_body = qd_sim[:3]
                            v_marker_body = R_marker_body.as_matrix() @ v_world_body
                            qd_sim_marker[:3] = v_marker_body

                            data.qpos = q_sim_marker
                            data.qvel = qd_sim_marker
                            mujoco.mj_forward(model, data)

                            torso_pos = data.qpos[:3]
                            foot_geom_pos = [data.geom_xpos[idx] - torso_pos for idx in foot_geom_idx]
                            foot_geom_pos = np.array(foot_geom_pos)
                            rot_mat, z_offset = minimize_z_difference(foot_geom_pos)
                            self.global_z_offset = -(z_offset + torso_pos[2]) + 0.010 if self.robot_name == "go2" else 0.005
                            # set bars
                            bars["z_offset"].set(self.global_z_offset)
                            global_rot = R.from_matrix(rot_mat)
                            global_rpy = global_rot.as_euler("xyz", degrees=False)
                            bars["roll_offset"].set(global_rpy[0])
                            bars["pitch_offset"].set(global_rpy[1])
                            bars["yaw_offset"].set(global_rpy[2])
                        else:
                            print(f"Robot {self.robot_name} not supported for calibration")

                    agent.set_state(qpos=q_sim, qvel=qd_sim)
                    ctrl = agent.get_action()
                    ctrl = np.clip(ctrl, last_ctrl - self.max_delta_ctrl, last_ctrl + self.max_delta_ctrl)
                    last_ctrl = ctrl
                    # print("ctrl", ctrl)
                    if np.allclose(ctrl, first_ctrl, atol=1e-3):
                        print("Control disabled")
                        ctrl = np.zeros_like(first_ctrl)
                    ctrl_real = ctrl_sim2real(
                        ctrl * self.config.gear_real,
                        self.config.locked_joint_idx,
                        self.config.nu_real - 1,
                    )
                    if self.robot_name == "h1":
                        self.set_action_h1(ctrl_real)
                    else:
                        self.set_action(ctrl_real) 

                    counter += 1

        except KeyboardInterrupt:
            print("Keyboard interrupt detected. Exiting...")
        finally:
            self.mocap_shm.close()
            root.destroy()

if __name__ == "__main__":
    controller = Controller(robot_name="h1")
    # controller.init_stand_go2() 
    # controller.init_stand_h1_2()
    # controller.init_stand_h1()
    controller.main_loop()