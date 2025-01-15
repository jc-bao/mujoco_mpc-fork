import time
import sys

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
import unitree_legged_const as h1
import numpy as np
import os

import matplotlib.pyplot as plt

# Add ../deploy to sys.path
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/deploy"
sys.path.append(path)
from config import H1Config

H1_NUM_MOTOR = 20

class H1JointIndex:
    # Right leg
    kRightHipYaw = 8
    kRightHipRoll = 0
    kRightHipPitch = 1
    kRightKnee = 2
    kRightAnkle = 11
    # Left leg
    kLeftHipYaw = 7
    kLeftHipRoll = 3
    kLeftHipPitch = 4
    kLeftKnee = 5
    kLeftAnkle = 10

    kWaistYaw = 6

    kNotUsedJoint = 9

    # Right arm
    kRightShoulderPitch = 12
    kRightShoulderRoll = 13
    kRightShoulderYaw = 14
    kRightElbow = 15
    # Left arm
    kLeftShoulderPitch = 16
    kLeftShoulderRoll = 17
    kLeftShoulderYaw = 18
    kLeftElbow = 19

motor_idx_to_ctrl_idx = {
    # left leg
    7: 0,
    3: 1,
    4: 2,
    5: 3,
    10: 4,
    # right leg
    8: 5,
    0: 6,
    1: 7,
    2: 8,
    11: 9,
    # torso
    6: 10,
    # left arm
    16: 11,
    17: 12,
    18: 13,
    19: 14,
    # right arm
    12: 15,
    13: 16,
    14: 17,
    15: 18,
    # not used
    9: -1
}

class Custom:
    def __init__(self):
        data = np.load("h1_data.npz")
        self.q_buffer = data["q"]
        self.qd_buffer = data["qd"]
        self.ctrl_buffer = data["ctrl"]
        self.max_delta_ctrl = 0.1

        self.kp_real = H1Config.kp_real * 1.0
        self.kd_real = H1Config.kd_real * 1.0
        self.ctrl0 = H1Config._targetPos_1

        self.time_ = 0.0
        self.control_dt_ = 0.02
        self.duration_ = self.ctrl_buffer.shape[0] * self.control_dt_
        self.counter_ = 0
        self.kp_low_ = 60.0
        self.kp_high_ = 200.0
        self.kd_low_ = 1.5
        self.kd_high_ = 5.0

        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.InitLowCmd()
        self.low_state = None
        self.crc = CRC()

    def Init(self):
        self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher_.Init()

        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateHandler, 10)

        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()

        status, result = self.msc.CheckMode()
        while result['name']:
            self.msc.ReleaseMode()
            status, result = self.msc.CheckMode()
            time.sleep(1)

    def is_weak_motor(self, motor_index):
        return motor_index in {
            H1JointIndex.kLeftAnkle,
            H1JointIndex.kRightAnkle,
            H1JointIndex.kRightShoulderPitch,
            H1JointIndex.kRightShoulderRoll,
            H1JointIndex.kRightShoulderYaw,
            H1JointIndex.kRightElbow,
            H1JointIndex.kLeftShoulderPitch,
            H1JointIndex.kLeftShoulderRoll,
            H1JointIndex.kLeftShoulderYaw,
            H1JointIndex.kLeftElbow,
        }

    def InitLowCmd(self):
        self.low_cmd.head[0] = 0xFE
        self.low_cmd.head[1] = 0xEF
        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0
        for i in range(H1_NUM_MOTOR):
            if self.is_weak_motor(i):
                self.low_cmd.motor_cmd[i].mode = 0x01
            else:
                self.low_cmd.motor_cmd[i].mode = 0x0A
            self.low_cmd.motor_cmd[i].q = h1.PosStopF
            self.low_cmd.motor_cmd[i].kp = 0
            self.low_cmd.motor_cmd[i].dq = h1.VelStopF
            self.low_cmd.motor_cmd[i].kd = 0
            self.low_cmd.motor_cmd[i].tau = 0

    def LowStateHandler(self, msg: LowState_):
        self.low_state = msg

    def main_loop(self):
        init_time = 2.0
        move_time = self.duration_
        total_time = init_time + move_time
        current_time = total_time + 1.0
        init_pos = np.zeros(H1_NUM_MOTOR)
        qreal_buffer = np.zeros((self.ctrl_buffer.shape[0], 10))
        # fig, axs = plt.subplots(3, 4)
        # axs = axs.flatten()
        # lines = []
        # for i in range(10):
        #     axs[i].plot(self.q_buffer[:, i], '--')
        #     lines.append(axs[i].plot(qreal_buffer[:, i])[0])
        #     axs[i].set_title(f"joint {i}")
        # plt.ion()
        last_ctrl = np.zeros(10)
        while True:
            if current_time > total_time:
                init_pos = [self.low_state.motor_state[i].q for i in range(H1_NUM_MOTOR)]
                current_time = 0.0
                qreal_buffer = np.zeros((self.ctrl_buffer.shape[0], 10))
                last_ctrl = np.zeros(10)
            if current_time < init_time:
                # init standing
                ratio = current_time / init_time
                for i in range(H1_NUM_MOTOR):
                    ctrl_idx = motor_idx_to_ctrl_idx[i]
                    self.low_cmd.motor_cmd[i].q = init_pos[i] + ratio * (0.0 - init_pos[i])
                    self.low_cmd.motor_cmd[i].dq = 0.0
                    self.low_cmd.motor_cmd[i].kp = self.kp_real[ctrl_idx]
                    self.low_cmd.motor_cmd[i].kd = self.kd_real[ctrl_idx]
                    self.low_cmd.motor_cmd[i].tau = 0.0
            else:
                time_idx = min(int((current_time - init_time) / self.control_dt_), self.ctrl_buffer.shape[0] - 1)
                for i in range(H1_NUM_MOTOR):
                    ctrl_idx = motor_idx_to_ctrl_idx[i]
                    if ctrl_idx == -1 or ctrl_idx >= 10:
                        ctrl = 0.0
                    else:
                        ctrl = self.ctrl_buffer[time_idx, ctrl_idx]
                        ctrl = np.clip(ctrl, last_ctrl[ctrl_idx] - self.max_delta_ctrl, last_ctrl[ctrl_idx] + self.max_delta_ctrl)
                        last_ctrl[ctrl_idx] = ctrl
                    
                    self.low_cmd.motor_cmd[i].q = ctrl * 0.0
                    self.low_cmd.motor_cmd[i].dq = 0.0
                    self.low_cmd.motor_cmd[i].kp = self.kp_real[ctrl_idx]
                    self.low_cmd.motor_cmd[i].kd = self.kd_real[ctrl_idx]
                    self.low_cmd.motor_cmd[i].tau = 0.0


                    if ctrl_idx >= 0 and ctrl_idx < 10:
                        qreal_buffer[time_idx, ctrl_idx] = self.low_state.motor_state[i].q

            self.low_cmd.crc = self.crc.Crc(self.low_cmd)
            self.lowcmd_publisher_.Write(self.low_cmd)

            current_time += self.control_dt_

            q_robot = [self.low_state.motor_state[i].q for i in range(H1_NUM_MOTOR)]
            print("q_robot", q_robot)

            # update plot
            # for i in range(10):
            #     lines[i].set_ydata(qreal_buffer[:, i])
            # plt.pause(0.001)

            time.sleep(self.control_dt_)

if __name__ == '__main__':
    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    input("Press Enter to continue...")

    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    custom = Custom()
    custom.Init()

    custom.main_loop()

    print("Done!")
