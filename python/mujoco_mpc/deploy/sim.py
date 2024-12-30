import os
import time
import mujoco
import mujoco.viewer
import numpy as np
import argparse
import matplotlib.pyplot as plt
import scienceplots
from scipy.spatial.transform import Rotation as R
import time
from copy import deepcopy
from multiprocessing import shared_memory
import struct
from loop_rate_limiters import RateLimiter
import xml.etree.ElementTree as ET

from config import G1Config, Go2Config, QuadrupedConfig
from utils import pack_state_data, unpack_control_data, apply_gear_to_control, ctrl_real2sim, ctrl_sim2real

plt.style.use(["science"])


class Sim:
    def __init__(self, robot_name="g1"):
        if robot_name == "g1":
            self.config = G1Config()
        elif robot_name == "go2":
            self.config = Go2Config()
        elif robot_name == "quadruped":
            self.config = QuadrupedConfig()
        else:
            raise ValueError(f"Robot {robot_name} not supported")

        # MuJoCo model setup
        
        self.mj_model = mujoco.MjModel.from_xml_path(self.config.xml_path_sim)
        self.mj_model.opt.timestep = self.config.dt_sim
        self.mj_data = mujoco.MjData(self.mj_model)
        mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, 0)
        self.n_sim_frame = int(self.config.dt_ctrl / self.config.dt_sim)
        self.default_ctrl = self.mj_model.key_ctrl[0].copy()  # default control is the ctrl in key frame
        print(f"Default control: {self.default_ctrl}")
        # exit()
        assert np.isclose(self.config.dt_ctrl, self.n_sim_frame * self.config.dt_sim), "Control timestep must be an integer multiple of simulation timestep"

        # Initialize state variables
        self.q = self.mj_model.key_qpos[0].copy()
        self.qd = self.mj_model.key_qvel[0].copy()
        assert self.config.nq_real == self.mj_model.nq, "Number of joints in MuJoCo model must match the number of joints in the configuration"
        # Print motor information and names
        print(f"Available attributes in MjModel: {dir(self.mj_model)}")
        print(f"Motor names: {self.mj_model.actuator_ctrlrange}")
        # Read motor gear values from XML
        self.motor_gears = self._read_motor_gears(self.config.xml_path_ctrl)
        # print motor gears
        # print(f"Motor gears: {self.motor_gears}")

        # Initialize an array with ones
        self.gear_array = np.ones(self.mj_model.nu)

        # Update the array with gear values at the corresponding joint indices
        for motor in self.motor_gears.values():
            joint_index = motor['joint_index'] - 1
            gear = motor['gear']
            print(f"Joint index: {joint_index}, Gear: {gear}")
            self.gear_array[joint_index] *= gear


        # Print the gear array for verification
        print(f"Gear array: {self.gear_array}")

        # Shared Memory for control inputs
        self.ctrl_shm_size = (self.config.nu_real + 1) * 8  # time + q_des
        # If the shared memory already exists, delete it
        
        
        self.ctrl_shm = shared_memory.SharedMemory(
            name="ctrl_shm", create=True, size=self.ctrl_shm_size
        )
        self.ctrl_buffer = self.ctrl_shm.buf

        # Shared Memory for state variables
        self.state_shm_size = (1 + self.config.nq_real + self.config.nqd_real) * 8  # time + q + qd
        self.state_shm = shared_memory.SharedMemory(
            name="state_shm", create=True, size=self.state_shm_size
        )
        self.state_buffer = self.state_shm.buf

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

    def main_loop(self):
        try:
            rate_limiter = RateLimiter(frequency=1 / self.config.dt_sim / self.config.real_time_factor)
            with mujoco.viewer.launch_passive(
                self.mj_model, self.mj_data, show_left_ui=True, show_right_ui=False
            ) as viewer:
                while True:
                    # Read control inputs from shared memory
                    _, q_des = unpack_control_data(self.ctrl_buffer, self.config.nu_real)
                    # check if q_des is close to zero
                    if np.allclose(q_des, np.zeros_like(q_des)):
                        q_des = self.default_ctrl
                        print("Resetting control to default")
                    
                    # apply gear to control
                    # q_des_with_gear = apply_gear_to_control(q_des, self.gear_array) 
                
                    # self.mj_data.ctrl[:] = q_des_with_gear 

                    # print(self.config.nu_real)
                    # q_des_sim equals to q_des without locked joints
                    # q_des_sim = ctrl_real2sim(q_des, self.config.locked_joint_idx, 21)
                    # print(q_des_sim.shape)
                    # print(q_des_with_gear)
                    self.mj_data.ctrl[:] = q_des

                    mujoco.mj_step(self.mj_model, self.mj_data)

                    # Get the state from the MuJoCo model
                    self.state_buffer[:] = pack_state_data(self.state_buffer, self.mj_data.time, self.mj_data.qpos, self.mj_data.qvel)

                    # Check if robot failed, if so, reset the simulation
                    if self.config.auto_reset:
                      vec_tar = np.array([0.0, 0.0, 1.0])
                      x_rot_torso = self.mj_data.xquat[1]
                      vec = R.from_quat(x_rot_torso, scalar_first=True).apply(vec_tar)
                      d2upright = np.linalg.norm(vec - vec_tar)
                      if d2upright > 0.8:
                          print(f"Warning: Robot is not upright: {d2upright}")
                          mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, 0)
                          print("Resetting simulation...")

                    # Update the viewer
                    viewer.sync()
                    rate_limiter.sleep()
        finally:
            # Clean up shared memory
            self.state_shm.close()
            self.state_shm.unlink()
            self.ctrl_shm.close()
            self.ctrl_shm.unlink()


if __name__ == "__main__":
    sim = Sim(robot_name="go2")
    sim.main_loop()
