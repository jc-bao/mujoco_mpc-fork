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

from config import G1Config, Go2Config, QuadrupedConfig, G1FixedConfig
from utils import pack_state_data, unpack_control_data, apply_gear_to_control, ctrl_real2sim, ctrl_sim2real

plt.style.use(["science"])


class Sim:
    def __init__(self, robot_name="g1"):
        if robot_name == "g1":
            self.config = G1Config()
        elif robot_name == "g1_fixed":
            self.config = G1FixedConfig()
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

   

    def main_loop(self):

        rate_limiter = RateLimiter(frequency=1 / self.config.dt_sim / self.config.real_time_factor)
        with mujoco.viewer.launch_passive(
            self.mj_model, self.mj_data, show_left_ui=True, show_right_ui=False
        ) as viewer:
            
            
            site_names = ['RR', 'Site2', 'Site3']  # Replace with your list of site names

            while True:
                

                mujoco.mj_step(self.mj_model, self.mj_data)
                # DEBUG
                # manually set position and orientation to zero
                # self.mj_data.qpos[:7] = 0
                # self.mj_data.qpos[2] = 1.0
                # self.mj_data.qpos[3] = 1.0
                # self.mj_data.qvel[:6] = 0
                q_sim = self.mj_data.qpos
                qd_sim = self.mj_data.qvel
                # Get site ID
                site_names = ['RR', 'RL', 'FR', 'FL']
                

                for site_name in site_names:
                    try:
                        site_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, site_name)
                        site_position = self.mj_data.site_xpos[site_id]
                        print(f"Position of site '{site_name}': {site_position}")
                    except Exception as e:
                        print(f"Error retrieving site '{site_name}': {e}")

                # Update the viewer
                viewer.sync()
                rate_limiter.sleep()
       


if __name__ == "__main__":
    sim = Sim(robot_name="g1_fixed")
    sim.main_loop()
