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

from config import G1Config, Go2Config, QuadrupedConfig
from utils import pack_state_data, unpack_control_data

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
        self.default_ctrl = self.mj_data.ctrl
        assert np.isclose(self.config.dt_ctrl, self.n_sim_frame * self.config.dt_sim), "Control timestep must be an integer multiple of simulation timestep"

        # Initialize state variables
        self.q = self.mj_model.key_qpos[0].copy()
        self.qd = self.mj_model.key_qvel[0].copy()
        assert self.config.nq_real == self.mj_model.nq, "Number of joints in MuJoCo model must match the number of joints in the configuration"

        # Shared Memory for control inputs
        self.ctrl_shm_size = (self.config.nu_real + 1) * 8  # time + q_des
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

    def main_loop(self):
        try:
            rate_limiter = RateLimiter(frequency=1 / self.config.dt_sim / self.config.real_time_factor)
            with mujoco.viewer.launch_passive(
                self.mj_model, self.mj_data, show_left_ui=True, show_right_ui=True
            ) as viewer:
                while True:
                    # Read control inputs from shared memory
                    _, q_des = unpack_control_data(self.ctrl_buffer, self.config.nu_real)
                    # check if q_des is close to zero
                    if np.allclose(q_des, np.zeros_like(q_des)):
                        q_des = self.default_ctrl
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
    sim = Sim(robot_name="g1")
    sim.main_loop()
