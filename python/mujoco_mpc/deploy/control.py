import numpy as np
import struct
from multiprocessing import shared_memory
from loop_rate_limiters import RateLimiter
import pathlib

import mujoco
from mujoco_mpc import agent as agent_lib

from config import G1Config, Go2Config, QuadrupedConfig
from utils import (
    pack_control_data,
    unpack_mocap_data,
    unpack_state_data,
    ctrl_sim2real,
    state_real2sim,
)


class Controller:
    def __init__(self, robot_name="g1", mujoco_mpc_mode="gui"):
        if robot_name == "g1":
            self.config = G1Config()
        elif robot_name == "go2":
            self.config = Go2Config()
        elif robot_name == "quadruped":
            self.config = QuadrupedConfig()
        else:
            raise ValueError(f"Robot {robot_name} not supported")
        self.mujoco_mpc_mode = mujoco_mpc_mode

        # State variables read from real robot
        try:
            self.state_shm = shared_memory.SharedMemory(name="state_shm")
            self.state_buffer = self.state_shm.buf
        except FileNotFoundError:
            print("State shared memory 'state_shm' not found.")
            exit()
        # Control variables written to real robot
        try:
            self.ctrl_shm = shared_memory.SharedMemory(name="ctrl_shm")
            self.ctrl_buffer = self.ctrl_shm.buf
        except FileNotFoundError:
            print("Could not create control shared memory 'ctrl_shm'.")
            exit()

        # Initialize control variables
        self.last_plan_time = 0.0
        self.state = None  # Will be initialized in main_loop

    def get_action(self, agent, qpos, qvel):
        agent.set_state(qpos=qpos, qvel=qvel)
        for _ in range(self.config.num_opt_steps):
            agent.planner_step()
        ctrl = agent.get_action()
        return ctrl

    def get_state(self):
        t_real, q_real, qd_real = unpack_state_data(
            self.state_buffer, self.config.nq_real, self.config.nqd_real
        )
        q_sim, qd_sim = state_real2sim(
            q_real,
            qd_real,
            self.config.locked_joint_idx,
            self.config.nq_ctrl,
            self.config.nqd_ctrl,
            self.config.nq_real,
            self.config.nqd_real,
        )
        return q_sim, qd_sim, t_real

    def main_loop(self):
        # Controller
        model = mujoco.MjModel.from_xml_path(self.config.xml_path_ctrl)
        rate_limiter = RateLimiter(frequency=1 / self.config.dt_ctrl)
        try:
            if self.mujoco_mpc_mode == "headless":
                agent = agent_lib.Agent(task_id=self.config.task_id, model=model)
                while True:
                    q_sim, qd_sim, t_real = self.get_state()
                    # NOTE: t_real is not used here
                    ctrl = self.get_action(agent, q_sim, qd_sim)
                    ctrl_real = ctrl_sim2real(
                        ctrl, self.config.locked_joint_idx, self.config.nu_real
                    )
                    self.ctrl_buffer[:] = pack_control_data(
                        self.ctrl_buffer, t_real, ctrl_real
                    )
                    rate_limiter.sleep()
            elif self.mujoco_mpc_mode == "gui":
                with agent_lib.Agent(
                    server_binary_path=pathlib.Path(agent_lib.__file__).parent
                    / "mjpc"
                    / "ui_agent_server",
                    task_id=self.config.task_id,
                    model=model,
                ) as agent:
                    while True:
                        q_sim, qd_sim, t_real = self.get_state()
                        agent.set_state(qpos=q_sim, qvel=qd_sim)
                        ctrl = agent.get_action()
                        ctrl_real = ctrl_sim2real(
                            ctrl, self.config.locked_joint_idx, self.config.nu_real
                        )
                        self.ctrl_buffer[:] = pack_control_data(
                            self.ctrl_buffer, t_real, ctrl_real
                        )
                        rate_limiter.sleep()

        except KeyboardInterrupt:
            print("Keyboard interrupt detected. Exiting...")
        finally:
            self.state_shm.close()
            self.ctrl_shm.close()


if __name__ == "__main__":
    controller = Controller(robot_name="g1", mujoco_mpc_mode="gui")
    controller.main_loop()
