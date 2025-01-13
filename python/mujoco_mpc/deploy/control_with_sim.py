import numpy as np
import struct
from multiprocessing import shared_memory
from loop_rate_limiters import RateLimiter
import pathlib
from scipy.spatial.transform import Rotation as R

import mujoco
import mujoco.viewer
from mujoco_mpc import agent as agent_lib

from config import G1PositionConfig, Go2PositionConfig, QuadrupedConfig, H1_2PositionConfig, H1_2_simpleConfig  
from utils import (
    pack_control_data,
    unpack_mocap_data,
    unpack_state_data,
    ctrl_sim2real,
    state_real2sim,
)


class Controller:
    def __init__(self, robot_name="g1", mujoco_mpc_mode="gui"):
        self.robot_name = robot_name
        if robot_name == "g1":
            self.config = G1PositionConfig()
        elif robot_name == "go2":
            self.config = Go2PositionConfig()
        elif robot_name == "quadruped":
            self.config = QuadrupedConfig()
        elif robot_name == "h1_2":
            self.config = H1_2PositionConfig()
        elif robot_name == "h1_2_simple":
            self.config = H1_2_simpleConfig()
        else:
            raise ValueError(f"Robot {robot_name} not supported")
        self.mujoco_mpc_mode = mujoco_mpc_mode

        # State variables read from real robot
        # try:
        #     self.state_shm = shared_memory.SharedMemory(name="state_shm")
        #     self.state_buffer = self.state_shm.buf
        # except FileNotFoundError:
        #     print("State shared memory 'state_shm' not found.")
        #     exit()
        # # Control variables written to real robot
        # try:
        #     self.ctrl_shm = shared_memory.SharedMemory(name="ctrl_shm")
        #     self.ctrl_buffer = self.ctrl_shm.buf
        # except FileNotFoundError:
        #     print("Could not create control shared memory 'ctrl_shm'.")
        #     exit()

        # Initialize control variables
        self.last_plan_time = 0.0
        self.state = None  # Will be initialized in main_loop

        # Mujoco model setup
        self.mj_model = mujoco.MjModel.from_xml_path(self.config.xml_path_sim)
        self.mj_model.opt.timestep = self.config.dt_sim
        self.mj_data = mujoco.MjData(self.mj_model)
        mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, 0)
        self.n_sim_frame = int(self.config.dt_ctrl / self.config.dt_sim)

        # compute ground truth mocap rotation matrix
        self.marker_R_robot_gt = R.from_euler('xyz', [self.config.sim_mocap_roll_offset, self.config.sim_mocap_pitch_offset, 0.0], degrees=False)
        self.marker_p_robot_gt = np.array([0.0, 0.0, self.config.sim_mocap_z_offset])
        self.marker_R_robot_est = R.from_euler('xyz', [0.0, 0.0, 0.0], degrees=False)
        self.marker_p_robot_est = np.array([0.0, 0.0, 0.0])

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
        # print(self.config.xml_path_ctrl)
        roll_offset = 0.05 * 0
        pitch_offset = 0.05 *0
        yaw_offset = 0.05 *0
        r_offset = np.array([0.01,0.01,0.01]) * 0
        model = mujoco.MjModel.from_xml_path(self.config.xml_path_ctrl)
        rate_limiter = RateLimiter(frequency=1 / self.config.dt_ctrl)
        with mujoco.viewer.launch_passive(
            self.mj_model, self.mj_data, show_left_ui=True, show_right_ui=False
        ) as viewer:
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
                    print(
                        "Agent server binary path:",
                        pathlib.Path(agent_lib.__file__).parent
                        / "mjpc"
                        / "ui_agent_server",
                    )
                    print("Task ID:", self.config.task_id)
                    print("Model Path:", self.config.xml_path_ctrl)
                    with agent_lib.Agent(
                        server_binary_path=pathlib.Path(agent_lib.__file__).parent
                        / "mjpc"
                        / "ui_agent_server",
                        task_id=self.config.task_id,
                        model=model,
                    ) as agent:
                        while True:
                            # q_sim, qd_sim, t_real = self.get_state()

                            # get state from simulator
                            q_sim = self.mj_data.qpos.copy()
                            qd_sim = self.mj_data.qvel.copy()

                            # # get marker position
                            # marker_pos_sim = q_sim[:3] + self.marker_p_robot_est
                            # robot_R_mocap = R.from_quat(q_sim[3:7])
                            # marker_R_mocap = self.marker_R_robot_gt * robot_R_mocap
                            # # convert it back to robot position with estimated marker position
                            # robot_pos_est = marker_pos_sim - self.marker_p_robot_est
                            # robot_R_mocap_est = self.marker_R_robot_est.inv() * marker_R_mocap
                            # q_sim[:3] = robot_pos_est
                            # q_sim[3:7] = robot_R_mocap_est.as_quat()


                            # compute mocap rotation matrix 
                            if self.robot_name == "h1_2":
                                q_sim, qd_sim = state_real2sim(
                                    q_sim,
                                    qd_sim,
                                    self.config.locked_joint_idx,
                                    self.config.nq_ctrl,
                                    self.config.nqd_ctrl,
                                    self.config.nq_real,
                                    self.config.nqd_real,
                                )
                            elif self.robot_name == "h1_2_simple":
                                q_sim, qd_sim = state_real2sim(
                                    q_sim,
                                    qd_sim,
                                    self.config.locked_joint_idx,
                                    self.config.nq_ctrl,
                                    self.config.nqd_ctrl,
                                    self.config.nq_real,
                                    self.config.nqd_real,
                                )
                            lin_vel = qd_sim[:3]
                            # apply rotation offset to lin_vel
                            lin_rot_mat = R.from_euler('xyz', [roll_offset, pitch_offset, yaw_offset], degrees=False)
                            lin_vel = lin_rot_mat.apply(lin_vel)
                            # calculate omega * r_offset
                            omega = qd_sim[3:6]
                            omega_r_offset = np.cross(omega, r_offset)
                            qd_sim[:3] = lin_vel + omega_r_offset
                            
                            # set state to agent and get action
                            agent.set_state(qpos=q_sim, qvel=qd_sim)
                            ctrl = agent.get_action()
                            print(ctrl)
                            if self.robot_name == "h1_2_simple":
                                ctrl_real = ctrl_sim2real(
                                    ctrl, self.config.locked_joint_idx, self.config.nu_real
                                )
                            else:
                                ctrl_real = ctrl_sim2real(
                                    ctrl, self.config.locked_joint_idx, self.config.nu_real
                                )

                            # step simulation
                            for _ in range(self.n_sim_frame):
                                self.mj_data.ctrl = ctrl_real
                                mujoco.mj_step(self.mj_model, self.mj_data)
                            viewer.sync()
                            rate_limiter.sleep()

            except KeyboardInterrupt:
                print("Keyboard interrupt detected. Exiting...")
            finally:
                pass
            # self.state_shm.close()
            # self.ctrl_shm.close()


if __name__ == "__main__":
    controller = Controller(robot_name="h1_2_simple", mujoco_mpc_mode="gui")
    controller.main_loop()
