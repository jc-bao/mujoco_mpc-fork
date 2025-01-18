import numpy as np
import struct
from multiprocessing import shared_memory
from loop_rate_limiters import RateLimiter
import pathlib
from scipy.spatial.transform import Rotation as R

import mujoco
import mujoco.viewer
from mujoco_mpc import agent as agent_lib

from config import (
    G1PositionConfig,
    Go2PositionConfig,
    QuadrupedConfig,
    H1_2PositionConfig,
    H1_2_simpleConfig,
    H1Config,
    H1_maniConfig,
    ObjectConfig,
)
from utils import (
    pack_control_data,
    unpack_mocap_data,
    unpack_state_data,
    ctrl_sim2real,
    state_real2sim,
)


class Controller:
    def __init__(self, robot_name="g1", mujoco_mpc_mode="gui", dump_data=False):
        self.max_delta_ctrl = 0.5
        self.control_gamma = 0.5
        self.dump_data = dump_data
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
        elif robot_name == "h1":
            self.config = H1Config()
        elif robot_name =="h1_mani":
            self.config = H1_maniConfig()
            self.object_config = ObjectConfig()
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
        self.marker_R_robot_gt = R.from_euler(
            "xyz",
            [
                self.config.sim_mocap_roll_offset,
                self.config.sim_mocap_pitch_offset,
                0.0,
            ],
            degrees=False,
        )
        self.marker_p_robot_gt = np.array([0.0, 0.0, self.config.sim_mocap_z_offset])
        self.marker_R_robot_est = R.from_euler("xyz", [0.0, 0.0, 0.0], degrees=False)
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
        ctrl_buffer_delay = np.zeros((self.config.sim_delay_frames+1, self.config.nu_real-1))
        mocap_delay_frames_q = np.zeros((self.config.mocap_delay_frames+1, 7))
        mocap_delay_frames_q[:, 3] = 1.0
        mocap_delay_frames_qd = np.zeros((self.config.mocap_delay_frames+1, 6))
        roll_offset = 0.00
        pitch_offset = 0.00 
        yaw_offset = 0.00
        r_offset = np.array([0.00, 0.00, 0.00])
        model = mujoco.MjModel.from_xml_path(self.config.xml_path_ctrl)
        rate_limiter = RateLimiter(frequency=1 / self.config.dt_ctrl)
        last_ctrl = np.zeros(self.config.nu_ctrl)
        cnt = 0
        if self.dump_data:
            data_cnt = 0
            buffer_size = 1000
            q_buffer = np.zeros((buffer_size, self.config.nq_ctrl))
            qd_buffer = np.zeros((buffer_size, self.config.nqd_ctrl))
            ctrl_buffer = np.zeros((buffer_size, self.config.nu_ctrl))
        with mujoco.viewer.launch_passive(
            self.mj_model, self.mj_data, show_left_ui=True, show_right_ui=True
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
                            mocap_delay_frames_q = np.roll(mocap_delay_frames_q, -1, axis=0)
                            mocap_delay_frames_q[-1] = q_sim[:7]
                            mocap_delay_frames_qd = np.roll(mocap_delay_frames_qd, -1, axis=0)
                            mocap_delay_frames_qd[-1] = qd_sim[:6]
                            if cnt % self.config.mocap_delay_interval == 0:
                                q_sim[:7] = mocap_delay_frames_q[0]
                                qd_sim[:6] = mocap_delay_frames_qd[0]

                            # get marker position
                            marker_pos_sim = q_sim[:3] + self.marker_p_robot_est
                            robot_R_mocap = R.from_quat(q_sim[3:7])
                            marker_R_mocap = self.marker_R_robot_gt * robot_R_mocap
                            # convert it back to robot position with estimated marker position
                            robot_pos_est = marker_pos_sim - self.marker_p_robot_est
                            robot_R_mocap_est = self.marker_R_robot_est.inv() * marker_R_mocap
                            q_sim[:3] = robot_pos_est
                            q_sim[2] += np.random.normal(0, self.config.sim_mocap_z_offset_noise)
                            q_sim[3:7] = robot_R_mocap_est.as_quat()

                            # compute mocap rotation matrix
                            if self.robot_name == "h1":
                                q_sim, qd_sim = state_real2sim(
                                    q_sim,
                                    qd_sim,
                                    self.config.locked_joint_idx,
                                    self.config.nq_ctrl,
                                    self.config.nqd_ctrl,
                                    self.config.nq_real-1,
                                    self.config.nqd_real-1,
                                )
                            elif self.robot_name == "h1_mani":
                                q_sim, qd_sim = state_real2sim(
                                    q_sim,
                                    qd_sim,
                                    self.config.locked_joint_idx,
                                    self.config.nq_ctrl,
                                    self.config.nqd_ctrl,
                                    self.config.nq_real-1,
                                    self.config.nqd_real-1,
                                )
                            else:
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
                            lin_rot_mat = R.from_euler(
                                "xyz",
                                [roll_offset, pitch_offset, yaw_offset],
                                degrees=False,
                            )
                            lin_vel = lin_rot_mat.apply(lin_vel)
                            # calculate omega * r_offset
                            omega = qd_sim[3:6]
                            omega_r_offset = np.cross(omega, r_offset)
                            qd_sim[:3] = lin_vel + omega_r_offset
                            # qd_sim[:3] *= 0.0
                            # set state to agent and get action
                            agent.set_state(qpos=q_sim, qvel=qd_sim)
                            ctrl = agent.get_action()
                            # clip ctrl with last_ctrl
                            ctrl = np.clip(ctrl, last_ctrl - self.max_delta_ctrl, last_ctrl + self.max_delta_ctrl)
                            ctrl = self.control_gamma * ctrl + (1 - self.control_gamma) * last_ctrl
                            if self.dump_data:
                                if data_cnt >= buffer_size:
                                    print("Buffer full, stopping data collection")
                                    break
                                print("dump progress: ", data_cnt / buffer_size)
                                q_buffer[data_cnt] = q_sim
                                qd_buffer[data_cnt] = qd_sim
                                ctrl_buffer[data_cnt] = ctrl
                                data_cnt += 1
                            if self.robot_name == "h1_2_simple":
                                ctrl_real = ctrl_sim2real(
                                    ctrl,
                                    self.config.locked_joint_idx,
                                    self.config.nu_real,
                                )
                            elif self.robot_name == "h1":
                                ctrl_real = ctrl_sim2real(
                                    ctrl,
                                    self.config.locked_joint_idx,
                                    self.config.nu_real - 1,
                                )
                                ctrl_delay = ctrl_sim2real(last_ctrl, self.config.locked_joint_idx, self.config.nu_real-1)
                            elif self.robot_name == "h1_mani":
                                ctrl_real = ctrl_sim2real(
                                    ctrl,
                                    self.config.locked_joint_idx,
                                    self.config.nu_real-1,
                                )
                                ctrl_delay = ctrl_sim2real(last_ctrl, self.config.locked_joint_idx, self.config.nu_real-1)
                            else:
                                ctrl_real = ctrl_sim2real(
                                    ctrl,
                                    self.config.locked_joint_idx,
                                    self.config.nu_real,
                                )
                            ctrl_buffer_delay = np.roll(ctrl_buffer_delay, -1, axis=0)
                            ctrl_buffer_delay[-1] = ctrl_real

                            # step simulation
                            for k in range(self.n_sim_frame):
                                if self.config.force_in_sim:
                                    # ctrl_delay = ctrl_buffer_delay[-2]
                                    if k < self.config.sim_delay_frames:
                                        pos_tar = ctrl_delay
                                    else:
                                        pos_tar = ctrl_real
                                    force = self.config.kp_real*(1.0) * (pos_tar - self.mj_data.qpos[7:]) - self.config.kd_real * self.mj_data.qvel[6:]
                                    # add random noise to force 
                                    force_ratio = np.random.normal(1, 0.001)
                                    force = force * force_ratio
                                    self.mj_data.ctrl = force
                                else:
                                    self.mj_data.ctrl = ctrl_real
                                mujoco.mj_step(self.mj_model, self.mj_data)

                            last_ctrl = ctrl
                            cnt += 1
                            viewer.sync()
                            rate_limiter.sleep()

            except KeyboardInterrupt:
                print("Keyboard interrupt detected. Exiting...")
            finally:
                if self.dump_data:
                    np.savez(
                        f"{self.robot_name}_data.npz",
                        q=q_buffer,
                        qd=qd_buffer,
                        ctrl=ctrl_buffer,
                    )
            # self.state_shm.close()
            # self.ctrl_shm.close()


if __name__ == "__main__":
    controller = Controller(robot_name="h1_mani", mujoco_mpc_mode="gui", dump_data=False)
    controller.main_loop()
