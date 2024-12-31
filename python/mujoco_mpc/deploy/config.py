import numpy as np


class G1Config:
    # model used in controller
    nq_ctrl: int = 28
    nqd_ctrl: int = 27
    nu_ctrl: int = 21
    dt_ctrl: float = 0.02

    # model used in real robot
    nq_real: int = 36
    nqd_real: int = 35
    nu_real: int = 29
    nu_sim: int = 29
    nu_plan: int = 21
    dt_real: float = 0.005
    gear_array = np.array([
        80.0, 80.0,80.0,
        100.0,50.0,50.0,
        80.0, 80.0,80.0,
        100.0,50.0,50.0,
        80,50,50,
        20,20,1,20,1,1,1,
        20,20,1,20,1,1,1,
       
    ])
    duration_1 = 500
    _targetPos_1 = np.array([ 
    -0.589255, -0.00637053, -0.0617263, 1.26429, -0.727269, -0.033994, 
    -0.613536, -0.0257312, 0.0270589, 1.26707, -0.70565, 0.0334864, 
    0.00208437, 0.00127694, 0.0327588, 
    0.371187, 0.511203, 0.0, 0.0543319, 0.0,0.0,0.0,
    0.371194, -0.511237, 0.0,0.0542188,0.0,0.0,0.0
    ])
    locked_joint_idx: np.ndarray = (
        np.array([2, 4, 5, 6, 9, 11, 12, 13]) + 12 + 3
    )  # locked joints in real robot
    kp_real: np.ndarray = (
        np.array(
            [
                100,
                100,
                100,
                200,
                20,
                20,
                100,
                100,
                100,
                200,
                20,
                20,
                400,
                400,
                400,
                90,
                60,
                20,
                60,
                4,
                4,
                4,
                90,
                60,
                20,
                60,
                4,
                4,
                4,
            ]
        )
        * 0.01
    )
    kd_real: np.ndarray = np.array(
        [
            10.0,10.0,10.0,20.0,2.0,2.0,
            10.0,10.0,10.0,20.0,2.0,2.0,
            40.0,10.0,10.0,
            10.0,6.0,2.0,
            6.0,0.4,0.4,0.4,
            10.0,6.0,2.0,
            6.0,0.4,0.4,0.4,
        ]
    )
    gear_real: np.ndarray = np.array(
        [80] * 3
        + [100]
        + [50] * 2
        + [80] * 3
        + [100]
        + [50] * 2
        + [80]
        + [50] * 2
        + [20] * 6
    )
    # mocap
    mocap_offset: np.ndarray = np.array([0.0, 0.0, 0.078])
    use_mocap_ang_vel: bool = False
    vicon_tracker_ip: str = "128.2.184.3"
    vicon_object_name: str = "g1"

    # controller
    xml_path_ctrl: str = (
        "/home/pcy/Research/code/mujoco_mpc-fork/mjpc/tasks/g1/walk/task.xml"
        # "/home/pcy/Research/code/mujoco_mpc-fork/build/mjpc/tasks/g1/walk/task_benchmark.xml"
        # "/Users/tairanhe/Workspace_MPC_Chaoyi/mpc_chaoyi/mjpc/tasks/g1/walk/task.xml"
    )
    num_opt_steps: int = 1
    task_id: str = "G1 Walk"

    # sim
    xml_path_sim: str = (
        "./model/g1/g1_gear.xml"
        # "/Users/tairanhe/Workspace_MPC_Chaoyi/mpc_chaoyi/mjpc/tasks/g1/walk/task.xml"
        # "/home/pcy/Research/code/mujoco_mpc-fork/mjpc/tasks/g1/g1.xml"
        # "/home/pcy/Research/code/mujoco_mpc-fork/mjpc/tasks/g1/stand/task.xml"
        # "./model/g1/g1_gear.xml"
    )
    dt_sim: float = 0.005
    ctrl_dt: float = 0.01
    real_time_factor: float = 1.0
    auto_reset: bool = True


class Go2Config:
    # model used in controller
    nq_ctrl: int = 19
    nqd_ctrl: int = 18
    nu_ctrl: int = 12
    dt_ctrl: float = 0.005

    # model used in real robot
    nq_real: int = 19
    nqd_real: int = 18
    nu_real: int = 12
    dt_real: float = 0.005
    locked_joint_idx: np.ndarray = np.zeros(0)
    kp_real: np.ndarray = np.array([60.0] * 12)
    kd_real: np.ndarray = np.array([3.0] * 12)
    # gear_real: np.ndarray = np.ones(12)
    gear_real: np.ndarray = np.array(
        [23.7] * 2
        + [45.43]
        + [23.7] * 2
        + [45.43]
        + [23.7] * 2
        + [45.43]
        + [23.7] * 2
        + [45.43]
    )

    # mocap
    mocap_offset: np.ndarray = np.array([0.0, 0.0, -0.060])
    use_mocap_ang_vel: bool = False
    vicon_tracker_ip: str = "128.2.184.3"
    vicon_object_name: str = "lecar_go2_mpc"

    # controller
    xml_path_ctrl: str = (
        # "/Users/tairanhe/Workspace_MPC_Chaoyi/mpc_chaoyi/mjpc/tasks/go2/task_flat.xml"
        "/Users/tairanhe/Workspace_MPC_Chaoyi/mpc_chaoyi/mjpc/tasks/go2/task_flat.xml"
        # "/Users/tairanhe/Workspace_MPC_Chaoyi/mpc_chaoyi/mjpc/tasks/go2/task_real.xml"
    )
    num_opt_steps: int = 1
    task_id: str = "Go2 Flat"

    # sim
    xml_path_sim: str = (
        # "/home/pcy/Research/code/mjpc_sim2real_john/mjpc_john/mjpc/tasks/quadruped/task_flat.xml"
        # "/home/pcy/Research/code/mujoco_mpc-fork/mjpc/tasks/go2/task_flat.xml"
        "./model/go2/go2_torque.xml"
    )
    dt_sim: float = 0.005
    real_time_factor: float = 1.0
    auto_reset: bool = True
    _targetPos_1 = [0.0, 1.36, -2.65, 0.0, 1.36, -2.65,-0.2, 1.36, -2.65, 0.2, 1.36, -2.65]
    _targetPos_2 = [0.0, 0.67, -1.3, 0.0, 0.67, -1.3,0.0, 0.67, -1.3, 0.0, 0.67, -1.3]
    _targetPos_3 = [-0.35, 1.36, -2.65, 0.35, 1.36, -2.65,-0.5, 1.36, -2.65, 0.5, 1.36, -2.65]
    startPos = [0.0] * 12
    duration_1 = 500
    duration_2 = 500
    duration_3 = 1000
    duration_4 = 900
    percent_1 = 0
    percent_2 = 0
    percent_3 = 0
    percent_4 = 0



class QuadrupedConfig(Go2Config):
    dt_ctrl: float = 0.005
    dt_sim: float = 0.005
    xml_path_ctrl: str = (
        "/home/pcy/Research/code/mjpc_sim2real_john/mjpc_john/mjpc/tasks/quadruped/task_flat.xml"
    )
    task_id: str = "Quadruped Flat"
    xml_path_sim: str = (
        "/home/pcy/Research/code/mjpc_sim2real_john/mjpc_john/mjpc/tasks/quadruped/task_flat.xml")


class G1FixedConfig:
    # model used in controller
    nq_ctrl: int = 28
    nqd_ctrl: int = 27
    nu_ctrl: int = 21
    dt_ctrl: float = 0.005

    # model used in real robot
    nq_real: int = 36
    nqd_real: int = 35
    nu_real: int = 29
    nu_sim: int = 29
    nu_plan: int = 21
    dt_real: float = 0.005
    gear_array = np.array([
        80.0, 80.0,80.0,
        100.0,50.0,50.0,
        80.0, 80.0,80.0,
        100.0,50.0,50.0,
        80,50,50,
        20,20,1,20,1,1,1,
        20,20,1,20,1,1,1,
       
    ])
    duration_1 = 500
    _targetPos_1 = np.array([ 
    -0.589255, -0.00637053, -0.0617263, 1.26429, -0.727269, -0.033994, 
    -0.613536, -0.0257312, 0.0270589, 1.26707, -0.70565, 0.0334864, 
    0.00208437, 0.00127694, 0.0327588, 
    0.371187, 0.511203, 0.0, 0.0543319, 0.0,0.0,0.0,
    0.371194, -0.511237, 0.0,0.0542188,0.0,0.0,0.0
    ])
    locked_joint_idx: np.ndarray = (
        np.array([2, 4, 5, 6, 9, 11, 12, 13]) + 12 + 3
    )  # locked joints in real robot
    kp_real: np.ndarray = np.array(
        [
            100,
            100,
            100,
            200,
            20,
            20,
            100,
            100,
            100,
            200,
            20,
            20,
            400,
            400,
            400,
            90,
            60,
            20,
            60,
            4,
            4,
            4,
            90,
            60,
            20,
            60,
            4,
            4,
            4,
        ]
    )
    kd_real: np.ndarray = np.array(
        [
            10.0,10.0,10.0,20.0,2.0,2.0,
            10.0,10.0,10.0,20.0,2.0,2.0,
            40.0,10.0,10.0,
            10.0,6.0,2.0,
            6.0,0.4,0.4,0.4,
            10.0,6.0,2.0,
            6.0,0.4,0.4,0.4,
        ]
    )

    # mocap
    mocap_offset: np.ndarray = np.array([0.0, 0.0, 0.078])
    use_mocap_ang_vel: bool = False
    vicon_tracker_ip: str = "128.2.184.3"
    vicon_object_name: str = "g1"

    # controller
    xml_path_ctrl: str = (
        # "/home/pcy/Research/code/mujoco_mpc-fork/mjpc/tasks/g1/stand/task.xml"
        # "/home/pcy/Research/code/mujoco_mpc-fork/build/mjpc/tasks/g1/walk/task_benchmark.xml"
        "/Users/tairanhe/Workspace_MPC_Chaoyi/mpc_chaoyi/mjpc/tasks/g1_fixed/task.xml"
    )
    num_opt_steps: int = 1
    task_id: str = "G1 Fixed"

    # sim
    xml_path_sim: str = (
        # "/Users/tairanhe/Workspace_MPC_Chaoyi/mpc_chaoyi/mjpc/tasks/g1_fixed/g1_fixed.xml"
        # "/Users/tairanhe/Workspace_MPC_Chaoyi/mpc_chaoyi/mjpc/tasks/g1/walk/task.xml"
        # "/home/pcy/Research/code/mujoco_mpc-fork/mjpc/tasks/g1/g1.xml"
        # "/home/pcy/Research/code/mujoco_mpc-fork/mjpc/tasks/g1/stand/task.xml"
        "./model/g1/g1_gear.xml"
    )
    dt_sim: float = 0.005
    ctrl_dt: float = 0.01
    real_time_factor: float = 1.0
    auto_reset: bool = False
    
