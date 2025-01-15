   
class H1Config:
    # model used in controller
    nq_ctrl: int = 17
    nqd_ctrl: int = 16
    nu_ctrl: int = 10
    dt_ctrl: float = 0.02

    # model used in real robot
    nq_real: int = 19 +7
    nqd_real: int = 18 +7
    nu_real: int = 19
    nu_sim: int = 19
   
    dt_real: float = 0.005
   
    duration_1 = 100
    _targetPos_1 = np.array([ 
   0.0,0.0,-0.6, 1.26429, -0.727269, 
    0.0,0.0,-0.6, 1.26707, -0.70565,
    0.0,
     0.0,0.0,0.0,0.0,
    0.0,0.0,0.0,0.0
    ])
    locked_joint_idx: np.ndarray = (
        np.array([10,11,12,13,14,15,16,17,18])
    )
    weak_motor_idx = np.array([4,9,11,12,13,14,15,16,17,18])  # locked joints in real robot
    kp_real: np.array = np.array(
        [
           100.0,100.0,100.0,150.0,20.0,
            100.0,100.0,100.0,150.0,20.0,
            500.0,
            250.0,250.0,250.0,
            160.0,80.0,80.0,80.0,
            250.0,250.0,250.0,
            160.0,80.0,80.0,80.0
        ]
    )
    kd_real: np.ndarray = np.array(
        [
           2.5,2.5,2.5,4.0,3.0,3.0,
           2.5,2.5,2.5,4.0,3.0,3.0,
           5.0,
           4.0,4.0,4.0,
           2.0,1.0,1.0,1.0,
           4.0,4.0,4.0,
           2.0,1.0,1.0,1.0,
        ]
    ) 
    gear_real = np.ones(19)
    # mocap
    mocap_offset: np.ndarray = np.array([0.0, 0.0, 0.078])
    use_mocap_ang_vel: bool = False
    vicon_tracker_ip: str = "128.2.184.3"
    vicon_object_name: str = "lecar_h1_mpc"

    # controller
    xml_path_ctrl: str = (
        str(MUJOCO_MPC_TASK_PATH / "h1_2/walk/task.xml")
        # "/Users/tairanhe/Workspace_MPC_Chaoyi/mpc_chaoyi/mjpc/tasks/g1/walk/task.xml"
        # "/home/pcy/Research/code/mujoco_mpc-fork/build/mjpc/tasks/g1/walk/task_benchmark.xml"
        # "/Users/tairanhe/Workspace_MPC_Chaoyi/mpc_chaoyi/mjpc/tasks/g1/walk/task.xml"
    )
    num_opt_steps: int = 1
    task_id: str = "H1_2 Walk"

    # sim
    xml_path_sim: str = (
        "./model/h1_2/h1_2_position.xml"
        # "/Users/tairanhe/Workspace_MPC_Chaoyi/mpc_chaoyi/mjpc/tasks/g1/walk/task.xml"
        # "/home/pcy/Research/code/mujoco_mpc-fork/mjpc/tasks/g1/g1.xml"
        # "/home/pcy/Research/code/mujoco_mpc-fork/mjpc/tasks/g1/stand/task.xml"
        # "./model/g1/g1_gear.xml"
    )
    dt_sim: float = 0.005
    ctrl_dt: float = 0.005
    real_time_factor: float = 1.0
    auto_reset: bool = True
    sim_mocap_z_offset = -0.00
    sim_mocap_roll_offset = 0.00
    sim_mocap_pitch_offset = -0.00
    auto_reset = True

    control_mode = "position"
    q_default = np.array([ 
    0.0,-0.6, 0.0, 1.26429, -0.727269, -0.033994, 
    0.0,-0.6, 0.0, 1.26707, -0.70565, 0.0334864, 
    0.0,
    0.371187, 0.511203, 0.0, 0.0543319, 0.0,0.0,0.0,
    0.371194, -0.511237, 0.0,0.0542188,0.0,0.0,0.0
    ])
