import numpy as np
import struct
import xml.etree.ElementTree as ET
import mujoco
import tkinter as tk

def create_bar(params_dict):
    root = tk.Tk()
    root.title("Input Bars")

    # Create a dictionary to store the scale widgets
    bars = {}

    # Create a scale widget for each parameter
    for param_name, param_config in params_dict.items():
        lower_bound = param_config.get('lower', 0)
        upper_bound = param_config.get('upper', 1) 
        step_size = param_config.get('step', 0.01)
        default_value = param_config.get('default', 0)
        
        # Create scale (slider) with the specified parameters
        bar = tk.Scale(
            root,
            from_=lower_bound,
            to=upper_bound,
            resolution=step_size,
            orient="horizontal",
            length=300,
            label=f"Adjust {param_name}",
        )
        bar.pack(pady=10)
        bar.set(default_value)
        bars[param_name] = bar

    # Create a function to keep the tkinter window running in a non-blocking way
    def update_gui():
        root.update_idletasks()
        root.update()

    return root, bars, update_gui



def pack_mocap_data(buffer, timestamp, q_mocap, qd_mocap):
    struct.pack_into("d", buffer, 0, timestamp)
    struct.pack_into(f"{q_mocap.shape[0]}d", buffer, 8, *q_mocap)
    struct.pack_into(f"{qd_mocap.shape[0]}d", buffer, 8 + q_mocap.shape[0] * 8, *qd_mocap)
    return buffer

def unpack_mocap_data(buffer):
    # Unpack all data at once
    unpacked_data = struct.unpack_from("13d", buffer, 0)

    timestamp = unpacked_data[0]  # first element is the timestamp (microseconds)
    position = unpacked_data[1:4]
    quaternion = unpacked_data[4:8]
    linear_velocity = unpacked_data[8:11]
    angular_velocity = unpacked_data[11:14]

    q = np.concatenate([position, quaternion])
    qd = np.concatenate([linear_velocity, angular_velocity])

    return q, qd

def pack_control_data(buffer, timestamp, q_des):
    struct.pack_into("d", buffer, 0, timestamp)
    struct.pack_into(f"{q_des.shape[0]}d", buffer, 8, *q_des)
    return buffer

def unpack_control_data(buffer, nu_real):
    ctrl_data = struct.unpack_from(f"{nu_real+1}d", buffer, 0)
    ctrl_time = ctrl_data[0]
    q_des = ctrl_data[1:nu_real+1]
    q_des = np.array(q_des)
    return ctrl_time, q_des

def pack_state_data(buffer, timestamp, q, qd):
    struct.pack_into("d", buffer, 0, timestamp)
    struct.pack_into(f"{q.shape[0]}d", buffer, 8, *q)
    struct.pack_into(f"{qd.shape[0]}d", buffer, 8 + q.shape[0] * 8, *qd)
    return buffer

def unpack_state_data(buffer, nq_real, nqd_real):
    state_data = struct.unpack_from(f"{nq_real+nqd_real+1}d", buffer, 0)
    state_time = state_data[0]
    q = state_data[1:nq_real+1]
    qd = state_data[nq_real+1:nq_real+nqd_real+1]
    q = np.array(q)
    qd = np.array(qd)
    return state_time, q, qd

def ctrl_sim2real(ctrl_sim, locked_joint_idx, nu_real):
    ctrl_real = np.zeros(nu_real)
    locked_mask = np.zeros(nu_real, dtype=bool)
    if len(locked_joint_idx) > 0:
        locked_mask[locked_joint_idx] = True
    ctrl_real[~locked_mask] = ctrl_sim
    return ctrl_real

def state_real2sim(q_real, qd_real, locked_joint_idx, nq_ctrl, nqd_ctrl, nq_real, nqd_real):
    locked_mask = np.zeros(nq_real-7, dtype=bool)
    if len(locked_joint_idx) > 0:
        locked_mask[locked_joint_idx] = True
    q_sim, qd_sim = np.zeros(nq_ctrl), np.zeros(nqd_ctrl)
    q_sim[:7] = q_real[:7]
    q_sim[7:nq_ctrl] = q_real[7:nq_real][~locked_mask]
    qd_sim[:6] = qd_real[:6]
    qd_sim[6:nqd_ctrl] = qd_real[6:nqd_real][~locked_mask]
    return q_sim, qd_sim

def apply_gear_to_control(ctrl, gear_array):
    """
    Multiplies the control signals by the gear values.

    Parameters:
    - ctrl: np.ndarray, the control signals without gear.
    - gear_array: np.ndarray, the gear values for each control signal.

    Returns:
    - np.ndarray, the control signals with gear applied.
    """
    return ctrl * gear_array

def ctrl_real2sim(ctrl_w_mask, locked_idx, nu_real):
    # print(ctrl_w_mask.shape)
    # print(locked_idx.shape)
    
    mask = np.ones(29, dtype=bool)
    mask[locked_idx] = False
    print(ctrl_w_mask[mask].shape)
    ctrl_sim = ctrl_w_mask[mask]
    # print(ctrl_sim.shape)
    return ctrl_sim
# Example usage:
# ctrl_with_gear = apply_gear_to_control(ctrl_without_gear, gear_array) 

def read_motor_gears(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    motor_gears = {}
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    print('Reading motor gears from xml file...')
    for motor in root.findall('.//motor'):
        name = motor.get('name')
        gear = motor.get('gear')
        joint_name = motor.get('joint')
        if name and gear and joint_name:
            # Find the joint index in the MuJoCo model
            joint_index = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            motor_gears[name] = {
                'gear': float(gear),
                'joint_index': joint_index
            }
    # print(f"motor_gears: {motor_gears}")
    return motor_gears

def fit_plane_svd(points):
    """
    Given N points (N >= 3) in 3D space, find the unit normal of the
    best-fit plane via SVD and return (normal, centroid).
    """
    # 1) Compute centroid
    centroid = np.mean(points, axis=0)  # shape (3,)

    # 2) Shift points so that centroid is at the origin
    shifted = points - centroid  # shape (N, 3)

    # 3) SVD on the shifted points
    #    U shape (N,N), S shape (N,), Vt shape (N,3), but effectively the
    #    last row of Vt gives the direction of minimal variance = plane normal
    U, S, Vt = np.linalg.svd(shifted, full_matrices=False)

    # 4) The plane normal is given by the last row of V^T
    normal = Vt[-1, :]

    # 5) Normalize it
    normal /= np.linalg.norm(normal)

    return normal, centroid

def rotation_matrix_from_vectors(vec_from, vec_to):
    """
    Returns the rotation matrix R that aligns vec_from to vec_to.
    Both vec_from and vec_to are 3D vectors.
    """
    # 1) Make sure both vectors are normalized
    v1 = vec_from / np.linalg.norm(vec_from)
    v2 = vec_to   / np.linalg.norm(vec_to)

    # 2) If they are (almost) the same, return identity
    dot = np.dot(v1, v2)
    if np.isclose(dot, 1.0):
        return np.eye(3)

    # 3) Cross product as rotation axis
    cross = np.cross(v1, v2)

    # 4) Construct the skew-symmetric cross-product matrix K
    K = np.array([
        [0,         -cross[2],  cross[1]],
        [cross[2],  0,         -cross[0]],
        [-cross[1], cross[0],   0       ]
    ])

    # 5) Rodrigues’ rotation formula: R = I + K + K^2 * (1 / (1 + dot))
    R = np.eye(3) + K + K @ K * (1.0 / (1.0 + dot))
    return R

def minimize_z_difference(points):
    """
    Given 4 or more points in 3D (shape (N,3)), this function:
      1) Finds the best-fit plane normal.
      2) Creates a rotation matrix R that will rotate the plane's normal
         to the z-axis about the origin (no centroid shifting).
      3) Returns (R, z_offset), where z_offset is the average z-value
         after rotation. If you rotate first (about the origin) and then
         subtract z_offset, your plane will be parallel to XY.
    """
    # 1) Fit the plane to get the normal (and centroid, which we will not shift by)
    normal, centroid = fit_plane_svd(points)

    # 2) Build rotation that takes 'normal' -> [0, 0, 1], around origin
    # make sure normal is always close to [0, 0, 1], otherwise, flip it
    if normal[2] < 0:
        normal = -normal
    R = rotation_matrix_from_vectors(normal, np.array([0, 0, 1]))

    # 3) Rotate all original points about the origin
    rotated_points = (R @ points.T).T

    # 4) The z_offset is the mean z of the rotated points
    z_offset = np.mean(rotated_points[:, 2])

    return R, z_offset