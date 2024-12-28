import numpy as np
import struct

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
