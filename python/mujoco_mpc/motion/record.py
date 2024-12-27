import time
from pathlib import Path
import numpy as np
import mujoco
import mujoco.viewer
import joblib
from loguru import logger
from termcolor import colored
import pandas as pd
from scipy.spatial.transform import Rotation as R

def add_visual_capsule(scene, point1, point2, radius, rgba):
    """Adds one capsule to an mjvScene."""
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom-1],
                        mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                        np.zeros(3), np.zeros(9), rgba.astype(np.float32))
    mujoco.mjv_makeConnector(scene.geoms[scene.ngeom-1],
                            mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                            point1[0], point1[1], point1[2],
                            point2[0], point2[1], point2[2])

def key_callback(keycode):
    global time_step, paused, motion_id, motion_data_keys
    if chr(keycode) == "R":
        logger.info(colored("Reset", "red"))
        time_step = 0
    elif chr(keycode) == " ":
        logger.info(colored("Paused", "green"))
        paused = not paused
    elif chr(keycode) == "N":
        if motion_id >= len(motion_data_keys) - 1:
            logger.info(colored("End of Motion", "red"))
            motion_id = 0
        else:
            motion_id += 1
            curr_motion_key = motion_data_keys[motion_id]
            logger.info(curr_motion_key)
    else:
        logger.info(colored(f"Not mapped: {chr(keycode)}", "red"))

def main(record: bool = True, file_name: str = "lift_hand") -> None:
    global time_step, paused, motion_id, motion_data_keys
    
    visualize_motion_file = f"../data/{file_name}.pkl"

    humanoid_xml = "/home/pcy/Research/code/mujoco_mpc-fork/mjpc/tasks/g1/tracking/task.xml"
    time_step, motion_id, paused = 0, 0, False
    dt = 1/30

    if visualize_motion_file is None:
        logger.error(colored("No motion file provided", "red"))
        return
        
    logger.info(colored(f"Visualizing Motion: {visualize_motion_file}", "green"))
    motion_data = joblib.load(visualize_motion_file)
    motion_data_keys = list(motion_data.keys())

    mj_model = mujoco.MjModel.from_xml_path(str(humanoid_xml))
    mj_data = mujoco.MjData(mj_model)

    body_names = ["pelvis", "head", "ltoe", "rtoe", "lheel", "rheel", "lknee", "rknee", 
                 "lhand", "rhand", "lelbow", "relbow", "lshoulder", "rshoulder", "lhip", "rhip"]
    columns = [f"{body}_{axis}" for body in body_names for axis in ['x', 'y', 'z']]
    sensor_data = []

    mj_model.opt.timestep = dt
    with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=key_callback) as viewer:
        for _ in range(24):
            add_visual_capsule(viewer.user_scn, np.zeros(3), np.array([0.001, 0, 0]), 0.05, np.array([1, 0, 0, 1]))
        logger.info(f"Created {viewer.user_scn.ngeom} capsules")
        
        while viewer.is_running():
            step_start = time.time()
            curr_motion_key = motion_data_keys[motion_id]
            curr_motion = motion_data[curr_motion_key]
            curr_time = int(time_step/dt) % curr_motion['dof'].shape[0]
            
            q_full = np.concatenate([curr_motion['root_trans_offset'][curr_time], 
                                   curr_motion['root_rot'][curr_time][[3, 0, 1, 2]], 
                                   curr_motion['dof'][curr_time]])
            
            # post process qpos

            # lock some joints
            locked_joint_idx = np.array([2, 4, 5, 6, 9, 11, 12, 13]) + 12 + 3 + 7
            locked_mask = np.zeros_like(q_full, dtype=bool)
            locked_mask[locked_joint_idx] = 1
            q_partial = q_full[~locked_mask]
            # clip to ctrllimit
            # actuator_ctrlrange = mj_model.actuator_ctrlrange
            # q_partial[7:] = np.clip(q_partial[7:], actuator_ctrlrange[:, 0], actuator_ctrlrange[:, 1])
            mj_data.qpos = q_partial

            # translate the robot to make sure the feet is always flat relative to the ground
            mujoco.mj_forward(mj_model, mj_data)
            # frist, get the xmat of foot
            for foot_side in ['right', 'left']:
                foot_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, f'{foot_side}_ankle_roll_link')
                foot_xmat = np.array(mj_data.xmat[foot_id]).reshape(3, 3)
                # get rpy of the foot
                foot_rot = R.from_matrix(foot_xmat)
                foot_rpy = foot_rot.as_euler('xyz')
                # get right foot roll joint id
                foot_roll_joint_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, f'{foot_side}_ankle_roll_joint')
                foot_roll_joint_qpos_idx = foot_roll_joint_id - 1 + 7 # remove the free joint and add back the free joint dof
                foot_pitch_joint_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, f'{foot_side}_ankle_pitch_joint')
                foot_pitch_joint_qpos_idx = foot_pitch_joint_id - 1 + 7 # remove the free joint and add back the free joint dof
                # set the roll and pitch of the foot to compensate the roll and pitch of the foot
                mj_data.qpos[foot_roll_joint_qpos_idx] = -foot_rpy[0]
                mj_data.qpos[foot_pitch_joint_qpos_idx] = -foot_rpy[1]

            
            mujoco.mj_forward(mj_model, mj_data)

            # compare tracking site tracking[lheel] and tracking[ltoe] position
            z_offset = 100.0
            for foot_side in ['r', 'l']:
                heel_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, f'tracking[{foot_side}heel]')
                toe_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, f'tracking[{foot_side}toe]')
                heel_pos = mj_data.site_xpos[heel_id]
                toe_pos = mj_data.site_xpos[toe_id]
                z_foot = (heel_pos[2] + toe_pos[2]) / 2
                if z_foot < z_offset:
                    z_offset = z_foot
            z_offset += 0.001
            # offset the robot by z_offset
            mj_data.qpos[2] = mj_data.qpos[2] - z_offset

            mujoco.mj_forward(mj_model, mj_data)

            if not paused:
                time_step += dt

            timestep_data = []
            for body_name in body_names:
                pos_sensor_name = f"tracking_pos[{body_name}]"
                sensor_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, pos_sensor_name)
                sensor_adr = mj_model.sensor_adr[sensor_id]
                pos = mj_data.sensordata[sensor_adr:sensor_adr+3]
                timestep_data.extend(pos)
            
            if record:
                frame_data = {
                    'timestamp': time_step,
                    'motion_id': motion_id,
                    'motion_key': curr_motion_key,
                    'frame': curr_time,
                    **{f'qpos_{i}': val for i, val in enumerate(mj_data.qpos)}
                }
                
                for i, col in enumerate(columns):
                    frame_data[col] = timestep_data[i]
                sensor_data.append(frame_data)

            if 'smpl_joints' in curr_motion:
                joint_gt = curr_motion['smpl_joints']
                for i in range(joint_gt.shape[1]):
                    viewer.user_scn.geoms[i].pos = joint_gt[curr_time, i]

            viewer.sync()

            if record:
                if int(time_step/dt) >= curr_motion['dof'].shape[0]:
                    break

    if record:
        csv_filename = f"../data/{file_name}.csv"
        pd.DataFrame(sensor_data).to_csv(csv_filename, index=False)
        logger.info(colored(f"Saved tracking data to: {csv_filename}", "green"))

if __name__ == "__main__":
    main(record=True, file_name="down_box_to_walk")