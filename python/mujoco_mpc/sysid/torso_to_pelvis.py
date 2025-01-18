# %%
import os
os.environ["DISPLAY"] = ":1"
import mujoco
import matplotlib.pyplot as plt
model = mujoco.MjModel.from_xml_path("../deploy/model/h1/h1_mani.xml")
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model)

# reset data
mujoco.mj_resetData(model, data)

# %%
# set torso_joint to 1.0
data.qpos[3] = 0.3
data.qpos[4] = np.sqrt(1.0 - data.qpos[3]**2)
data.qpos[7+10] = 1.0
# forward dynamics
mujoco.mj_forward(model, data)
# render frame
renderer.update_scene(data, camera=0)
pixels = renderer.render()

# %%
plt.imshow(pixels)

# %%

# create translation matrix
import numpy as np
from scipy.spatial.transform import Rotation as R
# print with 2 decimal places
np.set_printoptions(precision=2)

pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")

# get ground truth values
p_pelvis = data.xpos[pelvis_id]
quat_pelvis = data.xquat[pelvis_id]
X_pelvis = np.eye(4)
X_pelvis[0:3, 3] = p_pelvis
X_pelvis[0:3, 0:3] = R.from_quat(quat_pelvis, scalar_first=True).as_matrix()
print("X_pelvis Ground Truth:")
print(X_pelvis)

# calculate from torso position
p_torso = data.xpos[torso_id]
quat_torso = data.xquat[torso_id]
X_torso = np.eye(4)
X_torso[0:3, 3] = p_torso
X_torso[0:3, 0:3] = R.from_quat(quat_torso, scalar_first=True).as_matrix()
X_pelvis_torso = np.eye(4)
X_pelvis_torso[0:3, 0:3] = R.from_euler("xyz", [0.0, 0.0, 1.0], degrees=False).as_matrix()
X_pelvis_est = X_torso @ np.linalg.inv(X_pelvis_torso)
print("X_pelvis Estimate:")
print(X_pelvis_est)

# %%

# %%
