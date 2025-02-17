import mujoco

# load model
model = mujoco.MjModel.from_xml_path("tracking_realtime/task.xml")

# load data
data = mujoco.MjData(model)

# get mocap body number
print(f"nmocap: {model.nmocap}")

# get key frame mpos size
print(f"key_mpos: {model.key_mpos.shape}")