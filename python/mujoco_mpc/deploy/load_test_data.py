import numpy as np

data = np.load("h1_data.npz")
q_buffer = data["q"]
qd_buffer = data["qd"]
ctrl_buffer = data["ctrl"]

print(q_buffer.shape)
print(qd_buffer.shape)
print(ctrl_buffer.shape)