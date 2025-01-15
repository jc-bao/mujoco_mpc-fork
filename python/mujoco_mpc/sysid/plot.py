import numpy as np
import matplotlib.pyplot as plt

data = np.load("h1_data.npz")
q_buffer = data["q"]
qd_buffer = data["qd"]
ctrl_buffer = data["ctrl"]

plt.plot(ctrl_buffer[:, 2])
plt.show()