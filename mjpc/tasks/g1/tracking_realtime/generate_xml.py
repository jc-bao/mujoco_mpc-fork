import matplotlib.pyplot as plt
import numpy as np

mocap_objects = [
    "pelvis", "head", "ltoe", "rtoe", "lheel", "rheel", "lknee", "rknee",
    "lhand", "rhand", "lelbow", "relbow", "lshoulder", "rshoulder", "lhip", "rhip"
]

num_repetitions = 50
colors = [plt.cm.coolwarm(i / (num_repetitions - 1)) for i in range(num_repetitions)]

with open("mocap_output.xml", "w") as f:
    for i in range(num_repetitions):
        rgba = f'{colors[i][0]:.3f} {colors[i][1]:.3f} {colors[i][2]:.3f} 1'
        for obj in mocap_objects:
            f.write(f'    <body name="mocap[{obj}]{i}" mocap="true">\n')
            f.write(f'      <site name="mocap[{obj}]{i}" class="mocap_site" rgba="{rgba}"/>\n')
            f.write('    </body>\n')

data = "-0.08679 0.19292 0.86056 -0.04738 0.30140 1.36626 -0.17402 0.34120 -0.00107 0.00010 0.29688 0.00416 -0.16921 0.17126 -0.00093 -0.03162 0.12988 0.00260 -0.18658 0.21238 0.39614 0.01303 0.19736 0.39839 -0.29056 0.32638 0.73196 0.12556 0.30252 0.71085 -0.25784 0.25329 0.89793 0.10067 0.22799 0.87754 -0.16856 0.24912 1.07646 0.03103 0.23479 1.06491 -0.17502 0.20443 0.70755 -0.04873 0.17857 0.70803 "
data_repeat = data * 1
print(data_repeat)