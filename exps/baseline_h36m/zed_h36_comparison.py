import argparse
import os, sys
from scipy.spatial.transform import Rotation as R
import torch
import json
import time 

import numpy as np
from config  import config
from model import siMLPe as Model

from datasets.h36m_eval import H36MEval
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D 


joint_used_xyz = np.array([2,3,4,5,7,8,9,10,12,13,14,15,17,18,19,21,22,25,26,27,29,30]).astype(np.int64)


config.motion.h36m_target_length = config.motion.h36m_target_length_eval

#load ZED data
source_number = 2
source = f"/home/sosuke/thesis/siMLPe/data/zed_data/30fps_body34_fast_{source_number}.json"
if source.endswith('.json') or source.endswith('.jsonl'):
    with open(source, 'r') as file:
        zed_data = json.load(file)

zed_data =np.load(f"/home/sosuke/thesis/siMLPe/exps/baseline_h36m/zed_inference_results_{source_number}_norot_extrapolate.npz")

zed_input = zed_data['inputs']
zero_input = zed_data['zero_input']   # (N, 50, 22, 3)

zed_zero_input = zero_input[0]
zed_zero_input_22= zed_zero_input[-1]
print("zed input", zed_zero_input_22.shape)

zed_rotated = zed_zero_input_22.copy()
# 2. Apply the rotation 
# New X = Old X
# zed_rotated[:, 0] = zed_zero_input_22[:, 0]
# # New Y = -Old Z ("negative z of zed should become positive y")
# zed_rotated[:, 1] = -zed_zero_input_22[:, 2]
# # New Z = Old Y ("zed positive y to become h36's positive z")
# zed_rotated[:, 2] = zed_zero_input_22[:, 1]

#Load H36 data
h36data = np.loadtxt("/home/sosuke/thesis/siMLPe/data/h36m/S5/discussion_1.txt", delimiter=",")
print(f"Loaded shape: {h36data.shape}")
dataset = H36MEval(config, 'test')
input,output = dataset[0]

h36_input = input[0,:,:]
h36_22_input = h36_input[joint_used_xyz,:]
print("h36 input", h36_22_input.shape)







# --- 1. Define Connectivity ---
SKELETON_EDGES_22 = [
    # --- Right Leg (Starts at Knee) ---
    (0, 1),   # RKnee -> RAnkle
    (1, 2),   # RAnkle -> RToe
    (2, 3),   # RToe -> RTip 

    # --- Left Leg (Starts at Knee) ---
    (4, 5),   # LKnee -> LAnkle
    (5, 6),   # LAnkle -> LToe
    (6, 7),   # LToe -> LTip 

    # --- Spine & Head ---
    (8, 9),   # Spine (Torso) -> Neck
    (9, 10),  # Neck -> Head
    (10, 11), # Head -> Head Site (Top)

    # --- Left Arm ---
    (9, 12),  # Neck -> LShoulder
    (12, 13), # LShoulder -> LElbow
    (13, 14), # LElbow -> LWrist
    (14, 15), # LWrist -> LHand
    (15, 16), # LHand -> LTip

    # --- Right Arm ---
    (9, 17),  # Neck -> RShoulder
    (17, 18), # RShoulder -> RElbow
    (18, 19), # RElbow -> RWrist
    (19, 20), # RWrist -> RHand
    (20, 21)  # RHand -> RTip
]

def plot_skeleton(ax, data, title, edges=SKELETON_EDGES_22):
    """
    Plots a single skeleton (22 joints).
    data: (22, 3) array-like (numpy or torch)
    """
    # Convert Torch tensor to Numpy if necessary
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
        
    xs = data[:, 0]
    ys = data[:, 1]
    zs = data[:, 2]

    # 1. Plot Joints (Dots)
    ax.scatter(xs, ys, zs, c='red', s=20, marker='o')

    # 2. Plot Bones (Lines)
    for start, end in edges:
        if start < len(data) and end < len(data):
            ax.plot([xs[start], xs[end]], 
                    [ys[start], ys[end]], 
                    [zs[start], zs[end]], c='blue')
        
    # 3. Annotate Indices (useful for debugging)
    for i in range(len(data)):
        ax.text(xs[i], ys[i], zs[i], str(i), fontsize=9, color='black')

    # 4. Formatting
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Auto-scale axes to keep aspect ratio roughly correct
    max_range = np.array([xs.max()-xs.min(), ys.max()-ys.min(), zs.max()-zs.min()]).max() / 2.0
    mid_x = (xs.max()+xs.min()) * 0.5
    mid_y = (ys.max()+ys.min()) * 0.5
    mid_z = (zs.max()+zs.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
# --- 2. Run the Plotting ---

# Assuming your variables from the prompt:
# zed_input (numpy array): shape (22, 3)
# h36_input (torch tensor): shape (22, 3)

# If zed_input is (N, 22, 3) or similar, take the first frame:
# zed_to_plot = zed_input[0] if zed_input.ndim == 3 else zed_input 
# For now, assuming you have the (22,3) frame ready:
# zed_to_plot = zed_zero_input_22  
zed_to_plot = zed_rotated
h36_to_plot = h36_22_input

fig = plt.figure(figsize=(12, 6))
# Plot ZED Input
ax1 = fig.add_subplot(121, projection='3d')
plot_skeleton(ax1, zed_to_plot, "ZED Data (22 Joints)")

# Plot H36M Input
ax2 = fig.add_subplot(122, projection='3d')
plot_skeleton(ax2, h36_to_plot, "H36M Data (22 Joints)")

plt.tight_layout()
plt.show()