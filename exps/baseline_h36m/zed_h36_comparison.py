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

########################
#Loading zed data #1
########################
# zed_data =np.load(f"predictions/constrain_original/34f_arm_4_constrain_original.npz")
zed_data =np.load(f"predictions/original/34f_test_tilt_world_original.npz")
zed_input = zed_data['inputs']
zero_input = zed_data['zero_input']   # (N, 50, 22, 3)
zed_zero_input = zero_input[50]
zed_zero_input_22= zed_zero_input[-1]
print("zed input", zed_zero_input_22.shape)

zed_output = zed_data['zero_output']
zed_zero_output = zed_output[100]
zed_zero_output_22= zed_zero_output[10]
print("zed output", zed_zero_output_22.shape)


########################
#Loading zed data #2
########################
# zed_data2 =np.load(f"predictions/finetune/34f_test_world_finetune.npz")
zed_data2 =np.load(f"predictions/original/34f_test_tilt_cam_original.npz")
zed_input2 = zed_data2['inputs']
zero_input2 = zed_data2['zero_input']   # (N, 50, 22, 3)
zed_zero_input2 = zero_input2[50]
zed_zero_input_22_2= zed_zero_input2[-1]
print("zed 2 input", zed_zero_input_22_2.shape)

zed_output2 = zed_data2['zero_output']
zed_zero_output2 = zed_output2[0]
zed_zero_output_22_2= zed_zero_output2[10]
print("zed 2 output", zed_zero_output_22_2.shape)

########################
#Loading H36M data
########################
dataset = H36MEval(config, 'train')
print("H36M dataset length:", len(dataset))
input,output = dataset[1000]
input2,output2 = dataset[3000]
print("input shape",input.shape)

h36_input = input[0,:,:]
h36_output = output[0,:,:]
h36_22_input = h36_input[joint_used_xyz,:]
h36_22_output = h36_output[joint_used_xyz,:]
h36_input2 = input2[0,:,:]
h36_22_input2 = h36_input2[joint_used_xyz,:]

print("h36 input", h36_22_input.shape)



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

def plot_skeleton(ax, data, title, color='blue', label=None, edges=SKELETON_EDGES_22):
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
    ax.scatter(xs, ys, zs, c='red', s=20, marker='o',label=label)

    # 2. Plot Bones (Lines)
    for start, end in edges:
        if start < len(data) and end < len(data):
            ax.plot([xs[start], xs[end]], 
                    [ys[start], ys[end]], 
                    [zs[start], zs[end]], c=color, alpha = 0.7)
        
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

#Define what you want to plot
plot_1 = zed_zero_output_22
# plot_2 = h36_22_input
# plot_2 = zed_zero_input_22_2
plot_3=h36_22_output

# Ensure both are numpy for the limit calculation
d1,d2,d3 = None,None,None
d1 = plot_1.detach().cpu().numpy() if isinstance(plot_1, torch.Tensor) else plot_1
# d2 = plot_2.detach().cpu().numpy() if isinstance(plot_2, torch.Tensor) else plot_2
d3 = plot_3.detach().cpu().numpy() if isinstance(plot_3, torch.Tensor) else plot_3

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
if d1 is not None:
    plot_skeleton(ax, d1, "ZED world (Red) vs ZED cam (Blue)", color='red', label='ZED Data')
if d2 is not None:
    plot_skeleton(ax, d2, "ZED world (Red) vs ZED cam (Blue)", color='blue', label='H36M Data')
if d3 is not None:
    plot_skeleton(ax, d3, "ZED (Red) vs H36M (Green)", color='green', label='ZED Output Data')

valid_data = [d for d in [d1,d2,d3] if d is not None]
all_data = np.vstack(valid_data)

max_range = np.array([
    all_data[:,0].max() - all_data[:,0].min(), 
    all_data[:,1].max() - all_data[:,1].min(), 
    all_data[:,2].max() - all_data[:,2].min()
]).max() / 2.0

mid_x = (all_data[:,0].max() + all_data[:,0].min()) * 0.5
mid_y = (all_data[:,1].max() + all_data[:,1].min()) * 0.5
mid_z = (all_data[:,2].max() + all_data[:,2].min()) * 0.5

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

plt.tight_layout()
plt.show()