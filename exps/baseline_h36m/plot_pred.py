import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D 
''' Joint used xyz corresponnd to the following edges
RKnee, RAnkle, RToe, RSite
LKnee, LAnkle, LToe, LSite
Spine, Neck, Head, HeadSite
Lshoulder, Lelbow, Lwrist Lhand Lfinger 
Rshoulder, Relbow, Rwrist, Rhand, Rfinger '''

SKELETON_EDGES_22 = [
    # --- Right Leg (Starts at Knee) ---
    (0, 1),   # RKnee -> RAnkle
    (1, 2),   # RAnkle -> RToe
    (2, 3),   # RToe -> RTip (Optional, might be erratic)

    # --- Left Leg (Starts at Knee) ---
    (4, 5),   # LKnee -> LAnkle
    (5, 6),   # LAnkle -> LToe
    (6, 7),   # LToe -> LTip (Optional)

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

H36M_SKELETON_EDGES = [
    # --- Legs ---
    (0, 1),   # Hip (Root) -> RHip
    (1, 2),   # RHip -> RKnee
    (2, 3),   # RKnee -> RFoot (Ankle)
    (0, 6),   # Hip (Root) -> LHip
    (6, 7),   # LHip -> LKnee
    (7, 8),   # LKnee -> LFoot (Ankle)

    # --- Torso (Spine Chain) ---
    (0, 12),  # Hip (Root) -> Spine (Torso)
    (12, 13), # Spine -> Neck
    (13, 14), # Neck -> Head (Nose)
    (14, 15), # Head -> Site (Head Top) - Optional, gives height to head

    # --- Arms ---
    (13, 17), # Neck -> LShoulder
    (17, 18), # LShoulder -> LElbow
    (18, 19), # LElbow -> LWrist
    
    (13, 25), # Neck -> RShoulder
    (25, 26), # RShoulder -> RElbow
    (26, 27), # RElbow -> RWrist
]

H36M_FULL_EDGES = [
    # Right Leg
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
    
    # Left Leg
    (0, 6), (6, 7), (7, 8), (8, 9), (9, 10),
    
    # Spine
    (0, 12), (12, 13), (13, 14), (14, 15),
    
    # Left Arm (Note: Collar connects Spine to Shoulder in some raw data, 
    # but usually Neck->Shoulder is safer for visual)
    (13, 17), (17, 18), (18, 19), (19, 21), (21, 22), # Hand chain
    
    # Right Arm
    (13, 25), (25, 26), (26, 27), (27, 29), (29, 30)  # Hand chain
]

JOINT_NAMES = {
    0:  "Hip (Root)",
    1:  "RHip",
    2:  "RKnee",
    3:  "RFoot (Ankle)",
    4:  "RThumb (Site)",      # Often unused
    5:  "RSite (Toes)",       # Often unused
    6:  "LHip",
    7:  "LKnee",
    8:  "LFoot (Ankle)",
    9:  "LThumb (Site)",      # Often unused
    10: "LSite (Toes)",       # Often unused
    11: "Spine (Low)",        # Often unused
    12: "Spine (Torso)",
    13: "Neck",
    14: "Head (Nose/Jaw)",
    15: "Site (Head Top)",
    16: "LCollar",            # Often unused
    17: "LShoulder",
    18: "LElbow",
    19: "LWrist",
    20: "LThumb",             # Often unused
    21: "LHand",
    22: "LFinger",            # Often unused
    23: "LTip",               # Often unused
    24: "RCollar",            # Often unused
    25: "RShoulder",
    26: "RElbow",
    27: "RWrist",
    28: "RThumb",             # Often unused
    29: "RHand",
    30: "RFinger",            # Often unused
    31: "RTip"                # Often unused
}

JOINT_NAMES_22 = {
    0:  "RKnee",
    1:  "RFoot (Ankle)",
    2:  "RThumb (Site)",     
    3:  "RSite (Toes)",      
    4:  "LKnee",
    5:  "LFoot (Ankle)",
    6:  "LThumb (Site)",    
    7: "LSite (Toes)",       
    8: "Spine (Torso)",
    9: "Neck",
    10: "Head (Nose/Jaw)",
    11: "Site (Head Top)",           # Often unused
    12: "LShoulder",
    13: "LElbow",
    14: "LWrist",           # Often unused
    15: "LHand",
    16: "LFinger",            # Often unused
    17: "RShoulder",
    18: "RElbow",
    19: "RWrist",         # Often unused
    20: "RHand",
    21: "RFinger"           #
}

def compute_mpjpe(predicted, ground_truth):
    """
    predicted: numpy array of shape (Num_Frames, 22, 3)
    ground_truth: numpy array of shape (Num_Frames, 22, 3)
    Calculates the error of joints at specific indices only.
    """
    # print(f"predicted shape: {predicted.shape}, ground_truth shape: {ground_truth.shape}")
    assert predicted.shape == ground_truth.shape, "Shape mismatch between predicted and ground truth"
    
    # Compute Euclidean distances per joint per frame
    diffs = predicted - ground_truth
    dists = np.linalg.norm(diffs, axis=-1)  # Shape: (Num_Frames, 13)
    
    # Average over all joints and frames
    mpjpe = np.mean(dists)
    return mpjpe

def plot_single_skeleton(vis_input):
    fig = plt.figure(figsize=(14,8))
    ax = fig.add_subplot(111,projection='3d')

    for frame in range(1):
        xs = vis_input[frame,:,0]
        ys = vis_input[frame,:,1]
        zs = vis_input[frame,:,2]
        ax.scatter(xs, ys, zs, color='red', alpha=0.5)
        for p1, p2 in H36M_SKELETON_EDGES:
            ax.plot([xs[p1], xs[p2]], [ys[p1], ys[p2]], [zs[p1], zs[p2]], color='blue', alpha=0.7)
        
        for i in range(32):
            ax.text(xs[i], ys[i], zs[i], 
                    f"{i}: {JOINT_NAMES[i]}", 
                    fontsize=8)
    plt.show()

def plot_single_skeleton_zed(vis_input):
    fig = plt.figure(figsize=(14,8))
    ax = fig.add_subplot(111,projection='3d')

    for frame in range(1):
        xs = vis_input[frame,:,0]
        ys = vis_input[frame,:,1]
        zs = vis_input[frame,:,2]
        ax.scatter(xs, ys, zs, color='red', alpha=0.5)
        for p1, p2 in SKELETON_EDGES_22:
            ax.plot([xs[p1], xs[p2]], [ys[p1], ys[p2]], [zs[p1], zs[p2]], color='blue', alpha=0.7)
        
        for i in range(22):
            ax.text(xs[i], ys[i], zs[i], 
                    f"{i}: {JOINT_NAMES_22[i]}", 
                    fontsize=8)
    plt.show()

def plot_trajectory_h36m(vis_input, vis_target, vis_pred,highlight_frame=24):
    fig = plt.figure(figsize=(14,8))
    ax = fig.add_subplot(projection='3d')
    error_summary = []
    # Plot Input Trajectory
    for frame in range(vis_input.shape[0]):
        xs = vis_input[frame,:,0]
        ys = vis_input[frame,:,2] #flip
        zs = vis_input[frame,:,1]

        for p1, p2 in H36M_SKELETON_EDGES:
            if frame == vis_input.shape[0]-1:
                ax.plot([xs[p1], xs[p2]], [ys[p1], ys[p2]], [zs[p1], zs[p2]], color='blue', alpha=0.5, linewidth=1)
            else:
                ax.plot([xs[p1], xs[p2]], [ys[p1], ys[p2]], [zs[p1], zs[p2]], color='blue', alpha=0.1,linewidth=0.5)
        
    # Plot Target Trajectory
    for frame in range(vis_target.shape[0]):
        xs = vis_target[frame,:,0]
        ys = vis_target[frame,:,2]
        zs = vis_target[frame,:,1]

        for p1, p2 in H36M_SKELETON_EDGES:
            if frame == highlight_frame:
                ax.plot([xs[p1], xs[p2]], [ys[p1], ys[p2]], [zs[p1], zs[p2]], color='green', alpha=0.5, linewidth=1)
            else:
                ax.plot([xs[p1], xs[p2]], [ys[p1], ys[p2]], [zs[p1], zs[p2]], color='green', alpha=0.1,linewidth=0.5)

    # Plot Predicted Trajectory
    for frame in range(vis_pred.shape[0]):
        xs = vis_pred[frame,:,0]
        ys = vis_pred[frame,:,2]
        zs = vis_pred[frame,:,1]

        for p1, p2 in H36M_SKELETON_EDGES:
            if frame == highlight_frame:
                ax.plot([xs[p1], xs[p2]], [ys[p1], ys[p2]], [zs[p1], zs[p2]], color='red', alpha=0.5, linewidth=1)
            else:
                ax.plot([xs[p1], xs[p2]], [ys[p1], ys[p2]], [zs[p1], zs[p2]], color='red', alpha=0.1,linewidth=0.5)

    for err_frame in frames_to_compute_error:
            mpjpe = compute_mpjpe(vis_pred[err_frame], vis_target[err_frame])
            error_summary.append(f"T+{err_frame+1}: {mpjpe:.2f}m")

    ax.legend()
    ax.set_xlabel('X (Lateral)')
    ax.set_ylabel('Y (Depth)')
    ax.set_zlabel('Z (Height)')
    ax.set_title(f'Studying frame {sample}')

    legend_elements = [
    Line2D([0], [0], color='blue', lw=2, label='Past Frame (History)'),
    Line2D([0], [0], color='green', lw=2, label='Ground Truth (Real)'),
    Line2D([0], [0], color='red', lw=2, label='Prediction (Model)') ]

    info_text = f"Prediction Errors (MPJPE):\n" + "\n".join(error_summary)

    fig.text(0.75, 0.60, info_text, 
             fontsize=10, 
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Add the legend to the plot
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 0.8), borderaxespad=0.)
    plt.show()

def plot_trajectory_zed(vis_input, vis_target, vis_pred,highlight_frame=24):
    fig = plt.figure(figsize=(14,8))
    ax = fig.add_subplot(projection='3d')
    error_summary = []
    # Plot Input Trajectory
    for frame in range(vis_input.shape[0]):
        xs = vis_input[frame,:,0]
        ys = vis_input[frame,:,2] #flip
        zs = vis_input[frame,:,1]

        for p1, p2 in SKELETON_EDGES_22:
            if frame == vis_input.shape[0]-1:
                ax.plot([xs[p1], xs[p2]], [ys[p1], ys[p2]], [zs[p1], zs[p2]], color='blue', alpha=0.5, linewidth=1)
            else:
                ax.plot([xs[p1], xs[p2]], [ys[p1], ys[p2]], [zs[p1], zs[p2]], color='blue', alpha=0.1,linewidth=0.5)
        
    # # Plot Target Trajectory
    # for frame in range(vis_target.shape[0]):
    #     xs = vis_target[frame,:,0]
    #     ys = vis_target[frame,:,2]
    #     zs = vis_target[frame,:,1]

    #     for p1, p2 in SKELETON_EDGES_22:
    #         if frame == highlight_frame:
    #             ax.plot([xs[p1], xs[p2]], [ys[p1], ys[p2]], [zs[p1], zs[p2]], color='green', alpha=0.5, linewidth=1)
    #         else:
    #             ax.plot([xs[p1], xs[p2]], [ys[p1], ys[p2]], [zs[p1], zs[p2]], color='green', alpha=0.1,linewidth=0.5)

    # Plot Predicted Trajectory
    for frame in range(vis_pred.shape[0]):
        xs = vis_pred[frame,:,0]
        ys = vis_pred[frame,:,2]
        zs = vis_pred[frame,:,1]

        for p1, p2 in SKELETON_EDGES_22:
            if frame == highlight_frame:
                ax.plot([xs[p1], xs[p2]], [ys[p1], ys[p2]], [zs[p1], zs[p2]], color='red', alpha=0.5, linewidth=1)
            else:
                ax.plot([xs[p1], xs[p2]], [ys[p1], ys[p2]], [zs[p1], zs[p2]], color='red', alpha=0.1,linewidth=0.5)

    # for err_frame in frames_to_compute_error:
    #         mpjpe = compute_mpjpe(vis_pred[err_frame], vis_target[err_frame])
    #         error_summary.append(f"T+{err_frame+1}: {mpjpe:.2f}m")

    ax.legend()
    ax.set_xlabel('X (Lateral)')
    ax.set_ylabel('Y (Depth)')
    ax.set_zlabel('Z (Height)')
    ax.set_title(f'Studying frame {sample}')

    legend_elements = [
    Line2D([0], [0], color='blue', lw=2, label='Past Frame (History)'),
    Line2D([0], [0], color='green', lw=2, label='Ground Truth (Real)'),
    Line2D([0], [0], color='red', lw=2, label='Prediction (Model)') ]

    info_text = f"Prediction Errors (MPJPE):\n" + "\n".join(error_summary)
    
    fig.text(0.75, 0.60, info_text, 
             fontsize=10, 
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Add the legend to the plot
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 0.8), borderaxespad=0.)
    plt.show()

##################################
# MAIN CODE
##################################
zed = True
data = np.load("zed_inference_results.npz")

# Extract arrays
if zed:
    inputs = data['inputs']   # (N, 50, 22, 3)
    preds = data['preds']     # (N, 25, 22, 3)
    targets = 0
else:
    inputs = data['inputs']   # (3840, 50, 32, 3)
    targets = data['targets'] # (N, 25, 32, 3)
    preds = data['preds']     # (N, 25, 32, 3)


print(inputs.shape)
print(preds.shape)
sample = 1
input_time = inputs.shape[1]
pred_time = preds.shape[1]
num_joints = inputs.shape[2]


vis_input=  inputs[sample]   # (50, 32, 3)
# vis_target= targets[sample] # (25, 32, 3)
vis_target =0
vis_pred= preds[sample]     # (25, 32, 3)
frames_to_compute_error = [0, 4, 9, 14, 19, 24]
print(vis_input.shape)
# plot_single_skeleton_zed(vis_input)
plot_trajectory_zed(vis_input, vis_target, vis_pred)    
# plot_single_skeleton(vis_input, vis_target, vis_pred)