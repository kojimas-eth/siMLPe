import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib import animation as animation
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
    
    (13, 17), (17, 18), (18, 19), (19, 21), (21, 22), # Hand chain
    
    # Right Arm
    (13, 25), (25, 26), (26, 27), (27, 29), (29, 30)  # Hand chain
]

JOINT_NAMES = {
    0:  "Hip (Root)",
    1:  "RHip",
    2:  "RKnee",
    3:  "RFoot (Ankle)",
    4:  "RThumb (Site)",      
    5:  "RSite (Toes)",      
    6:  "LHip",
    7:  "LKnee",
    8:  "LFoot (Ankle)",
    9:  "LThumb (Site)",      
    10: "LSite (Toes)",       
    11: "Spine (Low)",        
    12: "Spine (Torso)",
    13: "Neck, Thorax",
    14: "Head (Nose/Jaw) (Neck/nose)",
    15: "Site (Head Top)",
    16: "LCollar",            
    17: "LShoulder",
    18: "LElbow",
    19: "LWrist",
    20: "LThumb",             
    21: "LHand",
    22: "LFinger",            
    23: "LTip",               
    24: "RCollar",            
    25: "RShoulder",
    26: "RElbow",
    27: "RWrist",
    28: "RThumb",             
    29: "RHand",
    30: "RFinger",            
    31: "RTip"                
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
    11: "Site (Head Top)",           
    12: "LShoulder",
    13: "LElbow",
    14: "LWrist",          
    15: "LHand",
    16: "LFinger",           
    17: "RShoulder",
    18: "RElbow",
    19: "RWrist",        
    20: "RHand",
    21: "RFinger"           
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
    dists = np.linalg.norm(diffs, axis=-1)  # Shape: (22, 13)
    
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

def plot_trajectory_zed(vis_input, vis_target, vis_pred,plot_length=10, highlight_frame=10):
    fig = plt.figure(figsize=(14,8))
    ax = fig.add_subplot(projection='3d')
    error_summary = []
    # # Plot Input Trajectory
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
    for frame in range(plot_length+1):
        xs = vis_target[frame,:,0]
        ys = vis_target[frame,:,2]
        zs = vis_target[frame,:,1]

        for p1, p2 in SKELETON_EDGES_22:
            if frame == highlight_frame:
                ax.plot([xs[p1], xs[p2]], [ys[p1], ys[p2]], [zs[p1], zs[p2]], color='green', alpha=0.5, linewidth=1)
            else:
                ax.plot([xs[p1], xs[p2]], [ys[p1], ys[p2]], [zs[p1], zs[p2]], color='green', alpha=0.1,linewidth=0.5)

    # Plot Predicted Trajectory
    for frame in range(plot_length+1):
        xs = vis_pred[frame,:,0]
        ys = vis_pred[frame,:,2]
        zs = vis_pred[frame,:,1]

        for p1, p2 in SKELETON_EDGES_22:
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
    if root:
        ax.set_title(f'Studying frame {sample} with bodies zeroed')
    else:
        ax.set_title(f'Studying frame {sample} with displacement interpolated')

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

def plot_entire_sequence(input,frames_to_plot):
    fig = plt.figure(figsize=(14,8))
    ax = fig.add_subplot(projection='3d')
    start = frames_to_plot[0]
    stop = frames_to_plot[1] + 1
    num_frames = stop-start
    label_joint_idx = 21 #RHand
    # Plot Input Trajectory
    for i, time in enumerate( range(start,stop)):
        color = plt.cm.coolwarm(i / num_frames)
        vis_input = input[time]

        xs = vis_input[-1,:,0]
        ys = vis_input[-1,:,2] #flip
        zs = vis_input[-1,:,1]

        for p1, p2 in SKELETON_EDGES_22:
            ax.plot([xs[p1], xs[p2]], [ys[p1], ys[p2]], [zs[p1], zs[p2]], color=color, alpha=0.9,linewidth=0.5)
            # if time%10 == 0:
            #     ax.plot([xs[p1], xs[p2]], [ys[p1], ys[p2]], [zs[p1], zs[p2]], color='red', alpha=0.4, linewidth=1)
            # else:
            #     ax.plot([xs[p1], xs[p2]], [ys[p1], ys[p2]], [zs[p1], zs[p2]], color=color, alpha=0.5,linewidth=0.5)
        
        ax.text(xs[label_joint_idx], ys[label_joint_idx], zs[label_joint_idx] + 0.05, 
                str(time), color='black', fontsize=8)
        
    ax.legend()
    ax.set_xlabel('X (Lateral)')
    ax.set_ylabel('Y (Depth)')
    ax.set_zlabel('Z (Height)')
    if root:
        ax.set_title(f'Studying frames {start} to {stop-1} with bodies zeroed')
    else:
        ax.set_title(f'Studying frames {start} to {stop-1} with displacement interpolated')

    legend_elements = [
    Line2D([0], [0], color='blue', lw=2, label='Past Frame (History)')]
 
    # Add the legend to the plot
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 0.8), borderaxespad=0.)
    plt.show()

def plot_2d_trajectory(vis_input, vis_target, vis_pred):
    x_gt = np.concatenate([vis_input[:,:,0], vis_target[:,:,0]], axis=0) 
    y_gt = np.concatenate([vis_input[:,:,2], vis_target[:,:,2]], axis=0)
    z_gt = np.concatenate([vis_input[:,:,1], vis_target[:,:,1]], axis=0)
    x_pred = np.concatenate([vis_input[:,:,0], vis_pred[:,:,0]], axis=0)
    y_pred = np.concatenate([vis_input[:,:,2], vis_pred[:,:,2]], axis=0)
    z_pred = np.concatenate([vis_input[:,:,1], vis_pred[:,:,1]], axis=0)

    total_frame = vis_input.shape[0] + vis_target.shape[0]
    time = [x for x in range(total_frame)]
    input_length = vis_input.shape[0]



    fig , (ax1,ax2,ax3) = plt.subplots(3,1, figsize=(18,6), sharex=True)
    for j in range(vis_input.shape[1]):
        ax1.plot(time, x_gt[:, j], color='green', alpha=0.3, linewidth=1, label='GT' if j==0 else "")
        ax1.plot(time[input_length-1:], x_pred[input_length-1:, j], color='red', linestyle='--', alpha=0.5, linewidth=1, label='Pred' if j==0 else "")

        # --- Plot Y Coordinate ---
        ax2.plot(time, y_gt[:, j], color='green', alpha=0.3, linewidth=1)
        ax2.plot(time[input_length-1:], y_pred[input_length-1:, j], color='red', linestyle='--', alpha=0.5, linewidth=1)

        # --- Plot Z Coordinate ---
        ax3.plot(time, z_gt[:, j], color='green', alpha=0.3, linewidth=1)
        ax3.plot(time[input_length-1:], z_pred[input_length-1:, j], color='red', linestyle='--', alpha=0.5, linewidth=1)

    for ax, label in zip([ax1, ax2, ax3], ['X Coordinate', 'Y Coordinate', 'Z Coordinate']):
            ax.axvline(x=input_length-1, color='black', linestyle=':', label='Start of Pred')
            ax.set_ylabel(f'{label} value')
            ax.grid(True, alpha=0.3)

    ax1.legend(loc='upper right')
    ax3.set_xlabel('Time (Frames)')
    plt.suptitle(f'2D Joint Trajectories over Time at frame {sample}', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"plots/{source_number}_2d_trajectory_plot_{sample}.png")
    plt.show()
        

def animate_trajectory_zed(vis_input, vis_target, vis_pred, skeleton_edges, interval=100):
    """
    Creates a side-by-side 3D animation:
    - Left: Ground Truth (Input -> Target)
    - Right: Prediction (Input -> Prediction)
    """
    
    # 1. Prepare Data
    # Concatenate input with target/pred to create continuous sequences
    # Input shape: (Frames, Joints, 3)
    seq_gt = np.concatenate([vis_input, vis_target], axis=0)
    seq_pred = np.concatenate([vis_input, vis_pred], axis=0)
    
    input_len = vis_input.shape[0]
    total_frames = seq_gt.shape[0]

    # 2. Setup Figure
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    
    # Titles
    ax1.set_title("Ground Truth (Blue=In, Green=Out)")
    ax2.set_title("Prediction (Blue=In, Red=Out)")
    
    # 3. Helper to determine global axis limits (crucial for stable 3D animation)
    all_data = np.concatenate([seq_gt, seq_pred], axis=0)
    # Note: Using indices 0, 2, 1 based on your original flip logic (x, z, y)
    x_min, x_max = all_data[:, :, 0].min(), all_data[:, :, 0].max()
    y_min, y_max = all_data[:, :, 2].min(), all_data[:, :, 2].max() # Z in data is Y in plot
    z_min, z_max = all_data[:, :, 1].min(), all_data[:, :, 1].max() # Y in data is Z in plot
    
    for ax in [ax1, ax2]:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=20, azim=45) # Optional: Set a nice camera angle

    # 4. Initialize Line Objects
    # We create a list of line objects for every edge in the skeleton
    # This is much faster than calling plot() every frame
    lines_gt = [ax1.plot([], [], [], linewidth=2)[0] for _ in skeleton_edges]
    lines_pred = [ax2.plot([], [], [], linewidth=2)[0] for _ in skeleton_edges]

    def update(frame):
        # Determine current color based on phase (Input vs Output)
        if frame < input_len:
            color_gt = 'blue'
            color_pred = 'blue'
            phase = "Input"
        else:
            color_gt = 'green'
            color_pred = 'red'
            phase = "Future"

        # Update Ground Truth Skeleton (Left)
        current_pose_gt = seq_gt[frame]
        for line, (p1, p2) in zip(lines_gt, skeleton_edges):
            line.set_data([current_pose_gt[p1, 0], current_pose_gt[p2, 0]], 
                          [current_pose_gt[p1, 2], current_pose_gt[p2, 2]]) # Swap Y/Z
            line.set_3d_properties([current_pose_gt[p1, 1], current_pose_gt[p2, 1]])
            line.set_color(color_gt)

        # Update Prediction Skeleton (Right)
        current_pose_pred = seq_pred[frame]
        for line, (p1, p2) in zip(lines_pred, skeleton_edges):
            line.set_data([current_pose_pred[p1, 0], current_pose_pred[p2, 0]], 
                          [current_pose_pred[p1, 2], current_pose_pred[p2, 2]]) # Swap Y/Z
            line.set_3d_properties([current_pose_pred[p1, 1], current_pose_pred[p2, 1]])
            line.set_color(color_pred)
            
        fig.suptitle(f"Frame: {frame} ({phase})", fontsize=14)
        return lines_gt + lines_pred

    # 5. Create Animation
    anim = animation.FuncAnimation(
        fig, update, frames=total_frames, interval=interval, blit=False
    )
    anim.save(f'plots/new_{source_number}_skeleton_animation_{sample}.mp4', writer='ffmpeg', fps=10)
    # anim.save(f'plots/{source_number}_original_source_{sample}.mp4', writer='ffmpeg', fps=10)

##################################
# MAIN CODE
##################################
zed = True #If using data from zed inference
root = True #If using data with everything zeroed
source_number = 2
sample = 10 #which frame to analyze

if zed:
    data = np.load(f"zed_inference_results_{source_number}_norot_extrapolate.npz")
    # data = np.load("sanity_check.npz")
    print(f"Loaded {len(data['inputs'])} samples from zed_inference_results_{source_number}.npz")
else:
    data = np.load("results_dump.npz")
    print(f"Loaded {len(data['inputs'])} samples from results_dump.npz")

frames_to_compute_error = [0, 3, 5, 9]


# Extract arrays
if zed:
    inputs = data['inputs']   # (N, 50, 22, 3)
    preds = data['preds']     # (N, 25, 22, 3)
    # targets = inputs[sample+25, -25:, :, :]  # (25, 22, 3)
    targets = preds
    vis_target = targets[0]


    zero_input = data['zero_input']   # (N, 50, 22, 3)
    zero_output = data['zero_output'] # (N, 25, 22, 3)
    zero_target = zero_input[sample+25, -25:, :, :]  # (25, 22, 3)

else:
    inputs = data['inputs']   # (3840, 50, 32, 3)
    targets = data['targets'] # (N, 25, 32, 3)
    preds = data['preds']     # (N, 25, 32, 3)
    
    vis_target= targets[sample] # (25, 32, 3)



input_time = inputs.shape[1]
pred_time = preds.shape[1]
num_joints = inputs.shape[2]

if root:
    vis_input= zero_input[sample]   # (50, 22, 3)
    vis_pred= zero_output[sample]     # (25, 22, 3  )
    vis_target = zero_target
else:
    vis_input=  inputs[sample]   # (50, 32, 3)
    vis_pred= preds[sample]     # (25, 32, 3)

#only take frist 10 frames
plot_trajectory_zed(vis_input,vis_target,vis_pred,plot_length=10, highlight_frame=3)
vis_target = vis_target[0:10, :, :]
vis_pred = vis_pred[0:10, :, :]
# print(vis_input.shape, vis_target.shape, vis_pred.shape)

plot_2d_trajectory(vis_input, vis_target, vis_pred)
animate_trajectory_zed(vis_input, vis_target, vis_pred, SKELETON_EDGES_22)

# plot_entire_sequence(zero_input, [max(0,sample), min(sample +15, len(zero_input)-1)])   



# plot_trajectory_h36m(vis_input, vis_target, vis_pred)
# animate_trajectory_zed(vis_input, vis_target, vis_pred, H36M_FULL_EDGES)
# plot_single_skeleton_zed(vis_input)