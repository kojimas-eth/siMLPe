import argparse
import os, sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from scipy.spatial.transform import Rotation as R
from zed_finetune.simple_prediction import constrain_prediction, SKELETON_EDGES_22
from  utils.kalman_filter import GlobalTrajectoryExtrapolator
from  utils.extended_KF import MEKFTrajectoryExtrapolator
import torch
import json
import time 

import numpy as np
from zed_finetune.vel_config  import config
from zed_finetune.velocity_model import siMLPe as Model

from datasets.h36m_eval import H36MEval

joint_used_xyz = np.array([2,3,4,5,7,8,9,10,12,13,14,15,17,18,19,21,22,25,26,27,29,30]).astype(np.int64)


def get_dct_matrix(N):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m

def zed34_to_h36m32(zed_data):
    """
    Converts ZED BODY_34 format to Human3.6M 32-joint format. Only contrains 22
    
    Args:
        zed_data: Numpy array of shape (Frames, 34, 3) 
                 
    Returns:
        h36m_data: Numpy array of shape (Frames, 32, 3)
    """
    frames = zed_data.shape[0]
    h36m_data = np.zeros((frames, 32, 3))

    # --- CENTER BODY (Root) ---
    h36m_data[:, 0, :] = zed_data[:, 0, :]   # Pelvis -> Hip (Root)

    # --- RIGHT LEG ---
    h36m_data[:, 1, :] = zed_data[:, 22 ,:]  # R Hip
    h36m_data[:, 2, :] = zed_data[:, 23, :]  # R Knee
    h36m_data[:, 3, :] = zed_data[:, 24, :]  # R Ankle
    h36m_data[:, 4, :] = zed_data[:, 25, :]  # R toe base


    r_ankle = zed_data[:, 24, :]
    r_heel  = zed_data[:, 33, :]
    r_vec = r_ankle - r_heel
    r_vec[:, 1] = 0 

    r_norm = np.linalg.norm(r_vec, axis=1, keepdims=True) + 1e-6
    r_dir_flat = r_vec / r_norm
    r_toe = r_ankle + (r_dir_flat * 0.15)
    r_toe[:, 1] = r_heel[:, 1] 
    h36m_data[:, 5, :] = r_toe

    
    # --- LEFT LEG ---
    h36m_data[:, 6, :] = zed_data[:, 18, :]  # L Hip
    h36m_data[:, 7, :] = zed_data[:, 19, :]  # L Knee
    h36m_data[:, 8, :] = zed_data[:, 20, :]  # L Ankle
    h36m_data[:, 9, :] = zed_data[:, 21, :]  # L sites
    
    l_ankle = zed_data[:, 20, :]
    l_heel  = zed_data[:, 32, :]
    l_vec = l_ankle - l_heel
    l_vec[:, 1] = 0 

    l_norm = np.linalg.norm(l_vec, axis=1, keepdims=True) + 1e-6
    l_dir_flat = l_vec / l_norm
    l_toe = l_ankle + (l_dir_flat * 0.15)
    l_toe[:, 1] = l_heel[:, 1] 

    h36m_data[:, 10, :] = l_toe

    # --- SPINE & HEAD ---
    h36m_data[:, 11, :] = zed_data[:, 1, :]  # Spine Navel -> Spine Low
    h36m_data[:, 12, :] = zed_data[:, 1, :]  # Spine Chest -> Spine Torso
    h36m_data[:, 13, :] = zed_data[:, 3, :]  # Neck -> Neck
    h36m_data[:, 14, :] = zed_data[:, 27, :] # Head -> Head (Nose/Jaw)
    h36m_data[:, 15, :] = (zed_data[:, 30, :] + zed_data[:,28,:])/2 # Nose -> Site (Head Top approximate)
    
    eye_midpoint = (zed_data[:, 30, :] + zed_data[:, 28, :]) / 2
    neck_pos = zed_data[:, 3, :] # You need to verify your Neck index!
    vec_neck_to_eyes = eye_midpoint - neck_pos

    head_up_norm = np.linalg.norm(vec_neck_to_eyes, axis=1, keepdims=True) + 1e-6
    head_up_dir = vec_neck_to_eyes / head_up_norm

    # 4. Extrapolate to Head Top
    FOREHEAD_HEIGHT = 0.12 
    h36m_data[:, 15, :] = eye_midpoint + (head_up_dir * FOREHEAD_HEIGHT)

    # --- LEFT ARM ---
    h36m_data[:, 16, :] = zed_data[:, 4, :]  # L Clavicle
    h36m_data[:, 17, :] = zed_data[:, 5, :]  # L Shoulder
    h36m_data[:, 18, :] = zed_data[:, 6, :]  # L Elbow
    h36m_data[:, 19, :] = zed_data[:, 7, :]  # L Wrist
    h36m_data[:, 20, :] = zed_data[:, 10, :]
    h36m_data[:, 21, :] = zed_data[:, 8, :]  # L Hand (Palm)
    h36m_data[:, 22, :] = zed_data[:, 9, :]
    h36m_data[:, 23, :] = zed_data[:, 9, :]

    # --- RIGHT ARM ---
    h36m_data[:, 24, :] = zed_data[:, 11, :]  # R Clavicle
    h36m_data[:, 25, :] = zed_data[:, 12, :] # R Shoulder
    h36m_data[:, 26, :] = zed_data[:, 13, :] # R Elbow
    h36m_data[:, 27, :] = zed_data[:, 14, :] # R Wrist
    h36m_data[:, 28, :] = zed_data[:, 17, :] # R Wrist
    h36m_data[:, 29, :] = zed_data[:, 15, :] # R Hand (Palm)
    h36m_data[:, 30, :] = zed_data[:, 16, :] # 
    h36m_data[:, 31, :] = zed_data[:, 16, :]
    return h36m_data

############################
#Load Model
###########################
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model-pth', type=str, default="/home/sosuke/thesis/siMLPe/exps/zed_finetune/log/snapshot/velocity_model_44000.pth", help='=encoder path')
args = parser.parse_args()

model = Model(config)

state_dict = torch.load(args.model_pth, weights_only=True, map_location="cpu")
converted_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("motion_transformer.transformer"):
        new_k = k.replace("motion_transformer.transformer", "motion_mlp.mlps")
        converted_state_dict[new_k] = v
    else:
        converted_state_dict[k] = v
model.load_state_dict(converted_state_dict, strict=True)
model.eval()
model.cuda()
config.motion.h36m_target_length = config.motion.h36m_target_length_eval

dct_m_np, idct_m_np = get_dct_matrix(config.motion.h36m_input_length_dct) # usually 50
dct_m = torch.tensor(dct_m_np).float().cuda().unsqueeze(0)
idct_m = torch.tensor(idct_m_np).float().cuda().unsqueeze(0)

####################################################################################
#Load ZED data
###################################################################################
source_number = "10"
source_file = f"still_walk_{source_number}"
suffix = "vel" #finetune, original, constrain
constrain = False

extreme = True
SCALE_FACTOR = 2.5 #how much to exaggerate input movement for extreme test
VEL_SCALE = 2.5  # Neural Network's velocity scale
YAW_SCALE = 2.0
LIMB_SCALE=1.0


# source = f"/home/sosuke/thesis/siMLPe/data/zed_data/{source_file}.json"
# source = f"/home/sosuke/thesis/siMLPe/data/jan22_extradata/{source_file}.json"
# source = f"/home/sosuke/thesis/siMLPe/data/training_data/{source_file}.json"
source = f"/home/sosuke/thesis/siMLPe/data/world_data/still_cam_3_7/{source_file}.json"
# source = f"/home/sosuke/thesis/siMLPe/data/{source_file}.json"
if source.endswith('.json') or source.endswith('.jsonl'):
    with open(source, 'r') as file:
        data = json.load(file)

timestamps = sorted([entry['timestamp'] for entry in data.values()])

all_inputs_saved = []
all_preds_saved = []
zeroed_input=[]
zeroed_output=[]

history_buffer = []
heading_list = []
vel_list = []
MAX_SINGLE_JOINT_MOVE=0.3
MAX_AVG_BODY_MOVE=0.3

error_time = 0
for timestamp_key, frame_data in data.items():
    error_time+=1
    for person in frame_data["body_list"]:

        glitch = False
        keypoints=person.get("keypoint", [])
        keypoints_array = np.array(keypoints) #(34,3)
        heading_quat = person.get("global_root_orientation", [0,0,0,1])
        heading_quat_array = np.array(heading_quat) #(4,)
        heading_list.append(heading_quat_array)

        curr_vel = person.get("velocity", [0,0,0])
        vel_list.append(curr_vel)

        if np.isnan(keypoints_array).any():
            print(f"NaN detected at timestamp {error_time}")

        # --- A. ZERO-JOINTS CHECK ---
        zero_joints = np.all(keypoints_array == 0, axis=1) 
        if zero_joints.any():
            # Get the indices of the bad joints
            bad_joint_indices = np.where(zero_joints)[0]
            print(f"⚠️ Zero-joints found at {error_time}. Indices: {bad_joint_indices}")


        if len(history_buffer) > 0:
            #--- B. Checking motion of joints, copy previous frame if too fast ---
            prev_frame = history_buffer[-1] 
            
            # Calculate Euclidean distance for each joint
            distances = np.linalg.norm(keypoints_array - prev_frame, axis=1)
            
            # A. Check for single joint exploding
            max_dist = np.max(distances)
            if max_dist > MAX_SINGLE_JOINT_MOVE:
                print(f"⚠️ Fast motion detected at {error_time}!")
                print(f"   -> Max joint moved {max_dist:.2f}m. Indices: {np.where(distances > MAX_SINGLE_JOINT_MOVE)[0]}")
                # glitch = True

            # B. Check for whole body teleporting (Camera lost tracking)
            avg_dist = np.mean(distances)
            if avg_dist > MAX_AVG_BODY_MOVE:
                print(f"⚠️ Whole body teleported at {error_time}! Avg move: {avg_dist:.2f}m")
                # glitch = True

        if not glitch:
            history_buffer.append(keypoints_array)
        else:
            history_buffer.append(prev_frame)          

        
        if len(history_buffer)> 50:
            history_buffer.pop(0)
            heading_list.pop(0)
            vel_list.pop(0)
        
        if len(history_buffer) == 50:
            
            past_frames_zed = np.array(history_buffer) # (50, 34, 3)
            if extreme:
            #Make the input extreme in case of testing the limit of model!
                root_idx = 0 
                start_xy = past_frames_zed[0:1, root_idx, :2]
                current_xy = past_frames_zed[:, root_idx, :2]
                movement_delta = current_xy - start_xy
                
                # 2. Calculate the extra travel needed and broadcast to all 34 joints
                extra_travel = movement_delta * (SCALE_FACTOR - 1.0)
                past_frames_zed[:, :, :2] += extra_travel[:, np.newaxis, :]

            h36m_32 = zed34_to_h36m32(past_frames_zed) # (50, 32, 3)
            input_absolute_22 = h36m_32[:, joint_used_xyz, :]
            all_inputs_saved.append(input_absolute_22)

            # Prepare Centered Input (For 'zeroed_input' & Model)
            root = h36m_32[:, 0:1, :] # Pelvis
            h36m_rooted = h36m_32 - root
            
            #Store the heading PRE-flipping of Y
            zed_quats_np = np.array(heading_list) # (50, 4)
            rotations = R.from_quat(zed_quats_np)

            #H36M Treats Facing forward as +Y whilst zed is -Y!!!
            correction_rot = R.from_euler('y', 180, degrees=True)
            corrected_inv_rotations = correction_rot * rotations.inv()

            inv_rot_matrices = corrected_inv_rotations.as_matrix()
            h36m_derotated = np.einsum('tij,tkj->tki', inv_rot_matrices, h36m_rooted)

            input_centered_22 = h36m_derotated[:, joint_used_xyz, :]
            zeroed_input.append(input_centered_22)

            # Setup Auto-Regressive Variables
            TOTAL_HORIZON = 60   
            PRED_STEP = 10       
            num_iterations = TOTAL_HORIZON // PRED_STEP
            

            dt = 1.0 / 25.0 #TODO: Changed from 25
            
            current_global_pos = past_frames_zed[-1, 0, :].copy() 
            
            # Freeze the last known global rotation for the prediction horizon
            last_global_quat = zed_quats_np[-1]
            initial_rot = R.from_quat(last_global_quat)
            
            # Extract Yaw (0), Pitch (1), and Roll (2) in radians
            initial_euler = initial_rot.as_euler('yxz', degrees=False)
            current_yaw = initial_euler[0]
            fixed_pitch = initial_euler[1]
            fixed_roll = initial_euler[2]
            
            future_rotations = [None] * TOTAL_HORIZON 
            future_global_pos = np.zeros((TOTAL_HORIZON, 3))


            input_tensor = torch.tensor(input_centered_22).float().cuda()
            input_tensor = input_tensor.reshape(1, 50, -1) 

            full_prediction_abs = []
            full_prediction_centered = []
            
            correction_inv = correction_rot.inv()

            # 2. Auto-Regressive Loop
            with torch.no_grad():
                for i in range(num_iterations):
                    # DCT
                    if config.deriv_input:
                        input_dct = torch.matmul(dct_m[:, :, :config.motion.h36m_input_length], input_tensor)
                    else:
                        input_dct = input_tensor.clone()

                    pred_dct, pred_vel_dct, pred_yaw_dct = model(input_dct)
                    
                    pred_std = torch.matmul(idct_m[:, :config.motion.h36m_input_length, :], pred_dct) 
                    pred_std = pred_std[:, :PRED_STEP, :] 

                    if config.deriv_output:
                        pred_std = pred_std * LIMB_SCALE
                        pred_std = pred_std + input_tensor[:, -1:, :].repeat(1, PRED_STEP, 1)

                    # Update Buffer with Pose
                    input_tensor = torch.cat([input_tensor[:, PRED_STEP:, :], pred_std], dim=1)

                    #Vel IDCT
                    pred_vel_std = torch.matmul(idct_m[:, :config.motion.h36m_input_length, :], pred_vel_dct)
                    pred_vel_std = pred_vel_std[:, :PRED_STEP, :] # Shape (1, 10, 2)
                    vel_numpy = pred_vel_std[0].cpu().numpy() # Shape (10, 2)
                    
                    # Yaw Rate IDCT
                    pred_yaw_std = torch.matmul(idct_m[:, :config.motion.h36m_input_length, :], pred_yaw_dct)
                    pred_yaw_std = pred_yaw_std[:, :PRED_STEP, :] 
                    yaw_numpy = pred_yaw_std[0].cpu().numpy() # Shape (10, 1)

                    for t in range(PRED_STEP):
                        abs_t = (i * PRED_STEP) + t
                        
                        # Position Integration 
                        real_vel_x = -vel_numpy[t, 0] * VEL_SCALE
                        real_vel_z = -vel_numpy[t, 1] * VEL_SCALE
                        
                        current_global_pos[0] += real_vel_x * dt
                        current_global_pos[2] += real_vel_z * dt
                        future_global_pos[abs_t] = current_global_pos.copy()

                        #Yaw Integration (Apply YAW_SCALE)
                        boosted_yaw_rate = yaw_numpy[t, 0] * YAW_SCALE
                        current_yaw += boosted_yaw_rate * dt
                        
                        # Rebuild the rotation
                        new_euler = [current_yaw, fixed_pitch, fixed_roll]
                        new_rot = R.from_euler('yxz', new_euler, degrees=False)
                        future_rotations[abs_t] = new_rot

                    # --- Post-Process Pose to Global Space ---
                    pred_numpy_centered = pred_std.cpu().numpy().reshape(PRED_STEP, 22, 3)
                    
                    if constrain:
                        pred_numpy_centered = constrain_prediction(input_absolute_22[-1], pred_numpy_centered)

                    full_prediction_centered.append(pred_numpy_centered)
                    
                    final_global_poses = np.zeros_like(pred_numpy_centered)

                    for t in range(PRED_STEP):
                        abs_t = (i * PRED_STEP) + t
                        final_rot = future_rotations[abs_t] * correction_inv
                        rot_matrix = final_rot.as_matrix()
                        rotated_pose = (rot_matrix @ pred_numpy_centered[t].T).T 
                        final_global_poses[t] = rotated_pose + future_global_pos[abs_t]
                        
                    full_prediction_abs.append(final_global_poses)

            # 4. Final Save
            final_pred_abs = np.concatenate(full_prediction_abs, axis=0)
            final_pred_centered = np.concatenate(full_prediction_centered, axis=0)
            
            all_preds_saved.append(final_pred_abs)
            zeroed_output.append(final_pred_centered)

                # time.sleep(2)
            if len(all_preds_saved) % 50 == 0:
                print(f"Predicted {len(all_preds_saved)} windows so far...")



# ---------------------------------------------------------
# 5. SAVE RESULTS
# ---------------------------------------------------------
print("Saving results to disk...")
all_inputs_saved = np.array(all_inputs_saved) # Shape (N, 50, 22, 3)
all_preds_saved = np.array(all_preds_saved)   # Shape (N, 25, 22, 3)

folder_name = f"vel_model/{suffix}"
# Create directory
os.makedirs(folder_name, exist_ok=True)

# Join path safely

if extreme:
    suffix += f"_scale{SCALE_FACTOR}"
save_path = os.path.join(folder_name, f"{source_file}_{suffix}.npz")

np.savez(save_path, 
         inputs=all_inputs_saved, 
         preds=all_preds_saved,
         zero_input = np.array(zeroed_input),
         zero_output = np.array(zeroed_output)
         )

print(f"Done! Saved {len(all_preds_saved)} samples to {save_path}_{source_file}.npz")


