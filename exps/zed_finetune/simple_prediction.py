import argparse
import os, sys
from scipy.spatial.transform import Rotation as R
import torch
import json
import time 

import numpy as np
from config  import config


joint_used_xyz = np.array([2,3,4,5,7,8,9,10,12,13,14,15,17,18,19,21,22,25,26,27,29,30]).astype(np.int64)
INTERPOLATION_RANGE = 10
PREDICTION_FRAMES = 60

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


###########################
#Load ZED data
###########################
source_number = 1
source_file = f"34f_arm_{source_number}"
# source = f"/home/sosuke/thesis/siMLPe/data/zed_data/{source_file}.json"
source = f"/home/sosuke/thesis/siMLPe/data/jan16/{source_file}.json"

if source.endswith('.json') or source.endswith('.jsonl'):
    with open(source, 'r') as file:
        data = json.load(file)

timestamps = sorted([entry['timestamp'] for entry in data.values()])
timestamps_sec = np.array(timestamps) / 1e9
deltas = np.diff(timestamps_sec)
median_delta = np.median(deltas)
calculated_fps = 1.0 / median_delta

print(f"Total Frames: {len(timestamps)}")
print(f"Min Delta: {np.min(deltas) * 1000:.2f} ms")
print(f"Median Delta: {median_delta * 1000:.2f} ms")
print(f"Estimated FPS: {calculated_fps:.2f}")

# Heuristic check
if 28 < calculated_fps < 32:
    print("--> This is likely 30 FPS data. Usually comes from fast quality setting")
elif 58 < calculated_fps < 62:
    print("--> This is likely 60 FPS data.")
elif 14 < calculated_fps < 16:
    print("--> This is likely 15 FPS data. Usually comes from both medium quality setting.") 


all_inputs_saved = []
all_preds_saved = []
zeroed_input=[]
zeroed_output=[]

history_buffer = []
MAX_SINGLE_JOINT_MOVE=0.3
MAX_AVG_BODY_MOVE=0.3

error_time = 0
for timestamp_key, frame_data in data.items():
    error_time+=1
    for person in frame_data["body_list"]:

        glitch = False
        keypoints=person.get("keypoint", [])
        keypoints_array = np.array(keypoints) #(34,3)

        if np.isnan(keypoints_array).any():
            print(f"NaN detected at timestamp {error_time}")
            glitch = True

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

        
        if len(history_buffer)> INTERPOLATION_RANGE:
            history_buffer.pop(0)
        
        if len(history_buffer) == INTERPOLATION_RANGE:
            
            # 1. Prepare Data
            past_frames_zed = np.array(history_buffer) # (Range, 34, 3)
            
            # Map ZED(34) -> H36M(32)
            h36m_32 = zed34_to_h36m32(past_frames_zed) # (Range, 32, 3)

            # A. Prepare Input (For 'all_inputs_saved')
            input_absolute_22 = h36m_32[:, joint_used_xyz, :]
            all_inputs_saved.append(input_absolute_22)
            
            root = h36m_32[:, 0:1, :] # Pelvis
            h36m_rooted = h36m_32 - root

            input_centered_22 = h36m_rooted[:, joint_used_xyz, :] # (Range, 22, 3)
            zeroed_input.append(input_centered_22)


            vel=np.diff(input_absolute_22,axis=0) #(Range-1,22,3)
            acc= np.diff(vel, axis=0)    #(Range-2,22,3)

            avg_velocity=np.mean(vel, axis=0)
            avg_acceleration=np.mean(acc,axis=0)

            predicted_frames =[]
            last_frame= input_absolute_22[-1]
            for i in range(1, PREDICTION_FRAMES +1) :
                displacement = (avg_velocity*i) + (0.5 * avg_acceleration*(i**2))
                next_pose = last_frame + displacement
                predicted_frames.append(next_pose)

            predicted_frames = np.array(predicted_frames) 
            predicted_frames_centered = predicted_frames - root[-1, :]

            all_preds_saved.append(predicted_frames)
            zeroed_output.append(predicted_frames_centered)

            if len(all_preds_saved) % 50 == 0:
                print(f"Predicted {len(all_preds_saved)} windows so far...")



# ---------------------------------------------------------
# 5. SAVE RESULTS
# ---------------------------------------------------------
print("Saving results to disk...")
all_inputs_saved = np.array(all_inputs_saved) # Shape (N, 50, 22, 3)
all_preds_saved = np.array(all_preds_saved)   # Shape (N, 25, 22, 3)

print("output shapes are ", all_inputs_saved.shape, all_preds_saved.shape)

folder_name = "predictions"
# Create directory
os.makedirs(folder_name, exist_ok=True)

# Join path safely
save_path = os.path.join(folder_name, f"{source_file}_interp.npz")

np.savez(save_path, 
         inputs=all_inputs_saved, 
         preds=all_preds_saved,
         zero_input = np.array(zeroed_input),
         zero_output = np.array(zeroed_output)
         )

print(f"Done! Saved {len(all_preds_saved)} samples to zed_inference_results.{source_file}.npz")
