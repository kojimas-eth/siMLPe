import argparse
import os, sys
from scipy.spatial.transform import Rotation as R
import torch
import json
import time 

import numpy as np
from config  import config


joint_used_xyz = np.array([2,3,4,5,7,8,9,10,12,13,14,15,17,18,19,21,22,25,26,27,29,30]).astype(np.int64)
INTERPOLATION_RANGE = 50
PREDICTION_FRAMES = 60

def zed34_to_h36m32(zed_data):
    """
    Converts ZED BODY_34 format to Human3.6M 32-joint format. Only contrains 22
    
    Args:
        zed_data: Numpy array of shape (Frames, 34, 3) 
                 
    Returns:
        h36m_data: Numpy array of shape (Frames, 32, 3)
    """
    input_is_single_frame = (zed_data.ndim == 2)
    
    # 2. If single frame, add a temporary batch dimension: (34,3) -> (1,34,3)
    if input_is_single_frame:
        zed_data = zed_data[np.newaxis, ...]

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
    
    if input_is_single_frame:
        h36m_data = np.squeeze(h36m_data, axis=0) # Returns (32, 3)

    return h36m_data
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

class JointKalmanFilter:
    def __init__(self, dt=1.0/15.0): # Assuming 30 FPS, adjust if needed
        # State vector: [x, y, z, vx, vy, vz]
        self.x = np.zeros(6) 
        
        # State Transition Matrix (F)
        # x_new = x + vx*dt
        self.F = np.eye(6)
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt
        
        # Measurement Matrix (H) - We only measure position (x,y,z)
        self.H = np.zeros((3, 6))
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.H[2, 2] = 1
        
        # Covariance Matrix (P) - Uncertainty in our state
        self.P = np.eye(6) * 100 
        
        # Measurement Noise (R) - How noisy is the ZED camera?
        # Increase this if the input jitters a lot.
        self.R = np.eye(3) * 0.1 
        
        # Process Noise (Q) - How much can a human change velocity?
        # Tune this! Higher = filter reacts faster but follows noise.
        # Lower = filter is smoother but lags on sudden moves.
        self.Q = np.eye(6) * 0.01

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:3] # Return predicted position

    def update(self, measurement):
        # Measurement residual (y)
        z = np.array(measurement)
        y = z - (self.H @ self.x)
        
        # Kalman Gain (K)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update State
        self.x = self.x + (K @ y)
        
        # Update Covariance
        self.P = (np.eye(6) - (K @ self.H)) @ self.P
    
class SkeletonKalmanFilter:
    def __init__(self, joints_count=22, dt=1.0):
        self.joints_count = joints_count
        # Create 22 independent filters, one for each joint
        self.filters = [JointKalmanFilter(dt=dt) for _ in range(joints_count)]

    def update(self, current_pose):
        """
        Updates all 22 joints with new observation data.
        current_pose: (22, 3) numpy array
        """
        for i in range(self.joints_count):
            self.filters[i].predict()          # Step 1: Predict state forward
            self.filters[i].update(current_pose[i]) # Step 2: Correct with measurement

    def coast(self):
        """
        Predicts forward without a measurement (for glitch handling).
        """
        for i in range(self.joints_count):
            self.filters[i].predict() # Step time, but trust internal model

    def predict_future(self, n_frames):
        """
        Predicts N frames into the future without modifying real state.
        Returns: (n_frames, 22, 3)
        """
        predictions = []
        
        # 1. Snapshot current state of all 22 filters
        temp_states = [f.x.copy() for f in self.filters]
        F = self.filters[0].F # Transition matrix is constant
        
        # 2. Simulate forward
        for _ in range(n_frames):
            frame_pose = []
            for i in range(self.joints_count):
                # Apply transition: x = F @ x
                temp_states[i] = F @ temp_states[i]
                frame_pose.append(temp_states[i][:3]) # Keep only (x,y,z)
            predictions.append(frame_pose)
            
        return np.array(predictions)
    
def constrain_prediction(past_frame, predicted_frame):
    ''' 
    Constrain the distance of keypoints to match realistic human body
    Doing it in Batches, so get all predictions first then call this function
    '''   
    reference_body = past_frame #(22,3)
    restrained_predicted_frames = np.copy(predicted_frame) #(Pred_frames, 22, 3)
    parents = np.array([e[0] for e in SKELETON_EDGES_22])
    children = np.array([e[1] for e in SKELETON_EDGES_22])

    ref_vectors = reference_body[children] - reference_body[parents] 
    ref_lengths = np.linalg.norm(ref_vectors, axis=1)

    pred_vectors = predicted_frame[:, children] - predicted_frame[:, parents]
    pred_dists = np.linalg.norm(pred_vectors, axis=2, keepdims=True)
    
    # print("prediction distance",pred_dists)
    # print("reference lengths",ref_lengths)
    
    # Avoid division by zero
    unit_directions = pred_vectors / (pred_dists + 1e-8)
    # print(unit_directions)

    for i, [p1,p2] in enumerate(SKELETON_EDGES_22):
        restrained_predicted_frames[:, p2, :] = restrained_predicted_frames[:, p1, :] + unit_directions[:, i, :] * ref_lengths[i]

    return restrained_predicted_frames 

if __name__ == "__main__":
    ###########################
    #Load ZED data
    ###########################
    source_number = 2
    source_file = f"walk_{source_number}"
    suffix = "KF"
    constrain = False
    # source = f"/home/sosuke/thesis/siMLPe/data/zed_data/{source_file}.json"
    source = f"/home/sosuke/thesis/siMLPe/data/world_data/world_data/{source_file}.json"

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
    VEL_DAMPING = 1.0
    ACC_DAMPING = 0.95

    error_time = 0

    human_tracker = SkeletonKalmanFilter(joints_count=22, dt=1.0/calculated_fps)

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


            if not glitch:
                history_buffer.append(keypoints_array) 
                h36m_32 = zed34_to_h36m32(keypoints_array)
                input_absolute_22 = h36m_32[joint_used_xyz, :]
                human_tracker.update(input_absolute_22)
            else:
                human_tracker.coast() # If glitch, just predict forward without update
     
            if len(history_buffer)> INTERPOLATION_RANGE:
                history_buffer.pop(0)
            
            if len(history_buffer) == INTERPOLATION_RANGE:
                
                # 1. Prepare Data
                past_frames_zed = np.array(history_buffer) # (Range, 34, 3)
                
                # Map ZED(34) -> H36M(32)
                h36m_32 = zed34_to_h36m32(past_frames_zed) # (Range, 32, 3)

                input_absolute_22 = h36m_32[:, joint_used_xyz, :]
                all_inputs_saved.append(input_absolute_22)
                
                root = h36m_32[:, 0:1, :] # Pelvis
                h36m_rooted = h36m_32 - root

                input_centered_22 = h36m_rooted[:, joint_used_xyz, :] # (Range, 22, 3)
                zeroed_input.append(input_centered_22)
                predicted_frames = human_tracker.predict_future(PREDICTION_FRAMES) #(Pred_frames, 22, 3)

                last_frame= input_absolute_22[-1]
                if constrain:
                    predicted_frames = constrain_prediction(last_frame, predicted_frames)

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

    folder_name = f"predictions/{suffix}"
    # Create directory
    os.makedirs(folder_name, exist_ok=True)

    # Join path safely
    save_path = os.path.join(folder_name, f"{source_file}_interp_{suffix}.npz")

    np.savez(save_path, 
            inputs=all_inputs_saved, 
            preds=all_preds_saved,
            zero_input = np.array(zeroed_input),
            zero_output = np.array(zeroed_output)
            )

    print(f"Done! Saved {len(all_preds_saved)} samples to {save_path}")
