import argparse
import os, sys
from scipy.spatial.transform import Rotation as R
import torch
import json
import time 

import numpy as np
from config  import config
from model import siMLPe as Model

joint_used_xyz = np.array([2,3,4,5,7,8,9,10,12,13,14,15,17,18,19,21,22,25,26,27,29,30]).astype(np.int64)

def load_zed_data(file_path):
    import numpy as np
    data = np.load(file_path)
    return data['inputs'], data['targets']


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
    h36m_data[:, 4, :] = zed_data[:, 25, :]  # R sites
   
    #approx the toe location
    r_heel = zed_data[:, 25, :] - zed_data[:, 33, :] 
    r_vec_ankle_toe_dir = -r_heel
    r_norm = np.linalg.norm(r_vec_ankle_toe_dir) + 1e-6
    r_unit_vec = r_vec_ankle_toe_dir / r_norm

    h36m_data[:, 5, :] = zed_data[:, 25, :] + r_unit_vec *1.5  # R toes

    
    # --- LEFT LEG ---
    h36m_data[:, 6, :] = zed_data[:, 18, :]  # L Hip
    h36m_data[:, 7, :] = zed_data[:, 19, :]  # L Knee
    h36m_data[:, 8, :] = zed_data[:, 20, :]  # L Ankle
    h36m_data[:, 9, :] = zed_data[:, 21, :]  # L sites
    
    #approx the toe location
    l_heel = zed_data[:, 21, :] - zed_data[:, 32, :] 
    l_vec_ankle_toe_dir = -l_heel
    l_norm = np.linalg.norm(l_vec_ankle_toe_dir) + 1e-6
    l_unit_vec = l_vec_ankle_toe_dir / l_norm
    h36m_data[:, 10, :] = zed_data[:, 21, :] + l_unit_vec *1.5 # L toes

    
    # --- SPINE & HEAD ---
    h36m_data[:, 11, :] = zed_data[:, 1, :]  # Spine Navel -> Spine Low
    h36m_data[:, 12, :] = zed_data[:, 2, :]  # Spine Chest -> Spine Torso
    h36m_data[:, 13, :] = zed_data[:, 3, :]  # Neck -> Neck
    h36m_data[:, 14, :] = zed_data[:, 26, :] # Head -> Head (Nose/Jaw)
    h36m_data[:, 15, :] = zed_data[:, 27, :] # Nose -> Site (Head Top approximate)

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
    h36m_data[:, 30, :] = zed_data[:, 16, :] # R Wrist
    h36m_data[:, 31, :] = zed_data[:, 16, :]
    return h36m_data

############################
#Load Model
###########################
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model-pth', type=str, default="/home/sosuke/thesis/siMLPe/checkpoints/h36m_model_35000.pth", help='=encoder path')
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

############################
#Load ZED data
###########################
source = "/home/sosuke/thesis/siMLPe/data/zed_data/30fps_body34_fast_3.json"
if source.endswith('.json') or source.endswith('.jsonl'):
    with open(source, 'r') as file:
        data = json.load(file)

all_inputs_saved = []
all_preds_saved = []
zeroed_input=[]
zeroed_outpout=[]

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
        
        if len(history_buffer) == 50:
            past_frames = np.array(history_buffer) #(50,34,3)
            h36m_order_converted = zed34_to_h36m32(past_frames)  #(50,32,3)
            
            root = h36m_order_converted[:, 0:1, :]
            h36m_rooted = h36m_order_converted - root
        
            input_22 = h36m_rooted[:, joint_used_xyz, :] # Shape (50, 22, 3)
        
            # 5. Save Input for Evaluation (Before reshaping)
            all_inputs_saved.append(h36m_order_converted[:, joint_used_xyz, :])
            zeroed_input.append(input_22)
            
            # step = config.motion.h36m_target_length_train #In our case 10
            # outputs = []
            # if step == 25:
            #     num_step = 1
            # else:
            #     num_step = 25 // step + 1
            
            # for idx in range(num_step):
            #     with torch.no_grad():
            #         input_tensor = torch.tensor(input_22).float().cuda()
            #         input_tensor = input_tensor.reshape(1, 50, -1) #(1, 50, 66)
                    
            #         if config.deriv_input:
            #             motion_input = input_tensor.clone()
            #             motion_input = torch.matmul(dct_m[:, :, :config.motion.h36m_input_length], input_tensor)
            #         else:
            #             motion_input = input_tensor.clone()
                    
            #         #Predict in DCT space
            #         output = model(motion_input)

            #         #Apply the inverse DCT
            #         output = torch.matmul(idct_m[:, :config.motion.h36m_input_length, :], output)[:, :step, :]
            #         if config.deriv_output:
            #             output = output + motion_input[:, -1:, :].repeat(1,step,1)

            #     output = output.reshape(-1, 22*3)
            #     output = output.reshape(1,step,-1)
            #     outputs.append(output)
            #     motion_input = torch.cat([motion_input[:, step:], output], axis=1)
            # motion_pred= torch.cat(outputs, axis=1)[:,:25]
            # motion_pred = motion_pred.cpu().numpy().reshape(25, 22, 3)
            # # print("shape of motion_pred is " ,motion_pred.shape)

            # # 1. Calculate the Linear Velocity of the root from the history by taking average
            # last_root_pos = root[-1, 0, :] # XYZ of root at frame 50
            # prev_root_pos = root[-20, 0, :] # XYZ of root at frame 30
            # velocity = (last_root_pos - prev_root_pos) / 20.0 

            # # 2. Project this velocity into the future (25 frames)
            # future_roots = []
            # for i in range(1, 26):
            #     new_root = last_root_pos + (velocity * i)
            #     future_roots.append(new_root)

            # future_roots = np.array(future_roots) # Shape (25, 3)
            # future_roots = future_roots[:, np.newaxis, :] # Shape (25, 1, 3)

            # # 3. Add the global root back to the prediction
            # # pred_numpy is (25, 22, 3). We broadcast add (25, 1, 3)
            # pred_numpy = motion_pred + future_roots

            # # Store Prediction
            # all_preds_saved.append(pred_numpy)
            

            # # time.sleep(2)
            # if len(all_preds_saved) % 50 == 0:
            #     print(f"Predicted {len(all_preds_saved)} windows so far...")




            #--- INFERENCE  ---
            with torch.no_grad():
                # Prepare Tensor: (Batch=1, Frames=50, Features=66)
                input_tensor = torch.tensor(input_22).float().cuda()
                input_tensor = input_tensor.reshape(1, 50, -1) #(1, 50, 66)

                # 1. Apply DCT (Coordinate Space -> Frequency Space)
                if config.deriv_input:
                    input_dct = torch.matmul(dct_m[:, :, :config.motion.h36m_input_length], input_tensor)
                else:
                    input_dct = input_tensor.clone()
                # 2. Model Prediction
                pred_dct = model(input_dct)

                # 3. Apply Inverse DCT (Frequency Space -> Coordinate Space)
                pred_std = torch.matmul(idct_m[:, :config.motion.h36m_input_length, :], pred_dct) # Shape: (1, 50, 66)
                pred_std = pred_std[:, :25, :]

                # 5. Add Residual (If model predicts offset from last frame)
                if config.deriv_output:
                    pred_std = pred_std + input_tensor[:, -1:, :].repeat(1, 25, 1)

                # --- C. POST-PROCESSING ---
                # Reshape back to (25, 22, 3)
                pred_numpy = pred_std.cpu().numpy().reshape(25, 22, 3)
                zeroed_outpout.append(pred_numpy)
                # --- D. RESTORE GLOBAL ROOT ---

                # 1. Calculate the Linear Velocity of the root from the history by taking average
                last_root_pos = root[-1, 0, :] # XYZ of root at frame 50
                prev_root_pos = root[-20, 0, :] # XYZ of root at frame 30
                velocity = (last_root_pos - prev_root_pos) / 20.0 

                # 2. Project this velocity into the future (25 frames)
                future_roots = []
                for i in range(1, 26):
                    new_root = last_root_pos + (velocity * i)
                    future_roots.append(new_root)

                future_roots = np.array(future_roots) # Shape (25, 3)
                future_roots = future_roots[:, np.newaxis, :] # Shape (25, 1, 3)

                # 3. Add the global root back to the prediction
                # pred_numpy is (25, 22, 3). We broadcast add (25, 1, 3)
                pred_numpy = pred_numpy + future_roots

                # Store Prediction
                all_preds_saved.append(pred_numpy)
                

                # time.sleep(2)
                if len(all_preds_saved) % 50 == 0:
                    print(f"Predicted {len(all_preds_saved)} windows so far...")



# ---------------------------------------------------------
# 5. SAVE RESULTS
# ---------------------------------------------------------
print("Saving results to disk...")
all_inputs_saved = np.array(all_inputs_saved) # Shape (N, 50, 22, 3)
all_preds_saved = np.array(all_preds_saved)   # Shape (N, 25, 22, 3)

np.savez("zed_inference_results.npz", 
         inputs=all_inputs_saved, 
         preds=all_preds_saved,
         zero_input = zeroed_input,
         zero_output = zeroed_outpout
         )

print(f"Done! Saved {len(all_preds_saved)} samples to zed_inference_results.npz")



'''
Do the following things:

collect enough inputs (50 in our case) to make a prediction
Remap the joints coming from zed to h36m format
During inference only use the 22 joints 
Apply zeroing of positions to all 50 frames
feed into the model and get prediction

Save the inputs and predictions as .npy files for later evaluation
'''

