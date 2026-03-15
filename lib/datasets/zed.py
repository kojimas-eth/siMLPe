import os
import glob
import json
import numpy as np
import torch
import torch.utils.data as data
import random
from scipy.spatial.transform import Rotation as R

class ZEDDataset(data.Dataset):
    def __init__(self, config, split_name, data_aug=False):
        super(ZEDDataset, self).__init__()
        self.split_name = split_name
        self.data_aug = data_aug
        
        # Path to your ZED data (JSON or NPZ files)
        self.data_dir = config.zed_data_dir
        
        # Define parameters
        self.input_len = config.motion.h36m_input_length
        self.target_len = config.motion.h36m_target_length_train
        self.shift_step = config.shift_step

        # Load data
        self.zed_seqs = self._load_zed_data()
        self._collect_chunks()

    def _load_zed_data(self):
        """
        Loads ZED JSON/NPZ files and converts them to (T, 22, 3) sequences.
        Assumes ZED data is in METERS
        """
        all_files= []

        if isinstance (self.data_dir,str):
            source_dir = [self.data_dir]
        else:
            source_dir = self.data_dir
        
        for folder in source_dir:
            file_list = glob.glob(os.path.join(folder, '*.json'))                  
            all_files.extend(file_list)

        all_files = sorted(all_files)
        rng = random.Random(1)
        rng.shuffle(all_files)

        split_idx = int(len(all_files) * 0.9)
        print(f"Found {len(all_files)} total files across {len(source_dir)} folders.")
       
        if self.split_name == "train":
            files_to_load = all_files[:split_idx]
        else:
            files_to_load = all_files[split_idx:]

        print(f"Loading {self.split_name} data: {len(files_to_load)} files found.")

        all_sequences = []
        joint_used_xyz = np.array([2,3,4,5,7,8,9,10,12,13,14,15,17,18,19,21,22,25,26,27,29,30]).astype(np.int64)


        for file_path in files_to_load:
            with open(file_path, 'r') as f:
                data = json.load(f)
                sorted_timestamps = sorted(data.keys(), key=lambda x: int(x))
    
                zed_keypoints = []
                zed_quats = []
                
                for ts in sorted_timestamps[:-30]:
                    frame_data = data[ts]
                    #skip no body frames
                    if not frame_data.get('body_list'):
                        if zed_keypoints:
                            zed_keypoints.append(zed_keypoints[-1]) # Repeat last keypoint if no body detected
                        continue

                    target_body = frame_data['body_list'][0]
             
                    raw_keypoint = target_body.get('keypoint')
                    np_keypoint = np.array(raw_keypoint)
                    zed_keypoints.append(np_keypoint)
                    quat = target_body.get('global_root_orientation', [0.0, 0.0, 0.0, 1.0])
                    zed_quats.append(quat)
            

                zed_keypoints_np = np.array(zed_keypoints)  # (Total frames, 34, 3)
                zed_quats_np = np.array(zed_quats)

                h36m_32 = self.zed34_to_h36m32(zed_keypoints_np) # (Total frames, 32, 3)
                root = h36m_32[:, 0:1, :] # Pelvis
                h36m_rooted = h36m_32 - root

                #Remove global rotation
                rotations = R.from_quat(zed_quats_np)
                
                #H36M Treats Facing forward as +Y whilst zed is -Y!!!
                correction_rot = R.from_euler('y', 180, degrees=True)
                corrected_inv_rotations = correction_rot * rotations.inv()
                inv_rot_matrices = corrected_inv_rotations.as_matrix()
                h36m_derotated = np.einsum('tij,tkj->tki', inv_rot_matrices, h36m_rooted)

                input_centered_22 = h36m_derotated[:, joint_used_xyz, :]

            
            all_sequences.append(input_centered_22)

        final_dataset= np.concatenate(all_sequences, axis=0)
        print(f"Combined zed final dataset shape is {final_dataset.shape}")
        return [final_dataset]

    def _collect_chunks(self):
        self.data_idx = []
        idx = 0
        for seq in self.zed_seqs:
            N = len(seq)
            if N < self.input_len + self.target_len:
                continue
            
            # Sliding window to create samples
            valid_frames = np.arange(0, N - self.input_len - self.target_len + 1, self.shift_step)
            self.data_idx.extend(zip([idx] * len(valid_frames), valid_frames.tolist()))
            idx += 1

    def __len__(self):
        return len(self.data_idx)
    
    def zed34_to_h36m32(self,zed_data):
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

    def __getitem__(self, index):
        seq_idx, start_frame = self.data_idx[index]
        end_frame = start_frame + self.input_len + self.target_len
        
        motion = self.zed_seqs[seq_idx][start_frame:end_frame]
        
        # Data Augmentation (Time Reversal)
        if self.data_aug and self.split_name == 'train':
            if torch.rand(1)[0] > 0.5:
                motion = np.flip(motion, axis=0).copy()

        # Convert to Tensor
        motion = torch.from_numpy(motion).float()

        # Normalize: Input to model should be in METERS
        h36m_motion_input = motion[:self.input_len]
        h36m_motion_target = motion[self.input_len:] 

        # Reshape to (T, 66) 
        h36m_motion_input = h36m_motion_input.reshape(self.input_len, -1)
        h36m_motion_target = h36m_motion_target.reshape(self.target_len, -1)

        return h36m_motion_input, h36m_motion_target