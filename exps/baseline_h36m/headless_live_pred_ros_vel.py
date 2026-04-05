import os, sys
from pathlib import Path

current_file_path = Path(__file__).resolve()
exps_dir = current_file_path.parent.parent

#Append file paths
sys.path.append(str(exps_dir))
lib_path = exps_dir.parent / 'lib'
if str(lib_path) not in sys.path:
    sys.path.append(str(lib_path))

import pyzed.sl as sl
import numpy as np
import argparse
from scipy.spatial.transform import Rotation as R
import torch
import json
import time 
from zed_finetune.vel_config  import config
from zed_finetune.velocity_model import siMLPe as Model

# --- ROS INTEGRATION ---
import rospy
import std_msgs.msg
from geometry_msgs.msg import Point
from skeleton_msgs.msg import SkeletonPrediction, SkeletonFrame

class LiveSkeletonPublisher:
    def __init__(self, topic_name='/human_skeleton/live_prediction'):
        try:
            rospy.init_node('simlpe_live_publisher', anonymous=True)
            print("✅ Initialized siMLPe ROS Publisher Node (Headless NN-Vel Mode).")
        except rospy.ROSException:
            pass
        self.pub = rospy.Publisher(topic_name, SkeletonPrediction, queue_size=1)
        
    def numpy_to_skeleton_frames(self, sequence_np):
        ros_frames = []
        for frame in sequence_np:
            frame_msg = SkeletonFrame()
            for joint in frame:
                p = Point(x=float(joint[0]), y=float(joint[1]), z=float(joint[2]))
                frame_msg.joints.append(p)
            ros_frames.append(frame_msg)
        return ros_frames

    def publish(self, input_sequence, predicted_sequence):
        if rospy.is_shutdown(): return
        msg = SkeletonPrediction()
        msg.header = std_msgs.msg.Header()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "world"
        msg.input_sequence = self.numpy_to_skeleton_frames(input_sequence)
        msg.predicted_sequence = self.numpy_to_skeleton_frames(predicted_sequence)
        self.pub.publish(msg)

def parse_args(init, opt):
    if len(opt.input_svo_file)>0 and opt.input_svo_file.endswith((".svo", ".svo2")):
        init.set_from_svo_file(opt.input_svo_file)
        print("[Sample] Using SVO File input: {0}".format(opt.input_svo_file))
    elif len(opt.ip_address)>0 :
        ip_str = opt.ip_address
        if ip_str.replace(':','').replace('.','').isdigit() and len(ip_str.split('.'))==4 and len(ip_str.split(':'))==2:
            init.set_from_stream(ip_str.split(':')[0],int(ip_str.split(':')[1]))
        elif ip_str.replace(':','').replace('.','').isdigit() and len(ip_str.split('.'))==4:
            init.set_from_stream(ip_str)
        else :
            print("Unvalid IP format. Using live stream")
    if ("HD2K" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD2K
    elif ("HD1200" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD1200
    elif ("HD1080" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD1080
    elif ("HD720" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD720
    elif ("SVGA" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.SVGA
    elif ("VGA" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.VGA

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
    frames = zed_data.shape[0]
    h36m_data = np.zeros((frames, 32, 3))
    h36m_data[:, 0, :] = zed_data[:, 0, :]   
    h36m_data[:, 1, :] = zed_data[:, 22 ,:]  
    h36m_data[:, 2, :] = zed_data[:, 23, :]  
    h36m_data[:, 3, :] = zed_data[:, 24, :]  
    h36m_data[:, 4, :] = zed_data[:, 25, :]  
    r_ankle = zed_data[:, 24, :]
    r_heel  = zed_data[:, 33, :]
    r_vec = r_ankle - r_heel
    r_vec[:, 1] = 0 
    r_norm = np.linalg.norm(r_vec, axis=1, keepdims=True) + 1e-6
    r_dir_flat = r_vec / r_norm
    r_toe = r_ankle + (r_dir_flat * 0.15)
    r_toe[:, 1] = r_heel[:, 1] 
    h36m_data[:, 5, :] = r_toe
    h36m_data[:, 6, :] = zed_data[:, 18, :]  
    h36m_data[:, 7, :] = zed_data[:, 19, :]  
    h36m_data[:, 8, :] = zed_data[:, 20, :]  
    h36m_data[:, 9, :] = zed_data[:, 21, :]  
    l_ankle = zed_data[:, 20, :]
    l_heel  = zed_data[:, 32, :]
    l_vec = l_ankle - l_heel
    l_vec[:, 1] = 0 
    l_norm = np.linalg.norm(l_vec, axis=1, keepdims=True) + 1e-6
    l_dir_flat = l_vec / l_norm
    l_toe = l_ankle + (l_dir_flat * 0.15)
    l_toe[:, 1] = l_heel[:, 1] 
    h36m_data[:, 10, :] = l_toe
    h36m_data[:, 11, :] = zed_data[:, 1, :]  
    h36m_data[:, 12, :] = zed_data[:, 1, :]  
    h36m_data[:, 13, :] = zed_data[:, 3, :]  
    h36m_data[:, 14, :] = zed_data[:, 27, :] 
    h36m_data[:, 15, :] = (zed_data[:, 30, :] + zed_data[:,28,:])/2 
    eye_midpoint = (zed_data[:, 30, :] + zed_data[:, 28, :]) / 2
    neck_pos = zed_data[:, 3, :] 
    vec_neck_to_eyes = eye_midpoint - neck_pos
    head_up_norm = np.linalg.norm(vec_neck_to_eyes, axis=1, keepdims=True) + 1e-6
    head_up_dir = vec_neck_to_eyes / head_up_norm
    FOREHEAD_HEIGHT = 0.12 
    h36m_data[:, 15, :] = eye_midpoint + (head_up_dir * FOREHEAD_HEIGHT)
    h36m_data[:, 16, :] = zed_data[:, 4, :]  
    h36m_data[:, 17, :] = zed_data[:, 5, :]  
    h36m_data[:, 18, :] = zed_data[:, 6, :]  
    h36m_data[:, 19, :] = zed_data[:, 7, :]  
    h36m_data[:, 20, :] = zed_data[:, 10, :]
    h36m_data[:, 21, :] = zed_data[:, 8, :]  
    h36m_data[:, 22, :] = zed_data[:, 9, :]
    h36m_data[:, 23, :] = zed_data[:, 9, :]
    h36m_data[:, 24, :] = zed_data[:, 11, :]  
    h36m_data[:, 25, :] = zed_data[:, 12, :] 
    h36m_data[:, 26, :] = zed_data[:, 13, :] 
    h36m_data[:, 27, :] = zed_data[:, 14, :] 
    h36m_data[:, 28, :] = zed_data[:, 17, :] 
    h36m_data[:, 29, :] = zed_data[:, 15, :] 
    h36m_data[:, 30, :] = zed_data[:, 16, :] 
    h36m_data[:, 31, :] = zed_data[:, 16, :]
    return h36m_data

def constrain_prediction(last_input_pose, pred_sequence):
    """ Placeholder for constrain function """
    return pred_sequence

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model-pth', type=str, default="/home/nvidia/Downloads/siMLPe/checkpoints/velocity_model_44000.pth", help='=encoder path')
args, unknown = parser.parse_known_args()

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
dct_m_np, idct_m_np = get_dct_matrix(config.motion.h36m_input_length_dct) 
dct_m = torch.tensor(dct_m_np).float().cuda().unsqueeze(0)
idct_m = torch.tensor(idct_m_np).float().cuda().unsqueeze(0)

def main(opt):
    print("Running Body Tracking (Headless NN-Vel Mode) ... Press Ctrl+C to quit.")

    ros_publisher = LiveSkeletonPublisher()

    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.coordinate_units = sl.UNIT.METER          
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.camera_fps = 30
    parse_args(init_params, opt)

    err = zed.open(init_params)
    if err > sl.ERROR_CODE.SUCCESS:
        print(f"Error opening camera: {err}")
        exit(1)

    positional_tracking_parameters = sl.PositionalTrackingParameters()
    positional_tracking_parameters.set_as_static = True
    zed.enable_positional_tracking(positional_tracking_parameters)
    
    body_param = sl.BodyTrackingParameters()
    body_param.enable_tracking = True                
    body_param.enable_body_fitting = True            
    body_param.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST
    body_param.body_format = sl.BODY_FORMAT.BODY_34  
    
    zed.enable_body_tracking(body_param)

    body_runtime_param = sl.BodyTrackingRuntimeParameters()
    body_runtime_param.detection_confidence_threshold = 50

    bodies = sl.Bodies()

    history_buffer = []
    heading_list = []

    # --- Hyperparameters ---
    extreme = False
    SCALE_FACTOR = 1.5
    constrain = False
    VEL_SCALE = 1.0
    YAW_SCALE = 1.0
    LIMB_SCALE = 1.0
    dt = 1.0 / 25.0

    try:
        while not rospy.is_shutdown():
            if zed.grab() == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_bodies(bodies, body_runtime_param)

                for body in bodies.body_list:
                    skeleton_3d = body.keypoint 
                    keypoints_array = np.array(skeleton_3d)  
                    heading_quat = body.global_root_orientation
                    heading_quat_array = np.array(heading_quat) 
                    
                    heading_list.append(heading_quat_array)
                    history_buffer.append(keypoints_array)
                            
                    if len(history_buffer) > 50:
                        history_buffer.pop(0)
                        heading_list.pop(0)
                    
                    if len(history_buffer) == 50:
                        past_frames_zed = np.array(history_buffer) 
                        
                        if extreme:
                            root_idx = 0 
                            start_xy = past_frames_zed[0:1, root_idx, :2]
                            current_xy = past_frames_zed[:, root_idx, :2]
                            movement_delta = current_xy - start_xy
                            
                            extra_travel = movement_delta * (SCALE_FACTOR - 1.0)
                            past_frames_zed[:, :, :2] += extra_travel[:, np.newaxis, :]

                        h36m_32 = zed34_to_h36m32(past_frames_zed) 
                        input_absolute_22 = h36m_32[:, joint_used_xyz, :]

                        root = h36m_32[:, 0:1, :] 
                        h36m_rooted = h36m_32 - root
                        
                        zed_quats_np = np.array(heading_list) 
                        rotations = R.from_quat(zed_quats_np)

                        correction_rot = R.from_euler('y', 180, degrees=True)
                        corrected_inv_rotations = correction_rot * rotations.inv()

                        inv_rot_matrices = corrected_inv_rotations.as_matrix()
                        h36m_derotated = np.einsum('tij,tkj->tki', inv_rot_matrices, h36m_rooted)

                        input_centered_22 = h36m_derotated[:, joint_used_xyz, :]

                        TOTAL_HORIZON = 30   
                        PRED_STEP = 10       
                        num_iterations = TOTAL_HORIZON // PRED_STEP
                        
                        current_global_pos = past_frames_zed[-1, 0, :].copy() 
                        
                        last_global_quat = zed_quats_np[-1]
                        initial_rot = R.from_quat(last_global_quat)
                        
                        initial_euler = initial_rot.as_euler('yxz', degrees=False)
                        current_yaw = initial_euler[0]
                        fixed_pitch = initial_euler[1]
                        fixed_roll = initial_euler[2]
                        
                        future_rotations = [None] * TOTAL_HORIZON 
                        future_global_pos = np.zeros((TOTAL_HORIZON, 3))

                        input_tensor = torch.tensor(input_centered_22).float().cuda()
                        input_tensor = input_tensor.reshape(1, 50, -1) 

                        full_prediction_abs = []
                        correction_inv = correction_rot.inv()

                        with torch.no_grad():
                            for i in range(num_iterations):
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

                                input_tensor = torch.cat([input_tensor[:, PRED_STEP:, :], pred_std], dim=1)

                                pred_vel_std = torch.matmul(idct_m[:, :config.motion.h36m_input_length, :], pred_vel_dct)
                                pred_vel_std = pred_vel_std[:, :PRED_STEP, :]
                                vel_numpy = pred_vel_std[0].cpu().numpy()
                                
                                pred_yaw_std = torch.matmul(idct_m[:, :config.motion.h36m_input_length, :], pred_yaw_dct)
                                pred_yaw_std = pred_yaw_std[:, :PRED_STEP, :] 
                                yaw_numpy = pred_yaw_std[0].cpu().numpy()

                                for t in range(PRED_STEP):
                                    abs_t = (i * PRED_STEP) + t
                                    
                                    real_vel_x = -vel_numpy[t, 0] * VEL_SCALE
                                    real_vel_z = -vel_numpy[t, 1] * VEL_SCALE
                                    
                                    current_global_pos[0] += real_vel_x * dt
                                    current_global_pos[2] += real_vel_z * dt
                                    future_global_pos[abs_t] = current_global_pos.copy()

                                    boosted_yaw_rate = yaw_numpy[t, 0] * YAW_SCALE
                                    current_yaw += boosted_yaw_rate * dt
                                    
                                    new_euler = [current_yaw, fixed_pitch, fixed_roll]
                                    new_rot = R.from_euler('yxz', new_euler, degrees=False)
                                    future_rotations[abs_t] = new_rot

                                pred_numpy_centered = pred_std.cpu().numpy().reshape(PRED_STEP, 22, 3)
                                
                                if constrain:
                                    pred_numpy_centered = constrain_prediction(input_absolute_22[-1], pred_numpy_centered)
                                
                                final_global_poses = np.zeros_like(pred_numpy_centered)

                                for t in range(PRED_STEP):
                                    abs_t = (i * PRED_STEP) + t
                                    final_rot = future_rotations[abs_t] * correction_inv
                                    rot_matrix = final_rot.as_matrix()
                                    rotated_pose = (rot_matrix @ pred_numpy_centered[t].T).T 
                                    final_global_poses[t] = rotated_pose + future_global_pos[abs_t]
                                    
                                full_prediction_abs.append(final_global_poses)

                        final_pred_abs = np.concatenate(full_prediction_abs, axis=0)

                        # --- ROS INTEGRATION: Broadcast the live data! ---
                        ros_publisher.publish(input_absolute_22, final_pred_abs)

    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Cleaning up...")

    # Cleanup
    zed.disable_body_tracking()
    zed.disable_positional_tracking()
    zed.close()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_svo_file', type=str, help='Path to an .svo file',default = '')
    parser.add_argument('--ip_address', type=str, help='IP Adress', default = '')
    parser.add_argument('--resolution', type=str, help='Resolution', default = '')
    opt, unknown = parser.parse_known_args()
    if len(opt.input_svo_file)>0 and len(opt.ip_address)>0:
        print("Specify only input_svo_file or ip_address, or none to use wired camera, not both. Exit program")
        exit()
    main(opt)