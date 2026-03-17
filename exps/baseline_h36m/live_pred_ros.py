import os, sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
lib_path = Path(__file__).resolve().parent.parent.parent / 'lib'
if str(lib_path) not in sys.path:
    sys.path.append(str(lib_path))

import cv2
import pyzed.sl as sl
import ogl_viewer.viewer as gl
import cv_viewer.tracking_viewer as cv_viewer
import numpy as np
import argparse
from scipy.spatial.transform import Rotation as R
from utils.kalman_filter import GlobalTrajectoryExtrapolator
from utils.extended_KF import MEKFTrajectoryExtrapolator
import torch
import json
import time 

from config import config
from model import siMLPe as Model
from datasets.h36m_eval import H36MEval

# --- ROS INTEGRATION: Imports ---
import rospy
import std_msgs.msg
from geometry_msgs.msg import Point
# Import from your new dedicated package!
from skeleton_msgs.msg import SkeletonPrediction, SkeletonFrame

# --- ROS INTEGRATION: Publisher Class ---
class LiveSkeletonPublisher:
    def __init__(self, topic_name='/human_skeleton/live_prediction'):
        # Initialize ROS Node
        try:
            rospy.init_node('simlpe_live_publisher', anonymous=True)
            print("✅ Initialized siMLPe ROS Publisher Node.")
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
# ----------------------------------------

def parse_args(init, opt):
    if len(opt.input_svo_file)>0 and opt.input_svo_file.endswith((".svo", ".svo2")):
        init.set_from_svo_file(opt.input_svo_file)
        print("[Sample] Using SVO File input: {0}".format(opt.input_svo_file))
    elif len(opt.ip_address)>0 :
        ip_str = opt.ip_address
        if ip_str.replace(':','').replace('.','').isdigit() and len(ip_str.split('.'))==4 and len(ip_str.split(':'))==2:
            init.set_from_stream(ip_str.split(':')[0],int(ip_str.split(':')[1]))
            print("[Sample] Using Stream input, IP : ",ip_str)
        elif ip_str.replace(':','').replace('.','').isdigit() and len(ip_str.split('.'))==4:
            init.set_from_stream(ip_str)
            print("[Sample] Using Stream input, IP : ",ip_str)
        else :
            print("Unvalid IP format. Using live stream")
    if ("HD2K" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD2K
        print("[Sample] Using Camera in resolution HD2K")
    elif ("HD1200" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD1200
        print("[Sample] Using Camera in resolution HD1200")
    elif ("HD1080" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD1080
        print("[Sample] Using Camera in resolution HD1080")
    elif ("HD720" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD720
        print("[Sample] Using Camera in resolution HD720")
    elif ("SVGA" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.SVGA
        print("[Sample] Using Camera in resolution SVGA")
    elif ("VGA" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.VGA
        print("[Sample] Using Camera in resolution VGA")
    elif len(opt.resolution)>0: 
        print("[Sample] No valid resolution entered. Using default")
    else : 
        print("[Sample] Using default resolution")

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

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model-pth', type=str, default="/home/nvidia/Downloads/siMLPe/exps/zed_finetune/log/snapshot/fixed_world_zed_finetuned_40000.pth", help='=encoder path')
# Allow ROS to pass its own arguments without crashing argparse
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

def draw_prediction_on_image(image, pred_22, root_pos, camera_info,image_scale):
    calib = camera_info.camera_configuration.calibration_parameters.left_cam
    sx, sy = image_scale 
    fx = calib.fx * sx
    fy = calib.fy * sy
    cx = calib.cx * sx
    cy = calib.cy * sy
    
    def project(point_3d):
        x, y, z = point_3d
        x=-x 
        u = int((x * fx / z) + cx)
        v = int((y * fy / z) + cy)
        return (u, v)

    joints_2d = {}
    root_2d = project(root_pos)
    for i, point_3d in enumerate(pred_22):
        joints_2d[i] = project(point_3d)

    color = (0, 255, 0) 
    thickness = 2
    if root_2d: cv2.circle(image, root_2d, 4, (0, 0, 255), -1) 
    for k, uv in joints_2d.items():
        if uv: cv2.circle(image, uv, 3, color, -1)

    def draw_line(p1_idx, p2_idx):
        if p1_idx in joints_2d and p2_idx in joints_2d:
            pt1 = joints_2d[p1_idx]
            pt2 = joints_2d[p2_idx]
            if pt1 and pt2:
                cv2.line(image, pt1, pt2, color, thickness)
    
    def draw_line_to_root(p_idx):
        if root_2d and p_idx in joints_2d:
            pt = joints_2d[p_idx]
            if pt:
                cv2.line(image, root_2d, pt, color, 1)

    draw_line_to_root(0) 
    draw_line(0, 1)      
    draw_line(1, 2)      
    draw_line_to_root(4) 
    draw_line(4, 5)      
    draw_line(5, 6)      
    draw_line_to_root(8) 
    draw_line(8, 9)      
    draw_line(9, 10)     
    draw_line(9, 12)     
    draw_line(12, 13)    
    draw_line(13, 14)    
    draw_line(14, 15)    
    draw_line(9, 17)     
    draw_line(17, 18)    
    draw_line(18, 19)    
    draw_line(19, 20)    

def map_h36m_22_to_zed_34_sparse(pred_22, root_pos):
    zed_34 = np.full((34, 3), np.nan)
    zed_34[0] = root_pos
    zed_34[23] = pred_22[0] 
    zed_34[24] = pred_22[1] 
    zed_34[25] = pred_22[2] 
    zed_34[19] = pred_22[4] 
    zed_34[20] = pred_22[5] 
    zed_34[21] = pred_22[6] 
    zed_34[1]  = pred_22[8]  
    zed_34[3]  = pred_22[9]  
    zed_34[27] = pred_22[10] 
    zed_34[5]  = pred_22[12] 
    zed_34[6]  = pred_22[13] 
    zed_34[7]  = pred_22[14] 
    zed_34[8]  = pred_22[15] 
    zed_34[12] = pred_22[17] 
    zed_34[13] = pred_22[18] 
    zed_34[14] = pred_22[19] 
    zed_34[15] = pred_22[20] 
    return zed_34

def main(opt):
    print("Running Body Tracking sample ... Press 'q' to quit, or 'm' to pause or restart")

    # --- ROS INTEGRATION: Instantiate the Publisher ---
    ros_publisher = LiveSkeletonPublisher()
    # ------------------------------------------------

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

    camera_info = zed.get_camera_information()
    display_resolution = sl.Resolution(min(camera_info.camera_configuration.resolution.width, 1280), min(camera_info.camera_configuration.resolution.height, 720))
    image_scale = [display_resolution.width / camera_info.camera_configuration.resolution.width
                 , display_resolution.height / camera_info.camera_configuration.resolution.height]

    viewer = gl.GLViewer()
    viewer.init(camera_info.camera_configuration.calibration_parameters.left_cam, body_param.enable_tracking,body_param.body_format)
    bodies = sl.Bodies()
    image = sl.Mat()
    key_wait = 10 

    all_inputs_saved = []
    all_preds_saved = []
    zeroed_input=[]
    zeroed_output=[]

    history_buffer = []
    heading_list = []
    prediction_skeleton = None

    while viewer.is_available():
        if zed.grab() <= sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
            zed.retrieve_bodies(bodies, body_runtime_param)

            for body in bodies.body_list:
                skeleton_3d = body.keypoint 
                keypoints_array= np.array(skeleton_3d)  
                heading_quat = body.global_root_orientation
                heading_quat_array = np.array(heading_quat) 
                heading_list.append(heading_quat_array)
                history_buffer.append(keypoints_array)
                        
                if len(history_buffer)> 50:
                    history_buffer.pop(0)
                    heading_list.pop(0)
                
                if len(history_buffer) == 50:
                    past_frames_zed = np.array(history_buffer) 
                    h36m_32 = zed34_to_h36m32(past_frames_zed) 
                    input_absolute_22 = h36m_32[:, joint_used_xyz, :]
                    all_inputs_saved.append(input_absolute_22)

                    root = h36m_32[:, 0:1, :] 
                    h36m_rooted = h36m_32 - root
                    
                    zed_quats_np = np.array(heading_list) 
                    rotations = R.from_quat(zed_quats_np)

                    correction_rot = R.from_euler('y', 180, degrees=True)
                    corrected_inv_rotations = correction_rot * rotations.inv()

                    inv_rot_matrices = corrected_inv_rotations.as_matrix()
                    h36m_derotated = np.einsum('tij,tkj->tki', inv_rot_matrices, h36m_rooted)

                    input_centered_22 = h36m_derotated[:, joint_used_xyz, :]
                    zeroed_input.append(input_centered_22)

                    TOTAL_HORIZON = 60   
                    PRED_STEP = 10       
                    num_iterations = TOTAL_HORIZON // PRED_STEP
                    
                    LIMB_SCALE=1.0
                    ROOT_SCALE=1.0
                    
                    extrapolator = MEKFTrajectoryExtrapolator(dt=1.0/25.0)

                    for i in range(50):
                        global_pos = past_frames_zed[i,0,:] 
                        global_quat = zed_quats_np[i]    
                        extrapolator.update(global_pos, global_quat)
                    future_global_pos, future_global_quats = extrapolator.predict_future(num_frames=TOTAL_HORIZON)
                    future_rotations = R.from_quat(future_global_quats)

                    input_tensor = torch.tensor(input_centered_22).float().cuda()
                    input_tensor = input_tensor.reshape(1, 50, -1) 

                    full_prediction_abs = []
                    full_prediction_centered = []
                    
                    future_rotations = R.from_quat(future_global_quats)
                    correction_inv = correction_rot.inv()

                    with torch.no_grad():
                        for i in range(num_iterations):
                            if config.deriv_input:
                                input_dct = torch.matmul(dct_m[:, :, :config.motion.h36m_input_length], input_tensor)
                            else:
                                input_dct = input_tensor.clone()
                            
                            pred_dct = model(input_dct)
                            pred_std = torch.matmul(idct_m[:, :config.motion.h36m_input_length, :], pred_dct) 
                            pred_std = pred_std[:, :PRED_STEP, :] 

                            if config.deriv_output:
                                pred_std = pred_std * LIMB_SCALE
                                pred_std = pred_std + input_tensor[:, -1:, :].repeat(1, PRED_STEP, 1)

                            input_tensor = torch.cat([input_tensor[:, PRED_STEP:, :], pred_std], dim=1)

                            pred_numpy_centered = pred_std.cpu().numpy().reshape(PRED_STEP, 22, 3)
                            full_prediction_centered.append(pred_numpy_centered)
                            
                            final_global_poses = np.zeros_like(pred_numpy_centered)

                            for t in range(PRED_STEP):
                                abs_t = (i * PRED_STEP) + t
                                final_rot = future_rotations[abs_t] * correction_inv
                                rot_matrix = final_rot.as_matrix()
                                rotated_pose = (rot_matrix @ pred_numpy_centered[t].T).T 
                                final_global_poses[t] = rotated_pose + future_global_pos[abs_t]
                                
                            full_prediction_abs.append(final_global_poses)

                    final_pred_abs = np.concatenate(full_prediction_abs, axis=0)
                    final_pred_centered = np.concatenate(full_prediction_centered, axis=0)
                    
                    all_preds_saved.append(final_pred_abs)
                    zeroed_output.append(final_pred_centered)

                    # --- ROS INTEGRATION: Broadcast the live data! ---
                    # input_absolute_22 shape is (50, 22, 3)
                    # final_pred_abs shape is (60, 22, 3)
                    ros_publisher.publish(input_absolute_22, final_pred_abs)
                    # -------------------------------------------------

                    if len(final_pred_abs) > 0:
                        last_pred_pose = final_pred_abs[-1] 
                        last_root = future_global_pos[-1]
                        
                        draw_prediction_on_image(image_left_ocv, last_pred_pose, last_root, camera_info, image_scale)
                    else:
                        prediction_skeleton = None

            image_left_ocv = image.get_data()
            cv_viewer.render_2D(image_left_ocv,image_scale, bodies.body_list, body_param.enable_tracking, body_param.body_format)
            cv2.imshow("ZED | 2D View", image_left_ocv)
            
            key = cv2.waitKey(key_wait)
            if key == 113: 
                print("Exiting...")
                break
            if key == 109: 
                if (key_wait>0):
                    print("Pause")
                    key_wait = 0 
                else : 
                    print("Restart")
                    key_wait = 10 
    viewer.exit()
    image.free(sl.MEM.CPU)
    zed.disable_body_tracking()
    zed.disable_positional_tracking()
    zed.close()
    cv2.destroyAllWindows()
    
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