import os, sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
lib_path = Path(__file__).resolve().parent.parent.parent / 'lib'
if str(lib_path) not in sys.path:
    sys.path.append(str(lib_path))
# zed_helpers_path = "/usr/local/zed/samples/body tracking/body tracking/python"

# # Verify the path exists to avoid confusing errors later
# if os.path.exists(zed_helpers_path):
#     sys.path.append(zed_helpers_path)
#     print(f"✅ Successfully added ZED helpers from: {zed_helpers_path}")
# else:
#     print(f"❌ Error: Could not find ZED helpers at {zed_helpers_path}")
#     print("   Please verify the path to the 'ogl_viewer' folder.")
#     sys.exit(1)

import cv2
import pyzed.sl as sl
import ogl_viewer.viewer as gl
import cv_viewer.tracking_viewer as cv_viewer
import numpy as np

import argparse
from scipy.spatial.transform import Rotation as R
from  utils.kalman_filter import GlobalTrajectoryExtrapolator
from  utils.extended_KF import MEKFTrajectoryExtrapolator
import torch
import json
import time 

from config  import config
from model import siMLPe as Model

from datasets.h36m_eval import H36MEval

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
# parser.add_argument('--model-pth', type=str, default="/home/nvidia/Downloads/siMLPe/checkpoints/h36m_model_35000.pth", help='=encoder path')
parser.add_argument('--model-pth', type=str, default="/home/nvidia/Downloads/siMLPe/exps/zed_finetune/log/snapshot/fixed_world_zed_finetuned_40000.pth", help='=encoder path')
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

def draw_prediction_on_image(image, pred_22, root_pos, camera_info,image_scale):
    """
    Projects 3D prediction onto the 2D image plane and draws it.
    """
    # Get Camera Intrinsic Parameters
    calib = camera_info.camera_configuration.calibration_parameters.left_cam
    sx, sy = image_scale 
    
    fx = calib.fx * sx
    fy = calib.fy * sy
    cx = calib.cx * sx
    cy = calib.cy * sy
    
    # Helper to project 3D (x,y,z) -> 2D (u,v)
    def project(point_3d):
        x, y, z = point_3d
        x=-x #Flipped
        u = int((x * fx / z) + cx)
        v = int((y * fy / z) + cy)
        return (u, v)

    # 1. Project All Points
    # Create a map of index -> 2D pixel coordinate
    joints_2d = {}
    
    # Project Root
    root_2d = project(root_pos)
    
    # Project the 22 predicted joints
    for i, point_3d in enumerate(pred_22):
        joints_2d[i] = project(point_3d)

    # 2. Define Connections (Bone indices based on your mapping)
    color = (0, 255, 0) 
    thickness = 2

    # Draw Points
    if root_2d: cv2.circle(image, root_2d, 4, (0, 0, 255), -1) # Red Root
    for k, uv in joints_2d.items():
        if uv: cv2.circle(image, uv, 3, color, -1)

    # Draw Lines
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
                cv2.line(image, root_2d, pt, color, 1) # Thinner line to root

    # --- Connectivity Logic ---
    # Legs (Right)
    draw_line_to_root(0) # Root -> R Knee
    draw_line(0, 1)      # R Knee -> R Ankle
    draw_line(1, 2)      # R Ankle -> R Foot

    # Legs (Left)
    draw_line_to_root(4) # Root -> L Knee
    draw_line(4, 5)      # L Knee -> L Ankle
    draw_line(5, 6)      # L Ankle -> L Foot

    # Spine
    draw_line_to_root(8) # Root -> Spine Torso
    draw_line(8, 9)      # Spine -> Neck
    draw_line(9, 10)     # Neck -> Head
    
    # Arms (Left)
    draw_line(9, 12)     # Neck -> L Shoulder (Approx connection)
    draw_line(12, 13)    # L Shoulder -> L Elbow
    draw_line(13, 14)    # L Elbow -> L Wrist
    draw_line(14, 15)    # L Wrist -> L Hand

    # Arms (Right)
    draw_line(9, 17)     # Neck -> R Shoulder (Approx connection)
    draw_line(17, 18)    # R Shoulder -> R Elbow
    draw_line(18, 19)    # R Elbow -> R Wrist
    draw_line(19, 20)    # R Wrist -> R Hand

def map_h36m_22_to_zed_34_sparse(pred_22, root_pos):
    """
    Maps 22-joint H36M predictions to ZED BODY_34.
    Unpredicted joints are set to NaN so they (and their connected bones) 
    are invisible in the viewer.
    """
    # 1. Initialize with NaN to hide everything by default
    zed_34 = np.full((34, 3), np.nan)
    
    zed_34[0] = root_pos

    # --- Legs (Knees, Ankles, Feet) ---
    zed_34[23] = pred_22[0] # R Knee
    zed_34[24] = pred_22[1] # R Ankle
    zed_34[25] = pred_22[2] # R Foot (Toe Base)

    
    zed_34[19] = pred_22[4] # L Knee
    zed_34[20] = pred_22[5] # L Ankle
    zed_34[21] = pred_22[6] # L Foot


    # --- Spine / Head ---
    zed_34[1]  = pred_22[8]  # Spine (Mapped from H36M SpineChest -> ZED 1)
    zed_34[3]  = pred_22[9]  # Neck
    zed_34[27] = pred_22[10] # Head (Nose)
    # pred_22[11] is Head Top - skipping

    # --- Left Arm ---
    zed_34[5]  = pred_22[12] # L Shoulder
    zed_34[6]  = pred_22[13] # L Elbow
    zed_34[7]  = pred_22[14] # L Wrist
    zed_34[8]  = pred_22[15] # L Hand (Palm) (Mapped from H36M 21)
    # pred_22[16] is L Thumb - skipping

    # --- Right Arm ---
    zed_34[12] = pred_22[17] # R Shoulder
    zed_34[13] = pred_22[18] # R Elbow
    zed_34[14] = pred_22[19] # R Wrist
    zed_34[15] = pred_22[20] # R Hand (Palm) (Mapped from H36M 29)
    # pred_22[21] is R Thumb - skipping

    return zed_34

def main(opt):
    print("Running Body Tracking sample ... Press 'q' to quit, or 'm' to pause or restart")

    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD1080 video mode
    init_params.coordinate_units = sl.UNIT.METER          # Set coordinate units
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.camera_fps = 30
    parse_args(init_params, opt)

    # Open the camera
    err = zed.open(init_params)
    if err > sl.ERROR_CODE.SUCCESS:
        exit(1)

    # Enable Positional tracking (mandatory for object detection)
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    # If the camera is static, uncomment the following line to have better performances
    positional_tracking_parameters.set_as_static = True
    zed.enable_positional_tracking(positional_tracking_parameters)
    
    body_param = sl.BodyTrackingParameters()
    body_param.enable_tracking = True                # Track people across images flow
    body_param.enable_body_fitting = True            # Smooth skeleton move
    body_param.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST
    body_param.body_format = sl.BODY_FORMAT.BODY_34  
    

    # Enable Object Detection module
    zed.enable_body_tracking(body_param)

    body_runtime_param = sl.BodyTrackingRuntimeParameters()
    body_runtime_param.detection_confidence_threshold = 50

    # Get ZED camera information
    camera_info = zed.get_camera_information()
    # 2D viewer utilities
    display_resolution = sl.Resolution(min(camera_info.camera_configuration.resolution.width, 1280), min(camera_info.camera_configuration.resolution.height, 720))
    image_scale = [display_resolution.width / camera_info.camera_configuration.resolution.width
                 , display_resolution.height / camera_info.camera_configuration.resolution.height]

    # Create OpenGL viewer
    viewer = gl.GLViewer()
    viewer.init(camera_info.camera_configuration.calibration_parameters.left_cam, body_param.enable_tracking,body_param.body_format)
    # Create ZED objects filled in the main loop
    bodies = sl.Bodies()
    image = sl.Mat()
    key_wait = 10 

    '''coming from pred'''
    all_inputs_saved = []
    all_preds_saved = []
    zeroed_input=[]
    zeroed_output=[]

    history_buffer = []
    heading_list = []
    prediction_skeleton = None


    while viewer.is_available():
        # Grab an image
        if zed.grab() <= sl.ERROR_CODE.SUCCESS:
            # Retrieve left image
            zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
            # Retrieve bodies
            zed.retrieve_bodies(bodies, body_runtime_param)

            for body in bodies.body_list:
                skeleton_3d = body.keypoint 
                keypoints_array= np.array(skeleton_3d)  #skeleton_array.shape = (34,3)
                heading_quat = body.global_root_orientation
                heading_quat_array = np.array(heading_quat) #(4,)
                heading_list.append(heading_quat_array)
                ''' Changing to prediction code here'''
                history_buffer.append(keypoints_array)
                        
                if len(history_buffer)> 50:
                    history_buffer.pop(0)
                    heading_list.pop(0)
                
                if len(history_buffer) == 50:
                    past_frames_zed = np.array(history_buffer) # (50, 34, 3)
                    # Map ZED(34) -> H36M(32)
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
                    
                    #Scale the velocity and limb movement if desired
                    LIMB_SCALE=1.0
                    ROOT_SCALE=1.0
                    
                    # Convert to Tensor (Model expects centered data)
                    extrapolator = MEKFTrajectoryExtrapolator(dt=1.0/25.0)

                    # 2. Feed the filter the past 50 frames (input sequence)
                    for i in range(50):
                        global_pos = past_frames_zed[i,0,:] # Shape (3,)
                        global_quat = zed_quats_np[i]    # Shape (4,)
                        extrapolator.update(global_pos, global_quat)
                    future_global_pos, future_global_quats = extrapolator.predict_future(num_frames=TOTAL_HORIZON)
                    future_rotations = R.from_quat(future_global_quats)

                    input_tensor = torch.tensor(input_centered_22).float().cuda()
                    input_tensor = input_tensor.reshape(1, 50, -1) 

                    full_prediction_abs = []
                    full_prediction_centered = []
                    
                    #Rotations for re-mapping back to zed's world space
                    future_rotations = R.from_quat(future_global_quats)
                    correction_inv = correction_rot.inv()

                    # 3. Auto-Regressive Loop
                    with torch.no_grad():
                        for i in range(num_iterations):
                            # DCT & Model
                            if config.deriv_input:
                                input_dct = torch.matmul(dct_m[:, :, :config.motion.h36m_input_length], input_tensor)
                            else:
                                input_dct = input_tensor.clone()
                            
                            pred_dct = model(input_dct)
                            pred_std = torch.matmul(idct_m[:, :config.motion.h36m_input_length, :], pred_dct) 
                            
                            # Take first 10 chunk of pred
                            pred_std = pred_std[:, :PRED_STEP, :] 

                            # Residual
                            if config.deriv_output:
                                pred_std = pred_std * LIMB_SCALE
                                pred_std = pred_std + input_tensor[:, -1:, :].repeat(1, PRED_STEP, 1)

                            # Update Buffer
                            input_tensor = torch.cat([input_tensor[:, PRED_STEP:, :], pred_std], dim=1)

                            # --- Post-Process ---
                            # 1. Centered Prediction
                            pred_numpy_centered = pred_std.cpu().numpy().reshape(PRED_STEP, 22, 3)
                            

                            full_prediction_centered.append(pred_numpy_centered)
                            
                            final_global_poses = np.zeros_like(pred_numpy_centered)

                            for t in range(PRED_STEP):
                                abs_t = (i * PRED_STEP) + t

                                #Apply Y-axis flip redo
                                final_rot = future_rotations[abs_t] * correction_inv
                                rot_matrix = final_rot.as_matrix()
                                
                                # Get the rotation matrix for this specific future frame
                                # rot_matrix = future_rotations[abs_t].as_matrix() # Shape (3, 3)
                                
                                # Re-apply KF rotation and position to all 22 joints
                                rotated_pose = (rot_matrix @ pred_numpy_centered[t].T).T 
                                final_global_poses[t] = rotated_pose + future_global_pos[abs_t]
                                
                            full_prediction_abs.append(final_global_poses)

                    # 4. Final Save
                    final_pred_abs = np.concatenate(full_prediction_abs, axis=0)
                    final_pred_centered = np.concatenate(full_prediction_centered, axis=0)
                    
                    all_preds_saved.append(final_pred_abs)
                    zeroed_output.append(final_pred_centered)

                    #Create the prediction skeleton object
                    # final_pred_abs is (Total_Horizon, 22, 3)
                    # Create the prediction skeleton object
                    if len(final_pred_abs) > 0:
                        
                        # 1. Get the absolute 3D locations for all 22 joints
                        last_pred_pose = final_pred_abs[-1] 
                        
                        # 2. Get the KF's predicted root (pelvis) position 
                        last_root = future_global_pos[-1]
                        
                        # 3. Draw it!
                        draw_prediction_on_image(image_left_ocv, last_pred_pose, last_root, camera_info, image_scale)
                        #Draw the ground truth current skeleton (sanity check)
                        # raw_input_batch = keypoints_array[np.newaxis, :, :] 
                        # h36m_current = zed34_to_h36m32(raw_input_batch)[0] # Shape (32, 3)
                        # current_root = h36m_current[0] 
                        # current_pose_22 = h36m_current[joint_used_xyz] 
                        # draw_prediction_on_image(image_left_ocv, current_pose_22, current_root, camera_info,image_scale)

                    else:
                        prediction_skeleton = None



            # Update GL view
            # viewer.update_view(image, bodies, prediction_skeleton) 
            
            # Update OCV view
            image_left_ocv = image.get_data()
            cv_viewer.render_2D(image_left_ocv,image_scale, bodies.body_list, body_param.enable_tracking, body_param.body_format)
            cv2.imshow("ZED | 2D View", image_left_ocv)
            
            
            key = cv2.waitKey(key_wait)
            if key == 113: # for 'q' key
                print("Exiting...")
                break
            if key == 109: # for 'm' key
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
    parser.add_argument('--input_svo_file', type=str, help='Path to an .svo file, if you want to replay it',default = '')
    parser.add_argument('--ip_address', type=str, help='IP Adress, in format a.b.c.d:port or a.b.c.d, if you have a streaming setup', default = '')
    parser.add_argument('--resolution', type=str, help='Resolution, can be either HD2K, HD1200, HD1080, HD720, SVGA or VGA', default = '')
    opt = parser.parse_args()
    if len(opt.input_svo_file)>0 and len(opt.ip_address)>0:
        print("Specify only input_svo_file or ip_address, or none to use wired camera, not both. Exit program")
        exit()
    main(opt) 