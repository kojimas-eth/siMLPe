import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib import animation as animation
import json
import math
from pathlib import Path

def plot_single_prediction_error(input, pred, truth, file_name):
    """Plot the error over time for all joints"""
    total_frames = pred.shape[0]
    time_axis= np.arange(total_frames+1)

    error = pred - truth
    dist_error = np.linalg.norm(error, axis=-1)  # Shape: (Frames, Joints)
    error_per_frame=np.mean(dist_error, axis=-1)  # Shape: (Frames,)
    error_per_frame = np.concatenate(([0.0],error_per_frame))  # Add zero error for input frame
    fig = plt.figure(figsize=(16, 8))

    plt.plot(time_axis, error_per_frame, label='MPJPE (Mean Per Joint Position Error)', color='red', linewidth=2)
    plt.title('Prediction Error Over Time', fontsize=14)
    plt.xlabel('Prediction Frame Number', fontsize=12)
    plt.ylabel('Average Joint Error (Meters)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Optional: Fill the area under the curve for better visibility
    plt.fill_between(time_axis, error_per_frame, color='red', alpha=0.1)
    filename = f"plots/{file_name}.png"
    plt.savefig(filename, dpi=100)
    plt.close()
    print(f"Saved error plot to plots/{filename}")


##################################
# MAIN CODE
##################################
# 1. Define the folder you are looking in
search_folder = Path("predictions")
speed= "m"
part="walk"

# 2. Use glob to find files starting with "34f_part"
files = search_folder.glob(f"34{speed}_{part}*")

# 3. Loop through them
# specific_file is a Path object, so you can easily access its parts
cumulative_error = []
for specific_file in files:
    print(f"Found file: {specific_file.name}")

    zed = True #If using data from zed inference
    root = True #If using data with everything zeroed
    sample = 0 #which frame to analyze
    pred_length = 60

    data = np.load(specific_file)
    print(f"Loaded {len(data['inputs'])} samples from {specific_file.name}")


    # Extract arrays
    if zed:
        inputs = data['inputs']   # (Total Sample, 50, 22, 3)
        preds = data['preds']     # (Total Sample, pred length, 22, 3)
        print("The shapes of input, preds and target are",inputs.shape, preds.shape)


    else:
        inputs = data['inputs']   # (3840, 50, 32, 3)
        targets = data['targets'] # (N, 25, 32, 3)
        preds = data['preds']     # (N, 25, 32, 3)
        
        vis_target= targets[sample] # (25, 32, 3)


    if root:
        inputs = data['zero_input']   # (N, 50, 22, 3)
        preds = data['zero_output'] # (N, 25, 22, 3)
        
        vis_input= inputs[sample]   # (50, 22, 3)
        vis_pred= preds[sample]     # (25, 22, 3  )

    else:
        vis_input=  inputs[sample]   # (50, 32, 3)
        vis_pred= preds[sample]     # (25, 32, 3)

    """Handle the case where prediction length is longer than 50 frames for ground truth"""
    total_samples = inputs.shape[0]
    
    input_time = inputs.shape[1]
    pred_time = preds.shape[1]
    num_joints = inputs.shape[2]

    # 1. Simple Case: Prediction is shorter than input window
    if pred_length <= input_time:
        # Logic: Start at 'sample + 50', take the first 'pred_length' frames
        # Safety: Ensure we don't go off the end of the dataset
        target_start_idx = sample + input_time
        
        if target_start_idx < total_samples:
            vis_target = inputs[target_start_idx, :pred_length, :, :]
        else:
            print("End of dataset reached, cannot fetch GT.")
            vis_target = np.zeros((pred_length, 22, 3)) # Placeholder

    # 2. Complex Case: Prediction is longer than input window (Stitching)
    else:
        num_full_windows = math.floor(pred_length / input_time)
        remainder = pred_length % input_time
        
        target_chunks = []
        
        # A. Collect full 50-frame windows
        for i in range(num_full_windows):
            # We step forward by 50 frames for each chunk
            # Chunk 0 starts at sample+50
            # Chunk 1 starts at sample+100
            idx = sample + input_time + (i * input_time)
            
            if idx < total_samples:
                target_chunks.append(inputs[idx, :, :, :]) # Take full 50 frames
            else:
                target_chunks.append(np.zeros((50, 22, 3))) # Padding if end of file
                
        # B. Collect the remainder (if any)
        if remainder > 0:
            idx = sample + input_time + (num_full_windows * input_time)

            if idx < total_samples:
                # Take the FIRST 'remainder' frames of this window
                target_chunks.append(inputs[idx, :remainder, :, :]) 
            else:
                target_chunks.append(np.zeros((remainder, 22, 3)))

        # C. Stitch them together
        vis_target = np.concatenate(target_chunks, axis=0)

    # Final shape check
    print(f"Target Shape: {vis_target.shape}") # Should be (pred_length, 22, 3)

    #Now save the errors from each loop
    error = vis_pred - vis_target
    dist_error = np.linalg.norm(error, axis=-1)  # Shape: (Frames, Joints)
    error_per_frame=np.mean(dist_error, axis=-1)  # Shape: (Frames,)
    cumulative_error.append(error_per_frame)


all_errors = np.vstack(cumulative_error)

# Calculate Mean and Standard Deviation across the files (axis 0)
avg_error_per_frame = np.mean(all_errors, axis=0) 
std_dev_per_frame = np.std(all_errors, axis=0) # Very useful for plotting shading!
print(avg_error_per_frame.shape)
print(f"Average error at Frame 0: {avg_error_per_frame[0]}")


"""Plotting the average error across files"""
total_frames = avg_error_per_frame.shape[0]
time_axis= np.arange(total_frames+1)


error_per_frame = np.concatenate(([0.0],avg_error_per_frame))  # Add zero error for input frame
upper_bound = error_per_frame + np.concatenate(([0.0],std_dev_per_frame))  
lower_bound = error_per_frame - np.concatenate(([0.0],std_dev_per_frame))  
lower_bound = np.maximum(lower_bound, 0)



fig = plt.figure(figsize=(16, 8))
plt.plot(time_axis, error_per_frame, label='MPJPE (Mean Per Joint Position Error)', color='red', linewidth=2)
plt.title('Averaged Prediction Error Over Time', fontsize=24)
plt.xlabel('Prediction Frame Number', fontsize=20)
plt.ylabel('Average Joint Error (Meters)', fontsize=20)
plt.grid(True, alpha=0.3)

# Increase X-axis tick label size
plt.xticks(fontsize=12) 
plt.yticks(fontsize=12)

plt.fill_between(time_axis, lower_bound, upper_bound, color='red', alpha=0.1, label='Standard Deviation (±1$\sigma$)')
plt.legend(loc='upper left', fontsize=12)


output_folder = Path("plots/error")
output_folder.mkdir(parents=True, exist_ok=True)
save_path = output_folder / f"avg_{speed}_{part}_err_with_std.png"
plt.savefig(save_path, dpi=100)
plt.close()
print(f"Saved error plot to {save_path}")