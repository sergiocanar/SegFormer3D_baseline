import os
import sys
import yaml
from tqdm import tqdm
from typing import Dict

import torch
import numpy as np

import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation

from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import Compose
from monai.transforms import Activations
from monai.transforms import AsDiscrete

sys.path.append("../../../")

from architectures.build_architecture import build_architecture

def load_config(config_path: str) -> Dict:
    """Loads the yaml config file

    Args:
        config_path (str): Path to the config file

    Returns:
        Dict: Configuration dictionary
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def get_max_slice(vol):
    '''
    Get the slice with the largest area in a 3D volume
    -----------------------------------------------
    Parameters:
    - vol: 3D numpy array
    -----------------------------------------------
    Returns:
    - max_slice: int
    - max_area: int
    
    '''
    _, _, z = vol.shape
    max_area = 0
    max_slice = 0
    
    for i in range(z):
        area = np.sum(vol[:, :, i])
        if area > max_area:
            max_area = area
            max_slice = i
    
    return max_slice, max_area

def save_biggest_area_img(vol_og, vol_gt, vol_pred, max_slice, save_path):
    
    '''
    Save the image with the largest area in the 3D volume as a PNG file
    ------------------------------------------------
    Parameters:
    -----------------------------------------------
    - vol_og: 3D numpy array: Channel from the input tensor
    - vol_gt: 3D numpy array: Ground truth segmentation volume
    - vol_pred: 3D numpy array: Predicted segmentation volume
    - max_slice: int: Slice with the largest area in the 3D volume
    - max_area: int: Area of the largest slice
    - save_path: str: Path to save the image
    -----------------------------------------------
    Returns:
    - None
    
    '''
    # Define the colors for each label (0: background, 1: non-enhancing, 2: whole tumor, 3: enhancing tumor)
    colors = [(0, 0, 0),  (255, 255, 0), (255, 0, 0),(0, 255, 255)]  # RGB values: black, yellow, red, cyan

    cmap = mcolors.ListedColormap(np.array(colors) / 255.0)

    # Create the figure and axis
    fig, ax = plt.subplots(1, 3, figsize=(20, 10))
    
    ax[0].imshow(vol_og[:, :, max_slice], cmap='gray', interpolation='none')  # Base grayscale image
    ax[0].set_title('Original volume')
    ax[0].axis('off')

    # Ground truth segmentation with volume overlay
    ax[1].imshow(vol_og[:, :, max_slice], cmap='gray', interpolation='none')  # Base grayscale image
    ax[1].imshow(vol_gt[:, :, max_slice], cmap=cmap, interpolation='none', vmin=0, vmax=3, alpha=0.5)  # Segmentation overlay with transparency
    ax[1].set_title('Ground truth segmentation')
    ax[1].axis('off')

    # Predicted segmentation with volume overlay
    ax[2].imshow(vol_og[:, :, max_slice], cmap='gray', interpolation='none')  # Base grayscale image
    ax[2].imshow(vol_pred[:, :, max_slice], cmap=cmap, interpolation='none', vmin=0, vmax=3, alpha=0.5)  # Segmentation overlay with transparency
    ax[2].set_title('Predicted segmentation')
    ax[2].axis('off')

    # Save the figure
    plt.savefig(save_path)
    plt.close(fig)


def save_animation(vol_og, vol_gt, vol_pred, save_path):
    '''
    Save the 3D volume as an animation (GIF).
    ------------------------------------------------
    Parameters:
    -----------------------------------------------
    - vol_og: 3D numpy array: Channel from the input tensor
    - vol_gt: 3D numpy array: Ground truth segmentation volume
    - vol_pred: 3D numpy array: Predicted segmentation volume
    - save_path: str: Path to save the animation
    -----------------------------------------------
    Returns:
    - None
    '''

    # Define the colors for each label (0: background, 1: non-enhancing, 2: whole tumor, 3: enhancing tumor)
    colors = [(0, 0, 0),  (255, 255, 0), (255, 0, 0),(0, 255, 255)]  # RGB values: black, yellow, red, cyan
    cmap = mcolors.ListedColormap(np.array(colors) / 255.0)  # Normalize to [0,1] for matplotlib

    # Get the number of slices (assumes all volumes have the same shape)
    _, _, num_slices = vol_og.shape

    # Create a figure and axes
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))

    # Function to update the figure for each frame (slice)
    def update(i):
        for ax in axes:
            ax.clear()  # Clear the previous frame

        # Display the original volume (grayscale)
        axes[0].imshow(vol_og[:, :, i], cmap='gray')
        axes[0].set_title('Original Volume')

        # Display the ground truth segmentation with the custom colormap and alpha 0.5
        gt_img = axes[1].imshow(vol_gt[:, :, i], cmap=cmap, alpha=0.5)
        axes[1].set_title('Ground Truth')

        # Display the predicted segmentation with the custom colormap and alpha 0.5
        pred_img = axes[2].imshow(vol_pred[:, :, i], cmap=cmap, alpha=0.5)
        axes[2].set_title('Predicted')

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=num_slices, repeat=True)

    # Save the animation as a GIF
    ani.save(save_path, writer='pillow', fps=10)

    plt.close(fig)  
    

def dice_score_per_class(pred, true, class_index):
    pred_binary = (pred == class_index).astype(int)
    true_binary = (true == class_index).astype(int)
    intersection = (pred_binary * true_binary).sum()
    union = pred_binary.sum() + true_binary.sum()
    if union == 0:
        return 1.0  # Handle empty masks
    return 2.0 * intersection / union



# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
this_file_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(this_file_dir)
grandparent_dir = os.path.dirname(parent_dir)
grand_grandparent_dir = os.path.dirname(grandparent_dir)

config_path = os.path.join(this_file_dir, "config.yaml")
weights2_path = '/home/scanar/BCV_project/SegFormer3D_baseline/experiments/brats2018/experiment6_final_sweep_bs4/model_checkpoints/best_dice_checkpoint/79ctfnn7/pytorch_model.bin'

# Model configuration
model_config = load_config(config_path)

# Build the SegFormer architecture
model = build_architecture(model_config)
model.to(device)

# Declare model weights
initial_weights = model.state_dict()

# Load weights
model_dict = torch.load(weights2_path)
model.load_state_dict(model_dict, strict=False)

# ---------------------------- #
# Inference
model.eval()

# Paths to the tensor and label files

data_path = os.path.join(grand_grandparent_dir, 'data')
brats2017_seg_path = os.path.join(data_path, 'brats2017_seg')
val_csv_path = os.path.join(brats2017_seg_path, 'val2018.csv')
val_df = pd.read_csv(val_csv_path)

val_paths = val_df['data_path'].to_numpy()
val_cases = val_df['case_name'].to_numpy()

plots_folder = os.path.join(this_file_dir, 'plots')
if not os.path.exists(plots_folder):
    os.makedirs(plots_folder)
    
txt_results_path = os.path.join(plots_folder, 'results.txt')

dice_Avg_lt = []
dice_class1_lt = []
dice_class2_lt = []
dice_class3_lt = []


with open(txt_results_path, 'w') as f:
    f.write('Results\n')

    with tqdm(total=len(val_paths)) as pbar:
        for i in range(len(val_paths)):
            
            #Extract the tensor and label paths for the i-th case in the validation set
            val_path_i = val_paths[i]
            val_case_i = val_cases[i]
            
            tensor_fold_path = val_path_i 
            tensor_path = os.path.join(tensor_fold_path, val_case_i + '_modalities.pt') #Path to the tensor file (is numpy array)
            label_path = os.path.join(tensor_fold_path, val_case_i + '_label.pt') #Path to the label file (is numpy array)
            
            
            #Load the tensor and label files
            input_np = torch.load(tensor_path) 
            label_np = torch.load(label_path)
            
            #Convert the numpy arrays to tensors and move them to the device
            input_tensor = torch.from_numpy(input_np).to(device).float()
            label_tensor = torch.from_numpy(label_np).to(device)

            #Add a batch dimension to the input and label tensors for the model
            input_tensor = input_tensor.unsqueeze(0)
            label_tensor = label_tensor.unsqueeze(0)

            # ---------------------------- #
            
            # Define the post-transforms for the predictions
            post_transform = Compose(
                Activations(sigmoid=True), # Sigmoid activation to the model output
                AsDiscrete(argmax=False, threshold=0.5) # Thresholding the output
            )

            # Get the predicted segmentation using sliding window inference from MONAI

            logits = sliding_window_inference(
                inputs=input_tensor,
                roi_size=[128, 128, 128],
                sw_batch_size=4,
                predictor=model,
                overlap=0.5,
            )

            decollated_preds = decollate_batch(logits) #Decollate the batch of predictions
            
            #Convert the predictions to the final output format
            output_convert = [
                post_transform(val_pred_tensor) for val_pred_tensor in decollated_preds
            ]
            
            #Get the final prediction volume
            final_pred = output_convert[0]
            final_pred_np = final_pred.cpu().numpy() #Convert the tensor to numpy array
            final_pred_np = final_pred_np > 0.5 #Threshold the output to get the final segmentation (0.5 threshold because of the sigmoid activation)

            # #Label 0: Background
            # #Label 1: Non-enhancing tumor (NCR/NET)
            # #Label 2: WT (whole tumor) = ET (enhancing tumor) + NCR/NET (non-enhancing tumor)
            # #Label 3: ET (enhancing tumor)


            class1_vol, class2_vol, class3_vol = final_pred_np #Split the final prediction volume into the 3 classes

            # print(np.sum(class1_vol))
            # print(np.sum(class2_vol))
            # print(np.sum(class3_vol))

            #Combine the 3 classes to get the final prediction volume (0: background, 1: non-enhancing, 2: whole tumor, 3: enhancing tumor)
            final_pred_vol = np.zeros_like(class1_vol)
            final_pred_vol = np.sum([class1_vol, class2_vol, class3_vol], axis=0)

            final_pred_vol = final_pred_vol.astype(np.uint8)
            final_pred_tensor = torch.from_numpy(final_pred_vol).to(device)
            final_pred_tensor = final_pred_tensor.unsqueeze(0) #Add a batch dimension to the final prediction tensor
            #-----------------------------#
            
            # Label volume creation
            
            label_tensor = torch.load(label_path)
            gt_final_vol = np.zeros_like(final_pred_vol)

            label1_np, label2_np, label3_np = label_tensor #Split the label volume into the 3 classes
            
            gt_final_vol = np.sum([label1_np, label2_np, label3_np], axis=0)
            gt_final_tensor = torch.from_numpy(gt_final_vol).to(device)
            gt_final_tensor = gt_final_tensor.unsqueeze(0)
            #-----------------------------#
            
            # Extract a channel from the input tensor for visualization
            input_np_channel = input_np[0, :, :]
            
            #-----------------------------#
            
            case_plots_folder = os.path.join(plots_folder, val_case_i)
            if not os.path.exists(case_plots_folder):
                os.makedirs(case_plots_folder)
            
            #-----------------------------#
                
            max_slice, max_area = get_max_slice(final_pred_vol)
            
            save_img_path = os.path.join(case_plots_folder, val_case_i + '_max_area.png')
            save_animation_path = os.path.join(case_plots_folder, val_case_i + '_animation.gif')
            save_np_path = os.path.join(case_plots_folder, val_case_i + '_final_pred.npy')
            save_gt_path = os.path.join(case_plots_folder, val_case_i + '_gt.npy')
            
            save_biggest_area_img(input_np_channel, gt_final_vol, final_pred_vol, max_slice, save_img_path)
            save_animation(input_np_channel, gt_final_vol, final_pred_vol, save_animation_path)
            np.save(save_np_path, final_pred_vol)
            np.save(save_gt_path, gt_final_vol)
            dice_class1 = dice_score_per_class(final_pred_vol, gt_final_vol, 1)
            dice_class2 = dice_score_per_class(final_pred_vol, gt_final_vol, 2)
            dice_class3 = dice_score_per_class(final_pred_vol, gt_final_vol, 3)
            
            mean_dice = (dice_class1 + dice_class2 + dice_class3) / 3
            
            print(f"Case: {val_case_i} with Max area: {max_area} and Mean Dice: {mean_dice}")
            print(f"Dice for class 1: {dice_class1}")
            print(f"Dice for class 2: {dice_class2}")
            print(f"Dice for class 3: {dice_class3}")
            
            f.write(f"Case: {val_case_i} with Max area: {max_area} and Mean Dice: {mean_dice}\n")
            f.write(f"Dice for class 1: {dice_class1}\n")
            f.write(f"Dice for class 2: {dice_class2}\n")
            f.write(f"Dice for class 3: {dice_class3}\n")
            f.write("\n")
            
            dice_Avg_lt.append(mean_dice)
            dice_class1_lt.append(dice_class1)
            dice_class2_lt.append(dice_class2)
            dice_class3_lt.append(dice_class3)
            
            pbar.set_description(f"Case: {val_case_i} with Max area: {max_area}")
            pbar.update(1)
        
        f.write(f"Average Dice: {np.mean(dice_Avg_lt)}\n")
        f.write(f"Average Dice for class 1: {np.mean(dice_class1_lt)}\n")
        f.write(f"Average Dice for class 2: {np.mean(dice_class2_lt)}\n")
        f.write(f"Average Dice for class 3: {np.mean(dice_class3_lt)}\n")
        f.write("\n")
        f.write("End of results\n")