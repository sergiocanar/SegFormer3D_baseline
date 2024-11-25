import os 
import numpy as np 
from tqdm import tqdm
from scipy.ndimage import label
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
from skimage.morphology import closing, ball, remove_small_objects, opening


def load_volume(file_path):
    return np.load(file_path)

def apply_closing(volume, radius=6):
    '''
    Apply morphological closing to a 3D volume
    -----------------------------------------------
    Parameters:
    - volume: 3D numpy array
    - radius: int
    -----------------------------------------------
    Returns:
    - closed_volume: 3D numpy array
    
    '''
    # Create a ball-shaped structuring element
    selem = ball(radius)
    
    # Apply morphological closing
    closed_volume = closing(volume, selem)
    
    return closed_volume

def apply_opening(volume, radius=6):
    '''
    Apply morphological opening to a 3D volume
    -----------------------------------------------
    Parameters:
    - volume: 3D numpy array
    - radius: int
    -----------------------------------------------
    Returns:
    - opened_volume: 3D numpy array
    
    '''
    # Create a ball-shaped structuring element
    selem = ball(radius)
    
    # Apply morphological opening
    opened_volume = opening(volume, selem)
    
    return opened_volume

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


def save_animation(vol_gt, vol_pred, save_path):
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
    num_slices = vol_gt.shape[2]
    
    # Create a figure and axes
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Function to update the figure for each frame (slice)
    def update(i):
        for ax in axes:
            ax.clear()  # Clear the previous frame

        # Display the original volume (grayscale)

        # Display the ground truth segmentation with the custom colormap and alpha 0.5
        gt_img = axes[0].imshow(vol_gt[:, :, i], cmap=cmap, alpha=0.5)
        axes[0].set_title('Ground Truth')

        # Display the predicted segmentation with the custom colormap and alpha 0.5
        pred_img = axes[1].imshow(vol_pred[:, :, i], cmap=cmap, alpha=0.5)
        axes[1].set_title('Predicted')

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

def post_process(pred_vol, out_path, type_of_post_processing="closing"):
    
    if type_of_post_processing == "closing":
        # Apply morphological closing
        pred_vol = apply_closing(pred_vol)
    elif type_of_post_processing == "remove_small_regions":
        # Remove small isolated regions
        pred_vol = remove_small_regions(pred_vol)
        
    elif type_of_post_processing == "opening":
        pred_vol = apply_opening(pred_vol)
    
    # Apply post-processing steps
    # Example: Remove small isolated regions
    pred_vol = remove_small_regions(pred_vol)
    
    # Save the processed volume
    np.save(out_path, pred_vol)
    
    return pred_vol

def remove_small_regions(volume, min_size=100):
    labeled_array, num_features = label(volume)
    for i in range(1, num_features + 1):
        if np.sum(labeled_array == i) < min_size:
            volume[labeled_array == i] = 0
    return volume

if __name__ == "__main__":
    predictions_dict = {}
    gt_dict = {}
    
    this_dir = os.path.dirname(os.path.abspath(__file__))
    pred_dir = os.path.join(this_dir, "plots")
    
    folders = [f for f in os.listdir(pred_dir) if os.path.isdir(os.path.join(pred_dir, f))]
    for folder in folders:
        folder_path = os.path.join(pred_dir, folder)
        files = os.listdir(folder_path)
        for file_ in files:
            if file_.endswith(".npy") and "pred" in file_:
                predictions_dict[folder] = os.path.join(folder_path, file_)    
                
            elif file_.endswith(".npy") and "gt" in file_:
                gt_dict[folder] = os.path.join(folder_path, file_)
    
                
    keys = sorted(list(predictions_dict.keys()))

    first_pred = load_volume(predictions_dict[keys[0]])
    first_gt = load_volume(gt_dict[keys[0]])
    
    post_process_dir = os.path.join(this_dir, "post_processed_opening ")
    if not os.path.exists(post_process_dir):
        os.makedirs(post_process_dir)
    
    txt_path = os.path.join(post_process_dir, "post_process_results_opening.txt")
    
    lt_dice = []
    lt_dice_class_1 = []
    lt_dice_class_2 = []
    lt_dice_class_3 = []
    
    
    with open(txt_path, "w") as f:
        f.write("Post-processing results\n")
        f.write("------------------------------------------------\n")
        dict_post_processed = {}
        
        # Post-process the predictions
        with tqdm(total=len(keys), desc="Post-processing volumes") as pbar:
            for key in keys:
                pred_vol = load_volume(predictions_dict[key])
                output_dir = os.path.join(pred_dir, key)
                save_animation_dir = os.path.join(post_process_dir, key)        
                if not os.path.exists(save_animation_dir):
                    os.makedirs(save_animation_dir)
                
                post_processed_vol_path = os.path.join(save_animation_dir, f"{key}_processed.npy")
                post_processed_vol = post_process(pred_vol, post_processed_vol_path, type_of_post_processing="closing")
                dict_post_processed[key] = post_processed_vol
                
                label_vol = load_volume(gt_dict[key])
                dice_score_class_1 = dice_score_per_class(post_processed_vol, label_vol, 1)
                dice_score_class_2 = dice_score_per_class(post_processed_vol, label_vol, 2)
                dice_score_class_3 = dice_score_per_class(post_processed_vol, label_vol, 3)
                mean_dice = (dice_score_class_1 + dice_score_class_2 + dice_score_class_3) / 3
                
                print(f"Post-processed volume: {key}")
                print(f"Dice score for class 1: {dice_score_class_1}")
                print(f"Dice score for class 2: {dice_score_class_2}")
                print(f"Dice score for class 3: {dice_score_class_3}")
                print(f"Mean Dice score: {mean_dice}")
                print()
                save_animation(label_vol, post_processed_vol, os.path.join(save_animation_dir,f"{key}_animation.gif"))
                lt_dice.append(mean_dice)
                lt_dice_class_1.append(dice_score_class_1)
                lt_dice_class_2.append(dice_score_class_2)
                lt_dice_class_3.append(dice_score_class_3)
                pbar.update(1)
            f.write("Mean Dice score: {}\n".format(np.mean(lt_dice)))
            f.write("Mean Dice score for class 1: {}\n".format(np.mean(lt_dice_class_1)))
            f.write("Mean Dice score for class 2: {}\n".format(np.mean(lt_dice_class_2)))
            f.write("Mean Dice score for class 3: {}\n".format(np.mean(lt_dice_class_3)))
            f.write("------------------------------------------------\n")