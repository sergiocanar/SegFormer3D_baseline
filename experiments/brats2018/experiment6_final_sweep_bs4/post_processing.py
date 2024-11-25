import os 
import numpy as np 
import nibabel as nib
import matplotlib.pyplot as plt

def load_volume(file_path):
    return np.load(file_path)

def post_process(pred_dir, out_dir):
    pass

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
            if file_.endswith(".npy"):
                predictions_dict[folder] = os.path.join(folder_path, file_)    
                
    keys = sorted(list(predictions_dict.keys()))

    first_pred = load_volume(predictions_dict[keys[0]])
    print(np.unique(first_pred))
