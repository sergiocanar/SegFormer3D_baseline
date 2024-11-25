import os 
import numpy as np 
import nibabel as nib
import matplotlib.pyplot as plt

def load_volume(file_path):
    return np.load(file_path)


def post_process(pred_dir, out_dir):
    pass

if __name__ == "__main__":
    predictions_lt = []
    gt_lt = []
    
    this_dir = os.path.dirname(os.path.abspath(__file__))
    pred_dir = os.path.join(this_dir, "plots")
    folders = os.listdir(pred_dir)
    for folder in folders:
        files = os.listdir(os.path.join(pred_dir, folder))
        for file_ in files:
            if file_.endswith(".npy"):
                predictions_lt.append(os.path.join(pred_dir, folder, file_))
    
    first_pred = load_volume(predictions_lt[0])

    