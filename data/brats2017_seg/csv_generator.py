import os 
import pandas as pd

def get_csv_data(data_dir, save_dir):
    """
    Get the data from the preprocessed data and save it to a CSV file.
    data_dir: path to the preprocessed data
    save_dir: path to save the CSV file
    """
    # Check if the data directory exists
    if not os.path.isdir(data_dir):
        print(f"Directory {data_dir} does not exist.")
        return None

    case_lt = sorted(next(os.walk(data_dir), (None, None, []))[1])
    if not case_lt:
        print(f"No subdirectories found in {data_dir}.")
        return None

    paths_cases = [os.path.join(data_dir, case) for case in case_lt]
    df = pd.DataFrame({"data_path": paths_cases, "case_name": case_lt})

    df.to_csv(save_dir, index=False)
    print(f"Data saved to {save_dir}")

    return None


if __name__ == "__main__":
    file_path = os.path.abspath(__file__)
    parent_dir = os.path.dirname(file_path)
    
    train_data_dir = os.path.join(parent_dir, "BraTS2018_Training_Data", 'train_data')
    save_dir = os.path.join(parent_dir, "train2018.csv")
    
    validation_data_dir = os.path.join(parent_dir,"BraTS2018_Validation_Data", 'val_data')
    save_val_dir = os.path.join(parent_dir, "val2018.csv")

    print("Getting the data from the preprocessed data and saving it to a csv file")
    get_csv_data(train_data_dir, save_dir)
    get_csv_data(validation_data_dir, save_val_dir)
    
    