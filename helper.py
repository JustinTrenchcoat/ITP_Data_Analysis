import matplotlib.pyplot as plt
from itp.profile import Profile
import gsw
from tqdm import tqdm
import os
import h5py
import numpy as np

# convert pressure to height
def height(pressure, latitude):
        return -gsw.conversions.z_from_p(pressure, latitude)


# check missing variable fields 
def checkField():
        # Directory and target variables
    datasets_dir = "datasets"
    target_vars = ["sa_adj", "te_cor", "co_adj", "pr_filt"]

    # Output files
    with open("good_data.txt", "w") as good_file, open("bad_list.txt", "w") as bad_file:
        folders = sorted([
            f for f in os.listdir(datasets_dir)
            if os.path.isdir(os.path.join(datasets_dir, f)) and f.startswith("itp") and f.endswith("cormat")
        ])

        for folder_name in tqdm(folders, desc="Processing folders", unit="folder"):
            folder_path = os.path.join(datasets_dir, folder_name)
            mat_files = sorted([
                f for f in os.listdir(folder_path)
                if f.endswith(".mat")
            ])

            for file_name in tqdm(mat_files, desc=f"{folder_name}", unit="file", leave=False):
                file_path = os.path.join(folder_path, file_name)

                try:
                    with h5py.File(file_path, 'r') as f:
                        all_vars = list(f.keys())
                        missing_vars = [var for var in target_vars if var not in all_vars]

                        if not missing_vars:
                            good_file.write(f"{folder_name}/{file_name}\n")
                        else:
                            bad_file.write(
                                f"{folder_name}/{file_name} | Missing: {', '.join(missing_vars)} | Found: {', '.join(all_vars)}\n"
                            )
                except Exception as e:
                    bad_file.write(f"{folder_name}/{file_name} | Error: {str(e)}\n")


# filter through all files in dataset
def traverse_datasets(datasets_dir, func):
    """
    Traverse through the dataset folders and apply the given function 'func' to each .mat file.

    Parameters:
    - datasets_dir: str, path to the main datasets folder
    - func: callable, function to apply on each .mat file
            signature should be func(folder_name, file_name, file_path)
    """
    folders = sorted([
        f for f in os.listdir(datasets_dir)
        if os.path.isdir(os.path.join(datasets_dir, f)) and f.startswith("itp") and f.endswith("cormat")
    ])

    for folder_name in tqdm(folders, desc="Processing folders", unit="folder"):
        folder_path = os.path.join(datasets_dir, folder_name)
        mat_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".mat")])

        for file_name in tqdm(mat_files, desc=f"{folder_name}", unit="file", leave=False):
            file_path = os.path.join(folder_path, file_name)
            try:
                func(folder_name, file_name, file_path)
            except Exception as e:
                print(f"Error processing {folder_name}/{file_name}: {e}")