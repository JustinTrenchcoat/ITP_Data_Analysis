import matplotlib.pyplot as plt
import gsw
from tqdm import tqdm
import os
import h5py
import numpy as np
import shutil

# convert pressure to height
def height(pressure, latitude):
        return -gsw.conversions.z_from_p(pressure, latitude)


def read_var(f, varname):
        data = np.array(f[varname])
        if data.dtype == "uint16":
            return data.tobytes().decode('utf-16-le')
        return data.reshape(-1)


# check missing variable fields 
def checkField(datasets_dir):
    # datasets_dir = "datasets"
    target_vars = ["sa_adj", "te_adj", "pr_filt"]

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


def countData(datasets_dir):
    total_profiles = 0

    # Loop over all itp*cormat folders
    for folder_name in os.listdir(datasets_dir):
        folder_path = os.path.join(datasets_dir, folder_name)

        if os.path.isdir(folder_path) and folder_name.startswith("itp") and folder_name.endswith("cormat"):
        # Get all .mat files
            mat_files = [f for f in os.listdir(folder_path) if f.endswith(".mat")]
            profile_count = len(mat_files)

            if profile_count == 0:
            # Delete empty folder
                shutil.rmtree(folder_path)
                print(f"Deleted empty folder: {folder_name}")
            else:
                total_profiles += profile_count
                print(f"{folder_name}: {profile_count} profiles")

    # Print total
    print(f"\nTotal number of remaining profiles: {total_profiles}")
