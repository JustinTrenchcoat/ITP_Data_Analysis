import gsw
from tqdm import tqdm
import os
import numpy as np
import shutil
import traceback
import h5py
import matplotlib.pyplot as plt


# convert pressure to height
def height(pressure, latitude):
        return -gsw.conversions.z_from_p(pressure, latitude)

def pressure(height, latitude):
    height = -height
    return gsw.conversions.p_from_z(height, latitude)

def read_var(f, varname):
        data = np.array(f[varname])
        if data.dtype == "uint16":
            return data.tobytes().decode('utf-16-le')
        return data.reshape(-1)


# check missing variable fields 
def checkField(datasets_dir):
    # datasets_dir = "datasets"
    target_vars = ["sa_adj", "te_adj", "pr_filt"]

    with open("bad_list.txt", "w") as bad_file:
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

                        if missing_vars:
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
    for folder_name in sorted(os.listdir(datasets_dir)):
        folder_path = os.path.join(datasets_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue  # skip non-folders

        print(f"\nProcessing folder: {folder_name}")

        # Get all .mat files
        all_mat_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.mat')])

        for file_name in tqdm(all_mat_files, desc=f"Filtering {folder_name}", leave=False):
            full_path = os.path.join(folder_path, file_name)

            try:
                func(full_path, file_name, folder_name)

            except Exception as e:
                print(f"Error processing file: {file_name}")
                traceback.print_exc()



def countData(datasets_dir):
    total_profiles = 0
    total_itps = 0

    # Loop over all itp*cormat folders
    for folder_name in os.listdir(datasets_dir):
        folder_path = os.path.join(datasets_dir, folder_name)

        if os.path.isdir(folder_path) and folder_name.startswith("itp") and folder_name.endswith("cormat"):
            
        # Get all .mat files
            mat_files = [f for f in os.listdir(folder_path) if f.endswith(".mat")]
            profile_count = len(mat_files)

            if profile_count == 0:
            # Delete empty folder
                # shutil.rmtree(folder_path)
                print(f"Deleted empty folder: {folder_name}")
            else:
                total_profiles += profile_count
                print(f"{folder_name}: {profile_count} profiles")
        total_itps +=1

    # Print total
    print(f"\nTotal number of remaining profiles: {total_profiles}")
    print(f"\nTotal number of ITP systems: {total_itps}")

def helPlot(x, y):
    plt.plot(x, y, marker='o',linestyle='dashed',linewidth=2, markersize=12)
    plt.xlabel("test x")
    plt.ylabel("test y")
    plt.title("test Plot")
    plt.grid(True)
    plt.gca().invert_yaxis()
    # Optional: Rotate date labels for clarity
    plt.xticks(rotation=45)
    plt.show()