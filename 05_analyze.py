import pickle
import numpy as np
import h5py
import os
import traceback
from tqdm import tqdm
from helper import *

# Path to datasets folder
datasets_dir = 'goldData'

all_diff = []
max_all = []
weird_list=[]
weird_max=[]
weird_min = []

# Loop over every itp*cormat folder
for folder_name in sorted(os.listdir(datasets_dir)):
    folder_path = os.path.join(datasets_dir, folder_name)
    if not os.path.isdir(folder_path):
        continue  # skip non-folders

    print(f"\nProcessing folder: {folder_name}")

    # Get all .mat files
    all_mat_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.mat')])

    for filename in tqdm(all_mat_files, desc=f"Filtering {folder_name}", leave=False):
        full_path = os.path.join(folder_path, filename)

        try:
            with h5py.File(full_path, 'r') as f:

                pr_filt = read_var(f, 'pr_filt')
                lat = read_var(f, "latitude")

                valid_mask = ~np.isnan(pr_filt)
                pr_filt = pr_filt[valid_mask]

                depth = height(pr_filt, lat)

                # Filter depth to only between 600 and 200
                in_range_mask = (depth >= 200) & (depth <= 600)

                depth_in_range = depth[in_range_mask]

                # Only calculate difference if we have enough points
                if len(depth_in_range) > 1:
                    depth_in_range = np.sort(depth_in_range)
                    depth_diff = np.diff(depth_in_range)
                    max_depth = max(depth_diff)
                    min_depth = min(depth_diff)
                    if (max_depth > 1) or (min_depth <0):
                        weird_list.append(full_path)
                        weird_max.append(max_depth)
                        weird_min.append(min_depth)
                    max_all.append(max_depth)
                    all_diff.extend(depth_diff)

        # Inside your for-loop where the exception occurs:
        except Exception as e:
            print(f"Error processing file: {filename}")
            traceback.print_exc()

# Convert to numpy array and save to pickle
all_diff = np.array(all_diff)

import pickle

with open("depth_differences.pkl", "wb") as f:
    pickle.dump(all_diff, f)

print("Saved depth differences to 'depth_differences.pkl'")
print("Max of Max depth difference in range:", max(max_all))
print("List of files with abnormal depth difference")
for i in range(len(weird_list)):
    print(f"File {weird_list[i]} has max value of {weird_max[i]}, min value of {weird_min[i]}")