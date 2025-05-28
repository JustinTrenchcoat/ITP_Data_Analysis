import h5py
import numpy as np
import os
import traceback
from tqdm import tqdm
from helper import *

def decode_ascii(matlab_str):
    return ''.join([chr(c) for c in matlab_str])

# Path to datasets folder
datasets_dir = 'datasets'

all_diff = []
max_all = []

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
                def read_var(varname):
                    return np.array(f[varname]).reshape(-1)

                pr_filt = read_var('pr_filt')
                lat = read_var("latitude")

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
print("Max depth difference in range:", max(max_all))




# import matplotlib.pyplot as plt
# import numpy as np
# import argparse
# import h5py
# from helper import *

# # set up file path
# full_path = r'D:\EOAS\ITP_Data_Analysis\datasets\itp1cormat\cor0001.mat'

# def smartPlot(plotType, sample, sysNum, profNum):
#     if plotType == "tvd":
#         depth = Profile.depth(sample)
#         temp = Profile.potential_temperature(sample)
#         time = Profile.python_datetime(sample)
#         time = time.strftime("%Y-%m-%d %H:%M:%S")

#         plotHelper(temp, depth, "Temperature", "Depth", sysNum,profNum,time)

#         plt.savefig(f"plots/temp_vs_depth_sys_{sysNum}_prof_{profNum}.png")
#         plt.show()
#     elif plotType == "svd":
#         depth = Profile.depth(sample)
#         salinity = Profile.absolute_salinity(sample)
#         time = Profile.python_datetime(sample)
#         time = time.strftime("%Y-%m-%d %H:%M:%S")

#         plotHelper(salinity, depth, "Salinity", "Depth", sysNum,profNum,time)
#         plt.savefig(f"plots/sal_vs_depth_sys_{sysNum}_prof_{profNum}.png")
#         plt.show()
 


# def plotHelper(x,y, xlabel, ylabel, sysNum, profNum,time):
#     # Filter y between 200 and 500
#     mask = (y >= 200) & (y <= 500)
#     x_filtered = x[mask]
#     y_filtered = y[mask]
#     plt.plot(x_filtered, y_filtered, marker='o',linestyle='dashed',linewidth=2, markersize=12)
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.title(f"{xlabel} vs {ylabel}, System# {sysNum} Profile# {profNum}, Time {time}")
#     plt.grid(True)
#     plt.gca().invert_yaxis()
#     # Optional: Rotate date labels for clarity
#     plt.xticks(rotation=45)
#     plt.show()

# def statHelper(all_depth_differences):
#     print("Statistical Summary of Depth Differences:")
#     print(f"Count         : {len(all_depth_differences)}")
#     print(f"Min           : {np.min(all_depth_differences)}")
#     print(f"Max           : {np.max(all_depth_differences)}")
#     print(f"Mean          : {np.mean(all_depth_differences)}")
#     print(f"Median        : {np.median(all_depth_differences)}")
#     print(f"Std Dev       : {np.std(all_depth_differences)}")
#     print(f"Variance      : {np.var(all_depth_differences)}")
#     print(f"25th Percentile (Q1): {np.percentile(all_depth_differences, 25)}")
#     print(f"75th Percentile (Q3): {np.percentile(all_depth_differences, 75)}")
#     print(f"IQR           : {np.percentile(all_depth_differences, 75) - np.percentile(all_depth_differences, 25)}")



# try:
#     with h5py.File(full_path, 'r') as f:
#         def read_var(varname):
#             return np.array(f[varname]).squeeze()
#         sa_cor = read_var('sa_cor')
#         pr_filt = read_var('pr_filt')
#         lat = read_var("latitude")
#         lon = read_var("longitude")

#         # Filter out NaNs
#         valid_mask = ~np.isnan(sa_cor) & ~np.isnan(pr_filt)
#         # valid_mask = ~np.isnan(pr_filt)
#         sa_cor = sa_cor[valid_mask]
#         pr_filt = pr_filt[valid_mask]

#         depth = height(pr_filt, lat)
#         depth = np.sort(depth)
#         depth_diff = np.diff(depth)
#         statHelper(depth_diff)
        
#         # Plot the histogram
#         plt.figure(figsize=(10, 6))
#         plt.hist(depth_diff, bins='auto', range=(0,2), edgecolor='black')  # adjust bin count or bin edges here
#         plt.title("Histogram of Depth Differences Between Consecutive Data Points")
#         plt.xlabel("Depth Difference")
#         plt.ylabel("Frequency")
#         plt.grid(True)
#         plt.tight_layout()
#         plt.show()
#         # plotHelper(sa_cor,depth, "Salinity", "Depth", 112, 1,"made up time")
# except Exception:
#     print("Error Reading Data!")
