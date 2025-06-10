import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
import traceback
from scipy.interpolate import CubicSpline
from helper import *
import re
from scipy.io import savemat

# this script looks through each profile extract depth, temperature and every other measurement, interpolate the values by depth grid of 0.25m
# should it look like a dataframe or?

# set up file path
# full_path = r'D:\EOAS\ITP_Data_Analysis\datasets\itp62cormat\cor0008.mat'
full_path = r'D:\EOAS\ITP_Data_Analysis\goldData\itp115cormat\cor0534.mat'

new_dir = "testData"
folder_name = "test"



def clean(x):
    # Convert to string
    x_str = str(x)
    # Remove brackets, whitespace
    x_str = x_str.strip("[] ").replace(" ", "")
    # Replace any non-alphanumeric or underscore/dash/dot characters with underscore
    x_str = re.sub(r'[^\w\-.]', '_', x_str)
    return x_str

def plot(x, y):
    plt.plot(x, y, marker='o',linestyle='dashed',linewidth=2, markersize=12)
    plt.xlabel("test x")
    plt.ylabel("test y")
    plt.title("test Plot")
    plt.grid(True)
    plt.gca().invert_yaxis()
    # Optional: Rotate date labels for clarity
    plt.xticks(rotation=45)
    plt.show()




try:
    with h5py.File(full_path, 'r') as f:
        pr_filt = read_var(f, 'pr_filt')
        lat = read_var(f, "latitude")
        lon = read_var(f, "longitude")
        psdate = read_var(f, "psdate")
        pstart = read_var(f, "pstart")
        pedate = read_var(f, "pedate")
        pstop = read_var(f, "pstop")
        te_adj = read_var(f, "te_adj")
        sa_adj = read_var(f, "sa_adj")

        helPlot(te_adj, pr_filt)
        
        valid_mask = ~np.isnan(te_adj) & ~np.isnan(pr_filt) & ~np.isnan(sa_adj)
        pr_filt = pr_filt[valid_mask]
        te_adj = te_adj[valid_mask]
        sa_adj = sa_adj[valid_mask]
        
        # Step 1: Sort by depth
        depth = height(pr_filt, lat)
        sorted_indices = np.argsort(depth)
        depths_sorted = depth[sorted_indices]
        temperatures_sorted = te_adj[sorted_indices]
        salinity_sorted = sa_adj[sorted_indices]
        if depth.max() > 800:
            raise ValueError(f'Abnormal Depth of {depth.max()} Please check profile {full_path}')
        
        # add checks for possible empty depth files
        if (depths_sorted.size == 0) or (temperatures_sorted.size == 0):
             print(f'File {full_path} has empty entries')

        plot(temperatures_sorted, depths_sorted)

        # Find T_Max first:
        # deep_index has all indeces of depth array who's value is above 200m
        deep_index = np.where(depths_sorted >= 200)[0]
        # temp_max_idx is the index of max value of all temperature values measured below 200 m, also the index for depth
        temp_max_idx = np.argmax(temperatures_sorted[deep_index])
        # temp_max_depth_idx is the index value of depth, for the depth with max temperature below 200 m
        temp_max_depth_idx = deep_index[temp_max_idx]
        T_Max = temperatures_sorted[temp_max_depth_idx]
        T_Max_Depth = depths_sorted[temp_max_depth_idx]
        print(temp_max_depth_idx)
        print(f'T_Max = {T_Max}')
        print(f'T_Max_Depth = {T_Max_Depth}')

        # look up from T_Max, find the T_min between (100, T_Max_depth)
        # select only from Tmin (above 400m?)to 5+Tmax for interpolation:
        # surface_index is all indeces of depth array who's above T_Max and below 100m 
        surface_index = np.where((100 <= depths_sorted) & (depths_sorted <= T_Max_Depth))[0]
        temp_min_idx = np.argmin(temperatures_sorted[surface_index])
        temp_min_depth_idx = surface_index[temp_min_idx]
        print(f"temp_min is : {temperatures_sorted[temp_min_depth_idx]}")
        print(f'Tmin depth is: {depths_sorted[temp_min_depth_idx]}')

        filter_mask = np.arange(temp_min_depth_idx, temp_max_depth_idx + 20)
        depth_filtered = depths_sorted[filter_mask]
        temp_filtered = temperatures_sorted[filter_mask]
        sal_filtered = salinity_sorted[filter_mask]
        helPlot(temp_filtered, depth_filtered)

        # Step 2: Create regular depth grid (every 0.25 m)
        regular_depths = np.arange(depth_filtered.min(), depth_filtered.max(), 0.25)
        

        # check 2 for encoutnering zero
        if len(regular_depths) < 2:
            print(f"{full_path} does not enough valid points.")

        # add check 3 for interpolation error:
        unique_vals, counts = np.unique(depth_filtered, return_counts=True)
        duplicates = unique_vals[counts > 1]
        if duplicates.size > 0:
            print(f"Duplicates found for file{full_path}")

        # Step 3: Interpolate each variable
        temp_interp = CubicSpline(depth_filtered, temp_filtered)
        sal_interp = CubicSpline(depth_filtered, sal_filtered)
        interpolated_temperatures = temp_interp(regular_depths)
        interpolated_salinity = sal_interp(regular_depths)
        
        if not (len(regular_depths) == len(interpolated_temperatures) == len(interpolated_salinity)):
              raise ValueError(f"Length mismatch in interpolated arrays in file: {full_path}")

        mdic = {"Depth":regular_depths, "Temperature":interpolated_temperatures, "Salinity":interpolated_salinity, "lat":lat,
                 "lon":lon, "startDate":psdate, "startTime":pstart, "endDate":pedate, "endTime":pstop}
        plot(interpolated_temperatures, regular_depths)
        
        # savemat(f"test_startTime{psdate_str}.mat", mdic)


except Exception as e:
    print(f"Error processing file: {full_path}")
    traceback.print_exc()