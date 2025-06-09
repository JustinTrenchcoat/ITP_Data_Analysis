import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
import traceback
from scipy.interpolate import interp1d
from helper import *
import re
from scipy.io import savemat

# this script looks through each profile extract depth, temperature and every other measurement, interpolate the values by depth grid of 0.25m
# should it look like a dataframe or?

# set up file path
# full_path = r'D:\EOAS\ITP_Data_Analysis\datasets\itp62cormat\cor0008.mat'
full_path = r'D:\EOAS\ITP_Data_Analysis\goldData\itp121cormat\cor0604.mat'

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
        print(f'Pressure length is {pr_filt.size}')
        lat = read_var(f, "latitude")
        lon = read_var(f, "longitude")
        psdate = read_var(f, "psdate")
        pstart = read_var(f, "pstart")
        pedate = read_var(f, "pedate")
        pstop = read_var(f, "pstop")
        te_adj = read_var(f, "te_adj")
        sa_adj = read_var(f, "sa_adj")

        psdate_str = clean(psdate)
        pstart_str = clean(pstart)
        pedate_str = clean(pedate)
        pstop_str = clean(pstop)
        
        valid_mask = ~np.isnan(te_adj) & ~np.isnan(pr_filt) & ~np.isnan(sa_adj)
        print(f"Total number of valid values: {sum(valid_mask)}")
        pr_filt = pr_filt[valid_mask]
        te_adj = te_adj[valid_mask]
        sa_adj = sa_adj[valid_mask]
        
        # Step 1: Sort by depth
        depth = height(pr_filt, lat)
        print(f"max depth is {depth.max()}, size is {depth.size}")
        sorted_indices = np.argsort(depth)
        depths_sorted = depth[sorted_indices]
        temperatures_sorted = te_adj[sorted_indices]
        salinity_sorted = sa_adj[sorted_indices]

        if depths_sorted.size == 0:
             print(f'File {full_path} has empty depth')
        if temperatures_sorted.size == 0:
             print(f'File {full_path} has empty Temp')
        if salinity_sorted.size == 0:
             print(f'File {full_path} has empty Sal')
        # Step 2: Create regular depth grid (every 0.25 m)
        regular_depths = np.arange(depths_sorted.min(), depths_sorted.max(), 0.25)
        print(f"Depth range: Max{depths_sorted.max()}, Min: {depths_sorted.min()}")

        if len(regular_depths) < 2:
            print(f"Skipping interpolation for {full_path}: not enough valid points.")

        
        unique_vals, counts = np.unique(depths_sorted, return_counts=True)
        duplicates = unique_vals[counts > 1]

        print("Duplicate values:", duplicates)
        if np.any(counts > 1):
            print("Duplicate x values found!")
        plot(temperatures_sorted, depths_sorted)

        # Step 3: Interpolate each variable
        temp_interp = interp1d(depths_sorted, temperatures_sorted, kind='linear', bounds_error=False, fill_value='extrapolate')
        sal_interp = interp1d(depths_sorted, salinity_sorted, kind='linear', bounds_error=False, fill_value='extrapolate')
        interpolated_temperatures = temp_interp(regular_depths)
        interpolated_salinity = sal_interp(regular_depths)
        
        if not (len(regular_depths) == len(interpolated_temperatures) == len(interpolated_salinity)):
              raise ValueError(f"Length mismatch in interpolated arrays in file: {full_path}")

        mdic = {"Depth":regular_depths, "Temperature":interpolated_temperatures, "Salinity":interpolated_salinity, "lat":lat,
                 "lon":lon, "startDate":psdate, "startTime":pstart, "endDate":pedate, "endTime":pstop}
        # savemat(f"test_startTime{psdate_str}.mat", mdic)


except Exception as e:
    print(f"Error processing file: {full_path}")
    traceback.print_exc()