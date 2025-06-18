import pandas as pd
from scipy.io import savemat
import re
import os
from scipy.interpolate import CubicSpline
from helper import *
import math

# def count(full_path, file_name, folder_name):
#     return 1

# grid_dir = "gridData"
# countData(grid_dir)
new_dir = 'testDir'
error_list=[]



full_path = r'D:\EOAS\ITP_Data_Analysis\goldData\itp62cormat\cor0608.mat'


def makeGrid(full_path, file_name, folder_name):
    with h5py.File(full_path, 'r') as f:
        pr_filt = read_var(f, 'pr_filt')
        lat = read_var(f, "latitude")
        lon = read_var(f, "longitude")
        psdate = read_var(f, "psdate")
        pedate = read_var(f, "pedate")
        te_adj = read_var(f, "te_adj")
        sa_adj = read_var(f, "sa_adj")
        
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
            error_list.append(f'{full_path}: abnormal depth')
            raise ValueError(f'Abnormal Depth of {depth.max()}! Please check profile {full_path}')

        # add checks for possible empty depth files
        if (depths_sorted.size == 0) or (temperatures_sorted.size == 0):
             error_list.append(f'{full_path}: empty entries')
             raise ValueError(f'File {full_path} has empty entries')

        # Find T_Max first:
        # deep_index has all indeces of depth array who's value is above 200m
        deep_index = np.where(depths_sorted >= 200)[0]
        # temp_max_idx is the index of max value of all temperature values measured below 200 m, also the index for depth
        temp_max_idx = np.argmax(temperatures_sorted[deep_index])
        # temp_max_depth_idx is the index value of depth, for the depth with max temperature below 200 m
        temp_max_depth_idx = deep_index[temp_max_idx]
        T_Max_Depth = depths_sorted[temp_max_depth_idx]

        # look up from T_Max, find the T_min between (100, T_Max_depth)
        # select only from Tmin (above 400m?)to 5+Tmax for interpolation:
        # surface_index is all indeces of depth array who's above T_Max and below 100m 
        surface_index = np.where((100 <= depths_sorted) & (depths_sorted <= T_Max_Depth))[0]
        temp_min_idx = np.argmin(temperatures_sorted[surface_index])
        temp_min_depth_idx = surface_index[temp_min_idx]

        filter_mask = np.arange(temp_min_depth_idx, temp_max_depth_idx + 20)
        depth_filtered = depths_sorted[filter_mask]
        temp_filtered = temperatures_sorted[filter_mask]
        sal_filtered = salinity_sorted[filter_mask]

        # Step 2: Create regular depth grid (every 0.25 m)
        regular_depths = np.arange(depth_filtered.min(), depth_filtered.max(), 0.25)
        
        # check 2 for encoutnering zero
        if len(regular_depths) < 2:
            error_list.append(f'{full_path}: lack enough points')
            raise ValueError(f"{full_path} does not enough valid points.")

        # add check 3 for interpolation error:
        unique_vals, counts = np.unique(depth_filtered, return_counts=True)
        duplicates = unique_vals[counts > 1]
        if duplicates.size > 0:
            error_list.append(f'{full_path}: replicate value')
            print(f"Duplicates found for file{full_path}")

        # Step 3: Interpolate each variable
        temp_interp = CubicSpline(depth_filtered, temp_filtered)
        sal_interp = CubicSpline(depth_filtered, sal_filtered)
        interpolated_temperatures = temp_interp(regular_depths)
        interpolated_salinity = sal_interp(regular_depths)
        
        if not (len(regular_depths) == len(interpolated_temperatures) == len(interpolated_salinity)):
              error_list.append(f'{full_path}: mismatch length')
              raise ValueError(f"Length mismatch in interpolated arrays in file: {full_path}")
        
        # Create DataFrame
        df = pd.DataFrame({
            'Depth': regular_depths,
            'Temperature': interpolated_temperatures,
            'Salinity': interpolated_salinity
            })
        
        df['latitude'] = lat[0].round(2)
        df['longitude'] = lon[0].round(2)
        # some files have empty start/end dates
        hasStart = isinstance(psdate, str)
        if hasStart:
            startDate = pd.to_datetime(psdate, format="%m/%d/%y").date()
            df['startDate'] = startDate
        hasEnd = isinstance(pedate, str)
        if hasEnd:
            endDate = pd.to_datetime(pedate, format="%m/%d/%y").date()
            df['endDate'] = endDate        
        # Create matching subfolder in gridData
        output_subfolder = os.path.join(new_dir, folder_name)
        os.makedirs(output_subfolder, exist_ok=True)

        # Output path
        output_filename = f"{file_name.rstrip('.mat')}.csv"
        output_path = os.path.join(output_subfolder, output_filename)

        # Save to CSV
        df.to_csv(output_path, index=False)
makeGrid(full_path, 'test_file', 'test_06')
# traverse_datasets(grid_dir, count)
# folder_path = r'D:\EOAS\ITP_Data_Analysis\gridDataMat\itp100cormat'

# mat_files = [f for f in os.listdir(folder_path) if f.endswith(".mat")]
# profile_count = len(mat_files)

# if profile_count == 0:
#      print("test 1")
# else:
#     print(f"{profile_count} profiles")


# bad_profile = []
# filename = "test"

# try:
#     with h5py.File(full_path, 'r') as f:
#         # read variables from single file for later reference.
#         pr_filt = read_var(f, 'pr_filt')
#         te_adj = read_var(f, 'te_adj')
#         sa_adj = read_var(f, "sa_adj")
#         lat = read_var(f, "latitude")
#         lon = read_var(f, "longitude")

#         # Filter out NaNs
#         valid_mask = ~np.isnan(te_adj) & ~np.isnan(pr_filt) & ~np.isnan(sa_adj)
#         pr_filt = pr_filt[valid_mask]
#         te_adj  = te_adj[valid_mask]
                

#         # check for empty values
#         if (pr_filt.size == 0) or (te_adj.size == 0):
#             bad_profile.append(filename)


#         # calculate depth
#         depth = height(pr_filt, lat)
#         dep_max = np.max(depth)
#         # filter out profiles that only gets to 200m max.
#         if dep_max <= 200:
#             bad_profile.append(filename)

#         # criteria for gold standard dataset:
#         # 1. Collected in Beaufort Gyre
#         # 2. The T_Max occured 300m below the sea surface
#         # 3. There are measurements below the T_Max for at least 10m

#         # Crit.1:
#         is_Beaufort = (73 <= lat <= 81) and (-160 <= lon <= -130)

#         # Crit 2:
#         # depth_index has all indeces of depth array who's below 200m
#         depth_index = np.where(depth >= 200)[0]
#         # temp_idx_max is the index of max value of all te_adj values measured below 200 m
#         depth_index_temp_max_idx = np.argmax(te_adj[depth_index])
#         # temp_max_depth_idx is the index value of depth, for the depth with max temperature below 200 m
#         temp_max_depth_idx = depth_index[depth_index_temp_max_idx]
#         # temp_max_depth is the value of depth at the max value of te_cor values that are under 200m
#         temp_max_depth = depth[temp_max_depth_idx]
#         max_deep = (temp_max_depth >= 250)

#         # Crit 3:
#         # the greatest depth measurement has to be 10 m deeper than temp_max_depth
#         deep_enough = (dep_max >= temp_max_depth+10)
#         helPlot(te_adj,depth)


#         # if the depth of temp_max beyond 200m, then it is good profile:
#         if (deep_enough and is_Beaufort and max_deep):
#             # --- COPY GOOD FILE TO goldData ---
#             print("good Profile")

#         else:
#             print("bad_profile")

# except Exception:
#     bad_profile.append(filename)
