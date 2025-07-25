import numpy as np
import h5py
from helper import *
import os
from scipy.interpolate import interp1d
import pandas as pd
import re
from scipy.io import savemat



# co_adj          conductivity (mmho) after lags and calibration adjustment          
# co_cor          conductivity (mmho) after lags applied          
# itpno            ITP number          
# latitude        start latitude (N+) of profile          
# longitude     start longitude (E+) of profile          
#           
# psdate           profile UTC start date (mm/dd/yy)          
# pstart            profile UTC start time (hh:mm:ss)  
# pedate          profile UTC end date (mm/dd/yy)           
# pstop            profile UTC stop time (hh:mm:ss)     
#      
# pr_filt           low pass filtered pressure (dbar)
# sa_adj          salinity after lags and calibration adjustment          
# sa_cor          salinity after lags applied          
# te_adj           temperature (C) in conductivity cell after lags          
# te_cor           temperature (C) at thermistor after lags applied 

# Path to datasets folder
datasets_dir = 'goldData'
new_dir = 'gridData'
new_matDir = 'gridDataMat'

error_list=[]

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
        start = np.floor(depth_filtered.min())
        end = np.ceil(depth_filtered.max())
        regular_depths = np.arange(start, end, 0.25)
        check_length = np.arange(depth_filtered.min(),depth_filtered.max(), 0.25)
        
        # check 2 for encoutnering zero
        if len(check_length) < 2:
            error_list.append(f'{full_path}: lack enough points')
            raise ValueError(f"{full_path} does not enough valid points.")

        # add check 3 for interpolation error:
        unique_vals, counts = np.unique(depth_filtered, return_counts=True)
        duplicates = unique_vals[counts > 1]
        if duplicates.size > 0:
            error_list.append(f'{full_path}: replicate value')
            print(f"Duplicates found for file{full_path}")

        # Step 3: Interpolate each variable
        temp_interp = interp1d(depth_filtered, temp_filtered,kind='linear', fill_value="extrapolate")
        sal_interp = interp1d(depth_filtered, sal_filtered,kind='linear', fill_value="extrapolate")
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

mat_error_list = []
def makeMatGrid(full_path, file_name, folder_name):
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

        # check for abnormal depth measurements
        if depth.max() > 800:
            mat_error_list.append(f'{full_path}: abnormal depth')
            raise ValueError(f'Abnormal Depth of {depth.max()}! Please check profile {full_path}')

        # add checks for possible empty depth files
        if (depths_sorted.size == 0) or (temperatures_sorted.size == 0) or (salinity_sorted.size == 0):
             mat_error_list.append(f'{full_path}: empty entries')
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
        # the problem being that some of the "Tmin" is at the Pacific water range, hence:
        
        # select only from Tmin (above 400m?)to 5+Tmax for interpolation:
        # surface_index is all indeces of depth array who's above T_Max and below 100m 
        surface_index = np.where((depths_sorted >= 100) & (depths_sorted <= T_Max_Depth))[0]
        temp_min_idx = np.argmin(temperatures_sorted[surface_index])
        temp_min_depth_idx = surface_index[temp_min_idx]

        filter_mask = np.arange(temp_min_depth_idx, temp_max_depth_idx + 8)
        depth_filtered = depths_sorted[filter_mask]
        temp_filtered = temperatures_sorted[filter_mask]
        sal_filtered = salinity_sorted[filter_mask]

         # Step 2: Create regular depth grid (every 0.25 m)
        start = np.floor(depth_filtered.min())
        end = np.ceil(depth_filtered.max())
        regular_depths = np.arange(start, end, 0.25)
        check_length = np.arange(depth_filtered.min(),depth_filtered.max(), 0.25)
        
        # check for too less number of observations
        if len(check_length) < 18:
            error_list.append(f'{full_path}: lack enough points')
            raise ValueError(f"{full_path} does not enough valid points.")

        # add check 3 for interpolation error:
        unique_vals, counts = np.unique(depth_filtered, return_counts=True)
        duplicates = unique_vals[counts > 1]
        if duplicates.size > 0:
            mat_error_list.append(f'{full_path}: replicate value')
            print(f"Duplicates found for file{full_path}")

        # Step 3: Interpolate each variable
        temp_interp = interp1d(depth_filtered, temp_filtered,kind='linear', fill_value="extrapolate")
        sal_interp = interp1d(depth_filtered, sal_filtered,kind='linear', fill_value="extrapolate")
        interpolated_temperatures = temp_interp(regular_depths)
        interpolated_salinity = sal_interp(regular_depths)

        import warnings

        # Define physical ranges (adjust if needed)
        TEMP_MIN, TEMP_MAX = -4, 4
        SAL_MIN, SAL_MAX = 0, 42

        # Check for NaNs
        if np.isnan(interpolated_temperatures).any():
            nan_indices = np.where(np.isnan(interpolated_temperatures))[0]
            msg = (f"{full_path}: NaNs found in interpolated temperatures at indices {nan_indices}. "
                f"Values: {interpolated_temperatures[nan_indices]}")
            mat_error_list.append(msg)
            raise ValueError(msg)

        if np.isnan(interpolated_salinity).any():
            nan_indices = np.where(np.isnan(interpolated_salinity))[0]
            msg = (f"{full_path}: NaNs found in interpolated salinity at indices {nan_indices}. "
                f"Values: {interpolated_salinity[nan_indices]}")
            mat_error_list.append(msg)
            raise ValueError(msg)

        # Check for out-of-range values
        temp_out_of_range_idx = np.where((interpolated_temperatures < TEMP_MIN) | (interpolated_temperatures > TEMP_MAX))[0]
        if temp_out_of_range_idx.size > 0:
            values = interpolated_temperatures[temp_out_of_range_idx]
            msg = (f"{full_path}: interpolated temperatures out of physical range ({TEMP_MIN} to {TEMP_MAX} °C) "
                f"at indices {temp_out_of_range_idx} with values {values}")
            mat_error_list.append(msg)
            raise ValueError(msg)

        sal_out_of_range_idx = np.where((interpolated_salinity < SAL_MIN) | (interpolated_salinity > SAL_MAX))[0]
        if sal_out_of_range_idx.size > 0:
            values = interpolated_salinity[sal_out_of_range_idx]
            msg = (f"{full_path}: interpolated salinity out of physical range ({SAL_MIN} to {SAL_MAX} PSU) "
                f"at indices {sal_out_of_range_idx} with values {values}")
            mat_error_list.append(msg)
            raise ValueError(msg)

        
        if not (len(regular_depths) == len(interpolated_temperatures) == len(interpolated_salinity)):
              mat_error_list.append(f'{full_path}: mismatch length')
              raise ValueError(f"Length mismatch in interpolated arrays in file: {full_path}")

        mdic = {"Depth":regular_depths, "Temperature":interpolated_temperatures, "Salinity":interpolated_salinity, "lat":lat,
                 "lon":lon, "startDate":psdate, "startTime":pstart, "endDate":pedate, "endTime":pstop}
        
        output_subfolder = os.path.join(new_matDir, folder_name)
        os.makedirs(output_subfolder, exist_ok=True)
        output_filename = file_name
        output_path = os.path.join(output_subfolder, output_filename)
        
        savemat(output_path, mdic)




# traverse_datasets(datasets_dir, makeGrid)
# with open("errorDF.txt", "w") as bad_file:
#     for file in error_list:
#         bad_file.write(f"{file}\n")
traverse_datasets(datasets_dir, makeMatGrid)
with open("mat_error.txt", "w") as bad_file:
    for file in mat_error_list:
        bad_file.write(f"{file}\n")