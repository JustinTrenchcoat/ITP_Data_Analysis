# The data in these files are reported at the same temporal resolution as the Level 1 files, 
# with NaNs filling gaps where bad data were removed. 
# The variables included in the cor####.mat files are:
#                                    
# co_adj          conductivity (mmho) after lags and calibration adjustment          
# co_cor          conductivity (mmho) after lags applied          
# itpno            ITP number          
# latitude        start latitude (N+) of profile          
# longitude     start longitude (E+) of profile          
# pr_filt           low pass filtered pressure (dbar)
#           
# psdate           profile UTC start date (mm/dd/yy)          
# pstart            profile UTC start time (hh:mm:ss)  
# pedate          profile UTC end date (mm/dd/yy)           
# pstop            profile UTC stop time (hh:mm:ss)     
#      
# sa_adj          salinity after lags and calibration adjustment          
# sa_cor          salinity after lags applied          
# te_adj           temperature (C) in conductivity cell after lags          
# te_cor           temperature (C) at thermistor after lags applied 

'''
This script will go through the rawData folder created by 01_pullData.py and extract data that satisfies our requirement. 
In this case, the requirement would be:
1. The data is collected in the Beaufort Gyre (Start latitude between 73N to 81N, start longitude between 160 W to 130W)
2. The data has measurement of depth that is at least 2 meters deeper than the AW Temperature Maximum(200m below water)

The new dataset would be stored in the goldData folder, organized by itp number.

for each profile, there will be NaN values for the salinity and other measurements, made to replace the bad values.
This script will not delete it, but hoping that later scripts are able to omit the NaN values

Similarly, some profiles would contain NaN as end dates and times.
'''
import h5py
import numpy as np
import os
import shutil
from tqdm import tqdm
from helper import *

# Path to datasets folder
datasets_dir = 'rawData'
golden_dir = 'goldData'

bad_profile = []

# # Loop over every itp*cormat folder
# for folder_name in sorted(os.listdir(datasets_dir)):
#     folder_path = os.path.join(datasets_dir, folder_name)

#     print(f"\nProcessing folder: {folder_name}")

#     # Get all .mat files
#     all_mat_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.mat')])

#     for filename in tqdm(all_mat_files, desc=f"Filtering {folder_name}", leave=False):
#         full_path = os.path.join(folder_path, filename)

#         try:
#             with h5py.File(full_path, 'r') as f:
#                 # read variables from single file for later reference.
#                 pr_filt = read_var(f, 'pr_filt')
#                 te_adj = read_var(f, 'te_adj')
#                 sa_adj = read_var(f, "sa_adj")
#                 lat = read_var(f, "latitude")
#                 lon = read_var(f, "longitude")

#                 # Filter out NaNs
#                 valid_mask = ~np.isnan(te_adj) & ~np.isnan(pr_filt) & ~np.isnan(sa_adj)
#                 pr_filt = pr_filt[valid_mask]
#                 te_adj  = te_adj[valid_mask]
                

#                 # check for empty values
#                 if (pr_filt.size == 0) or (te_adj.size == 0):
#                     bad_profile.append(filename)
#                     continue

#                 # calculate depth
#                 depth = height(pr_filt, lat)
#                 dep_max = np.max(depth)
#                 # filter out profiles that only gets to 200m max.
#                 if dep_max <= 200:
#                     bad_profile.append(filename)
#                     continue

#                 # criteria for gold standard dataset:
#                 # 1. Collected in Beaufort Gyre
#                 # 2. The T_Max occured 300m below the sea surface
#                 # 3. There are measurements below the T_Max for at least 10m

#                 # Crit.1:
#                 is_Beaufort = (73 <= lat <= 81) and (-160 <= lon <= -130)

#                 # Crit 2:
#                 # depth_index has all indeces of depth array who's below 200m
#                 depth_index = np.where(depth >= 200)[0]
#                 # temp_idx_max is the index of max value of all te_adj values measured below 200 m
#                 depth_index_temp_max_idx = np.argmax(te_adj[depth_index])
#                 # temp_max_depth_idx is the index value of depth, for the depth with max temperature below 200 m
#                 temp_max_depth_idx = depth_index[depth_index_temp_max_idx]
#                 # temp_max_depth is the value of depth at the max value of te_cor values that are under 250m, 
#                 # since pacific water can't be deeper than that.
#                 temp_max_depth = depth[temp_max_depth_idx]
#                 max_deep = (temp_max_depth >= 250)

#                 # Crit 3:
#                 # the greatest depth measurement has to be 10 m deeper than temp_max_depth
#                 deep_enough = (dep_max >= temp_max_depth+10)

#                 # if it fulfills all three criteria, then it is good profile:
#                 if (deep_enough and is_Beaufort and max_deep):
#                     # COPY GOOD FILE TO goldData
#                     dest_folder = os.path.join(golden_dir, folder_name)
#                     os.makedirs(dest_folder, exist_ok=True)
#                     dest_path = os.path.join(dest_folder, filename)
#                     shutil.copy2(full_path, dest_path)
#                 else:
#                     bad_profile.append(filename)

#         except Exception:
#             bad_profile.append(filename)

# quick check for bad profiles
# print(f"there are in total {len(bad_profile)} bad profiles")
# sanity checks for those gold-standard data
# checkField(golden_dir)
countData(golden_dir)