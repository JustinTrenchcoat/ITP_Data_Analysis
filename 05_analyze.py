import pickle
import numpy as np
import h5py
import os
import traceback
from tqdm import tqdm
from helper import *

import pandas as pd
import matplotlib.pyplot as plt
import gsw
import datetime
import seaborn as sns
from matplotlib.colors import SymLogNorm

# Path to datasets folder
datasets_dir = 'goldData'

all_diff = []
max_all = []
weird_list=[]
weird_max=[]
weird_min = []

def analyze_depth(full_path, file_name, folder_name):
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

# traverse_datasets(datasets_dir, analyze_depth)
# # Convert to numpy array and save to pickle
# all_diff = np.array(all_diff)

# import pickle

# with open("depth_differences.pkl", "wb") as f:
#     pickle.dump(all_diff, f)

# print("Saved depth differences to 'depth_differences.pkl'")
# print("Max of Max depth difference in range:", max(max_all))
# print("List of files with abnormal depth difference")
# for i in range(len(weird_list)):
#     print(f"File {weird_list[i]} has max value of {weird_max[i]}, min value of {weird_min[i]}")

# 09 analysis improvisation:
# this will read from gridDataMat folder, and 
ocean_df = pd.DataFrame()
def readNC(full_path, file_name, folder_name):
    with h5py.File(full_path, 'r') as f:
        depth = read_var(f, 'Depth')
        temp = read_var(f, "Temperature")
        salinity = read_var(f, "Salinity")
        lat = read_var(f, "lat")
        lon = read_var(f, "lon")
        psdate = read_var(f, "startDate")
        new_df = pd.DataFrame({
            "profNum": file_name,
            "depth" : depth,
            "temp" : temp,
            'lat' : lat,
            'lon': lon,
            'psdate':psdate
        })



  
        for i in range(len(profN)):
            mask_cl = connect_layer_mask[i]
            mask_int = interface_layer_mask[i]
            mask_ml = mixed_layer_mask[i]
            mask_sc = staircase_mask[i]
            new_df = pd.DataFrame({
                "profileNumber" : profN[i],
                "depth" : depth[i],
                'temp' : temp[i],
                'date' : date[i],
                "salinity" : salinity[i],
                'mask_cl' : mask_cl,
                'mask_int' : mask_int,
                'mask_ml' : mask_ml,
                "mask_sc" : mask_sc,
                "lon" : lon[i],
                "lat" : lat[i]
            })
            new_df['pressure'] = pressure(depth[i],lat[i])
            new_df['itpNum'] = itp_num

            # background infos, do it here because we calculate it per-profile:
            # N Sqaured:
            n_sq = gsw.Nsquared(salinity[i], temp[i], new_df['pressure'], lat[i])[0]
            # padding for last value as the function returns only N-1 values
            n_sq_padded = np.append(n_sq, np.nan)
            new_df['n_sq'] = n_sq_padded
            # turner angle and R_rho
            [turner_angle, R_rho,p_mid] = gsw.Turner_Rsubrho(salinity[i], temp[i], new_df['pressure'])
            new_df['turner_angle'] = np.append(turner_angle,np.nan)
            new_df['R_rho'] = np.append(R_rho,np.nan)
            ####################
            ls.append(new_df)
    return ls

#######################################################
# Read Data and save, Only run once                   #
#######################################################
# tagData_dir = 'tagData'
# df_list = []
# for fileName in tqdm(sorted(os.listdir(tagData_dir)), desc="Processing files"):
#     match = re.search(r'itp(\d+)cormat\.nc', fileName)
#     if match:
#             itp_num = int(match.group(1))
#             full_path = os.path.join(tagData_dir, fileName)
#             df_list = readNC(full_path, df_list, itp_num)
#             final_df = pd.concat(df_list, ignore_index=True)
#             final_df.to_pickle("final.pkl")
#######################################################
#                                                     #
#######################################################

