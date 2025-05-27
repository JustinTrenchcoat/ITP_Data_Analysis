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

import h5py
import numpy as np
import os
import gsw
from tqdm import tqdm
from helper import *

def decode_ascii(matlab_str):
    return ''.join([chr(c) for c in matlab_str])

# Path to datasets folder
datasets_dir = 'datasets'

# Loop over every itp*cormat folder
for folder_name in sorted(os.listdir(datasets_dir)):
    folder_path = os.path.join(datasets_dir, folder_name)
    if not os.path.isdir(folder_path):
        continue  # skip non-folders

    print(f"\nProcessing folder: {folder_name}")

    good_profile = []
    bad_profile = []

    # Get all .mat files
    all_mat_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.mat')])

    for filename in tqdm(all_mat_files, desc=f"Filtering {folder_name}", leave=False):
        full_path = os.path.join(folder_path, filename)

        try:
            with h5py.File(full_path, 'r') as f:
                def read_var(varname):
                    return np.array(f[varname]).squeeze()

                # sa_cor = read_var('sa_cor')
                pr_filt = read_var('pr_filt')
                date = decode_ascii(read_var("psdate"))
                time = decode_ascii(read_var("pstart"))
                lat = read_var("latitude")
                lon = read_var("longitude")

                # Filter out NaNs
                # valid_mask = ~np.isnan(sa_cor) & ~np.isnan(pr_filt)
                valid_mask = ~np.isnan(pr_filt)
                # sa_cor = sa_cor[valid_mask]
                pr_filt = pr_filt[valid_mask]

                depth = height(pr_filt, lat)
                dep_max = max(depth)

                if (dep_max >= 400) and (73 <= lat <= 81) and (-160 <= lon <= -130):
                    good_profile.append(filename)
                else:
                    bad_profile.append(filename)

        except Exception:
            bad_profile.append(filename)

    # Delete bad profiles
    for filename in bad_profile:
        try:
            os.remove(os.path.join(folder_path, filename))
            print(f"Deleted: {filename}")
        except Exception as e:
            print(f"Failed to delete {filename}: {e}")
