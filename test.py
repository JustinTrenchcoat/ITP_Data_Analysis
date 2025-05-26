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

# Folder containing your .mat files
folder_path = 'itp6cormat'

# Store results
good_profile = []
bad_profile = []
missing_profile = []

def decode_ascii(matlab_str):
    return ''.join([chr(c) for c in matlab_str])

# Get all .mat filenames in the folder
all_mat_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.mat')])

# Process each file
for filename in tqdm(all_mat_files, desc="Processing profiles"):
    full_path = os.path.join(folder_path, filename)

    try:
        with h5py.File(full_path, 'r') as f:
            def read_var(varname):
                return np.array(f[varname]).squeeze()

            sa_cor = read_var('sa_cor')
            pr_filt = read_var('pr_filt')

            # Decode string fields
            date = decode_ascii(read_var("psdate"))
            time = decode_ascii(read_var("pstart"))
            lat = read_var("latitude")
            lon = read_var("longitude")

            # Filter out NaNs
            valid_mask = ~np.isnan(sa_cor) & ~np.isnan(pr_filt)
            sa_cor = sa_cor[valid_mask]
            pr_filt = pr_filt[valid_mask]

            depth = height(pr_filt, lat)
            dep_max = max(depth)

            if (dep_max >= 400) and (73 <= lat <= 81) and (-160 <= lon <= -130):
                good_profile.append(filename)
            else:
                bad_profile.append(filename)

    except Exception as e:
        bad_profile.append(filename)

# Delete bad profile files
for filename in bad_profile:
    full_path = os.path.join(folder_path, filename)
    try:
        if os.path.isfile(full_path):
            os.remove(full_path)
            print(f"Deleted: {filename}")
        else:
            print(f"File not found: {filename}")
    except Exception as e:
        print(f"Failed to delete {filename}: {e}")
