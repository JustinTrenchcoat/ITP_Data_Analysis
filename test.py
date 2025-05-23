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
folder_path = 'itp120cormat'

# Store results
itpnos = []

# Loop over all expected file names
for i in tqdm(range(1, 10)):
    filename = f"cor{i:04d}.mat"  # Formats as cor0001.mat, ..., cor0014.mat
    full_path = os.path.join(folder_path, filename)
    
    if not os.path.isfile(full_path):
        print(f"Missing: {filename}")
        continue

    try:
        with h5py.File(full_path, 'r') as f:
            # Extract itpno value
            itpno = int(np.array(f['itpno']).squeeze())
            itpnos.append((filename, itpno))
            def read_var(varname):
                return np.array(f[varname]).squeeze()

            sa_cor = read_var('sa_cor')
            pr_filt = read_var('pr_filt')

            valid_mask = ~np.isnan(sa_cor) & ~np.isnan(pr_filt)
            sa_cor = sa_cor[valid_mask]
            pr_filt = pr_filt[valid_mask]

                # Decode single string (e.g., one profile)
            date = decode_ascii(read_var("psdate"))
            time = decode_ascii(read_var("pstart"))
            lon = read_var("longitude")
            lat = read_var("latitude")  # Assuming this is a number
    except Exception as e:
        print(f"Error reading {filename}: {e}")

    

    print(absolute_salinity(sa_cor, pr_filt, lon,lat))

# Print summary of each file
for fname, itp in itpnos:
    print(f"{fname}: ITP {itp}")

# Print unique ITP numbers
unique_itps = sorted(set(itp for _, itp in itpnos))
print("\nUnique ITP numbers found:")
print(unique_itps)
