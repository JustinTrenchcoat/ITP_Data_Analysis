import h5py
import numpy as np
import matplotlib.pyplot as plt
from helper import *

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

# Load the .mat file
file_path = 'itp120cormat/cor0073.mat'

# Decode ASCII arrays to strings
def decode_ascii(matlab_str):
    return ''.join([chr(c) for c in matlab_str])

# After reading the file:
with h5py.File(file_path, 'r') as f:
    def read_var(varname):
        return np.array(f[varname]).squeeze()

    sa_cor = read_var('sa_cor')
    pr_filt = read_var('pr_filt')

    # Decode single string (e.g., one profile)
    date = decode_ascii(read_var("psdate"))
    time = decode_ascii(read_var("pstart"))
    lat = read_var("latitude")  # Assuming this is a number

    # Optional: filter out NaNs or bad values
    valid_mask = ~np.isnan(sa_cor) & ~np.isnan(pr_filt)
    sa_cor = sa_cor[valid_mask]
    pr_filt = pr_filt[valid_mask]
    depth = height(pr_filt,lat)

# Plot
print(max(depth))
