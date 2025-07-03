import pandas as pd
from helper import *
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import gsw
import datetime
import re
import seaborn as sns
from matplotlib.colors import SymLogNorm
import matplotlib.colors as colors
import matplotlib.dates as mdates
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter1d
import math
'''
In the dataset:
every row is one observation from one profile.
It has to be ordered in time so that time series analysis would work.
'''
# Read in the data:
def readNC(full_path, ls, itp_num):
    ds = nc.Dataset(full_path)
    with ds as dataset:
        # extract variables:
        # the prof variable in nc file is not profile number, but index. 
        # FloatID is the true profile number from each ITP system
        profN = dataset.variables['FloatID'][:]
        # the nc file mistakenly wrote pressure instead of depth
        depth = dataset.variables['pressure'][:]
        temp = dataset.variables["ct"][:]
        salinity = dataset.variables["sa"][:]
        connect_layer_mask = dataset.variables['mask_cl'][:]
        interface_layer_mask = dataset.variables['mask_int'][:]
        mixed_layer_mask = dataset.variables["mask_ml"][:]
        staircase_mask = dataset.variables["mask_sc"][:]
        dates = dataset.variables["dates"][:]
        lon = dataset.variables["lon"][:]
        lat = dataset.variables["lat"][:]
        date = pd.to_datetime(dates, unit = 's')
        date = date.date
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
            #############################################################################
            # background infos, do it here because we calculate it per-profile:
            # N Sqaured:
            # Apply centered rolling window smoothing (you can adjust window size)
            temp_smooth = gaussian_filter1d(temp[i], sigma=80, mode='nearest')
            salt_smooth = gaussian_filter1d(salinity[i], sigma=80, mode='nearest')
            pres_smooth = gaussian_filter1d(new_df['pressure'], sigma=80, mode='nearest')
            depth_smooth = gaussian_filter1d(new_df['depth'], sigma=80, mode='nearest')
            # add new cols:
            new_df['dT/dZ'] = np.gradient(temp_smooth, depth_smooth)
            new_df['dS/dZ'] = np.gradient(salt_smooth, depth_smooth)
            n_sq = gsw.Nsquared(salt_smooth, temp_smooth, pres_smooth, lat[i])[0]
            # padding for last value as the function returns only N-1 values
            n_sq_padded = np.append(n_sq, np.nan)
            new_df['n_sq'] = n_sq_padded
            # turner angle and R_rho
            [turner_angle, R_rho,p_mid] = gsw.Turner_Rsubrho(salt_smooth, temp_smooth, pres_smooth)
            new_df['turner_angle'] = np.append(turner_angle,np.nan)
            new_df['R_rho'] = np.append(R_rho,np.nan)
            ####################
            ls.append(new_df)
    return ls
#######################################################
#         Read Data and save, Only run once           #
#######################################################
tagData_dir = 'prod_files'
df_list = []
for fileName in tqdm(sorted(os.listdir(tagData_dir)), desc="Processing files"):
    match = re.search(r'itp(\d+)cormat\.nc', fileName)
    if match:
            itp_num = int(match.group(1))
            full_path = os.path.join(tagData_dir, fileName)
            df_list = readNC(full_path, df_list, itp_num)
final_df = pd.concat(df_list, ignore_index=True)
final_df.to_pickle("final.pkl")
#######################################################
#                                                     #
#######################################################