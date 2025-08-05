import os
import re
import warnings
import traceback
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.io import loadmat
import gsw
from tqdm import tqdm
from helper import *
from scipy.interpolate import interp1d

# reads .mat files in the gridDataMat folder, calculate background properties,
# and save them into a .pkl file for further analysis
def singleRead(full_path, ls, profile_num,sys_num):
    data = loadmat(full_path)
    
    # subject to changes in variable names in .mat file
    depth = data['Depth'].squeeze()
    lat = data['lat'].squeeze()
    lon = data['lon'].squeeze()
    psdate = data['startDate'].squeeze()
    date = pd.to_datetime(psdate, format='%m/%d/%y').date()
    temp = data['Temperature'].squeeze()
    salinity = data['Salinity'].squeeze()
    pres = pressure(depth, lat)

    assert not np.isnan(depth).any(), "depth contains NaN values"
    assert not np.isnan(temp).any(), "temp contains NaN values"
    assert not np.isnan(salinity).any(), "salinity contains NaN values"
    assert not np.isnan(pres).any(), "pres contains NaN values"

    # convert salinity and temperature to absolute salinity and conservative temperature
    salinity = gsw.SA_from_SP(salinity, pres, lon, lat)
    temp = gsw.CT_from_t(salinity, temp, pres)
    ###############################################################
    new_df = pd.DataFrame({
            "depth" : depth,
            'temp' : temp,
            'date' : date,
            "salinity" : salinity,
            "lon" : lon,
            "lat" : lat,
            "pressure" : pres
        })

    # add new cols:
    new_df['dT/dZ'] = gaussian_filter1d(np.gradient(temp, depth),sigma=80, mode='nearest')
    new_df['dS/dZ'] = gaussian_filter1d(np.gradient(salinity, depth),sigma=80, mode='nearest')

    n_sq = gaussian_filter1d(gsw.Nsquared(salinity, temp, pres, lat)[0], sigma=80,mode="nearest")

    [turner_angle, R_rho, p_mid] = gsw.Turner_Rsubrho(salinity, temp, pres)
    depth_mid = height(p_mid, lat)
    # R_rho in gsw routine is temp over sal, but what we need is sal over temp:
    R_rho = np.reciprocal(R_rho)
    # filter out R_rho that is with in(0,100)
    R_rho = np.where((R_rho > 0) & (R_rho < 100), R_rho, 0)

    # make grid given p_mid, then re-fit it into depth grid.
    R_rho_interp = interp1d(depth_mid, R_rho,kind='linear', fill_value="extrapolate")
    turner_angle_interp = interp1d(depth_mid, turner_angle,kind='linear', fill_value="extrapolate")
    n_sq_interp = interp1d(depth_mid, n_sq,kind='linear', fill_value="extrapolate")

    interpolated_R_rho = R_rho_interp(depth)
    interpolated_n_sq = n_sq_interp(depth)
    interpolated_turner = turner_angle_interp(depth)

    # smooth the noisy dataset, sigma set to 80 to ensure that the smoothing "window" is about 20m, subject to change
    turner_angle_smoothed = gaussian_filter1d(interpolated_turner, sigma=80, mode="nearest")
    R_rho_smoothed = gaussian_filter1d(interpolated_R_rho, sigma=80, mode="nearest")
    n_sq_smoothed = gaussian_filter1d(interpolated_n_sq, sigma=80, mode="nearest")
    # fit the background properties to columns:


    new_df['n_sq'] = n_sq_smoothed
    new_df['turner_angle'] = turner_angle_smoothed
    new_df['R_rho'] = R_rho_smoothed

    # sanity check
    assert len(new_df['turner_angle']) == len(depth), f"Wrong dimension"

    # Add profile number column here
    new_df['profileNum'] = profile_num
    new_df["systemNum"] = sys_num

    ls.append(new_df)
    return ls

#######################################################
#         Read Data and Save, Only Run Once           #
#######################################################

datasets_dir = 'gridDataMat'
df_list = []

for folder_name in sorted(os.listdir(datasets_dir)):
    folder_path = os.path.join(datasets_dir, folder_name)
    if not os.path.isdir(folder_path):
        continue  # skip non-folders

    print(f"\nProcessing folder: {folder_name}")

    # Extract profile number from filename
    match = re.search(r'(\d+)', folder_name)
    if match:
        system_num = int(match.group(1))
    else:
        system_num = None  # Or raise an error if mandatory

    # Get all .mat files
    all_mat_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.mat')])

    for file_name in tqdm(all_mat_files, desc=f"Filtering {folder_name}", leave=False):
        full_path = os.path.join(folder_path, file_name)

        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")

                # Extract profile number from filename
                match = re.search(r'(\d+)', file_name)
                if match:
                    profile_num = int(match.group(1))
                else:
                    profile_num = None 

                # Pass profile number to singleRead
                df_list = singleRead(full_path, df_list, profile_num,system_num)

                for warning in w:
                    if (issubclass(warning.category, RuntimeWarning) and
                        "invalid value encountered in ct_from_t" in str(warning.message)):
                        print(f"RuntimeWarning in file: {file_name}")
                    elif (issubclass(warning.category, RuntimeWarning) and
                        "invalid value encountered in sa_from_sp" in str(warning.message)):
                        print(f"RuntimeWarning in file: {file_name}")

        except Exception as e:
            print(f"Error processing file: {file_name}")
            traceback.print_exc()
            
# concat all dataframe to save disk space
final_df = pd.concat(df_list, ignore_index=True)
final_df.to_pickle("test.pkl")
