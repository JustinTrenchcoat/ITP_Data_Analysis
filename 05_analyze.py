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

def singleRead(full_path, ls, profile_num):
    data = loadmat(full_path)
    
    # Adjust according to actual variable names in your .mat file
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

    # extra lines for debugging#####################
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
    n_sq_padded = np.append(n_sq, np.nan)
    new_df['n_sq'] = n_sq_padded

    [turner_angle, R_rho, _] = gsw.Turner_Rsubrho(salinity, temp, pres)
    turner_angle = gaussian_filter1d(turner_angle, sigma=80, mode="nearest")
    R_rho = 1/(R_rho)
    R_rho = gaussian_filter1d(R_rho, sigma=80, mode="nearest")
    new_df['turner_angle'] = np.append(turner_angle, np.nan)
    new_df['R_rho'] = np.append(R_rho, np.nan)

    # sanity check
    assert len(new_df['turner_angle']) == len(depth), f"Wrong dimension"

    # Add profile number column here
    new_df['profileNum'] = profile_num

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
                    profile_num = None  # Or raise an error if mandatory

                # Pass profile number to singleRead
                df_list = singleRead(full_path, df_list, profile_num)

                for warning in w:
                    if (issubclass(warning.category, RuntimeWarning) and
                        "invalid value encountered in ct_from_t" in str(warning.message)):
                        print(f"RuntimeWarning in file: {file_name}")

        except Exception as e:
            print(f"Error processing file: {file_name}")
            traceback.print_exc()

final_df = pd.concat(df_list, ignore_index=True)
final_df.to_pickle("test.pkl")
#######################################################
