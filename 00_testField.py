from minisom import MiniSom
import numpy as np
import matplotlib.pyplot as plt
import h5py
import traceback
from sklearn.preprocessing import MinMaxScaler
from helper import *
from scipy.interpolate import interp1d
import os
import pandas as pd
import re

full_path = r'D:\EOAS\ITP_Data_Analysis\goldData\itp1cormat\cor0001.mat'
new_dir = 'gridData'
folder_name = 'testFolder'
filename = 'testFile'

try:
            with h5py.File(full_path, 'r') as f:

                pr_filt = read_var(f,'pr_filt')
                lat = read_var(f,"latitude")
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

                # Step 2: Create regular depth grid (every 0.25 m)
                regular_depths = np.arange(depths_sorted.min(), depths_sorted.max(), 0.25)

                # Step 3: Interpolate each variable
                temp_interp = interp1d(depths_sorted, temperatures_sorted, kind='linear', bounds_error=False, fill_value='extrapolate')
                sal_interp = interp1d(depths_sorted, salinity_sorted, kind='linear', bounds_error=False, fill_value='extrapolate')
                interpolated_temperatures = temp_interp(regular_depths)
                interpolated_salinity = sal_interp(regular_depths)
                
                if not (len(regular_depths) == len(interpolated_temperatures) == len(interpolated_salinity)):
                    raise ValueError(f"Length mismatch in interpolated arrays in file: {full_path}")

                # Create DataFrame
                df = pd.DataFrame({
                    'Depth': regular_depths,
                    'Temperature': interpolated_temperatures,
                    'Salinity': interpolated_salinity
                })

                # Create matching subfolder in gridData
                output_subfolder = os.path.join(new_dir, folder_name)
                os.makedirs(output_subfolder, exist_ok=True)

                def clean(x):
                    # Convert to string
                    x_str = str(x)
                    # Remove brackets, whitespace
                    x_str = x_str.strip("[] ").replace(" ", "")
                    # Replace any non-alphanumeric or underscore/dash/dot characters with underscore
                    x_str = re.sub(r'[^\w\-.]', '_', x_str)
                    return x_str

                lon_str = clean(lon)
                lat_str = clean(lat)
                psdate_str = clean(psdate)
                pstart_str = clean(pstart)
                pedate_str = clean(pedate)
                pstop_str = clean(pstop)

                # Output path
                output_filename = f"{filename.rstrip('.mat')}_{lon_str}_{lat_str}_{psdate_str}_{pstart_str}_{pedate_str}_{pstop_str}.csv"
                output_path = os.path.join(output_subfolder, output_filename)

                # Save to CSV
                df.to_csv(output_path, index=False)

                
except Exception as e:
            print(f"Error processing file: {full_path}")
            traceback.print_exc()
