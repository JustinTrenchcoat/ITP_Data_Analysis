import matplotlib.pyplot as plt
import numpy as np
import argparse
import h5py
import traceback
import gsw
from scipy.interpolate import interp1d


# set up file path
full_path = r'D:\EOAS\ITP_Data_Analysis\datasets\itp69cormat\cor0362.mat'

try:
    with h5py.File(full_path, 'r') as f:
        # List all variable names (keys) in the file
        print(f"Variables in {full_path}:")
        for key in f.keys():
            print(f" - {key}")
        
        # Now you can read variables like f['pr_filt'], etc.
except Exception as e:
    print(f"Error processing file: {full_path}")
    traceback.print_exc()


def statHelper(all_depth_differences):
    print("Statistical Summary of Depth Differences:")
    print(f"Count         : {len(all_depth_differences)}")
    print(f"Min           : {np.min(all_depth_differences)}")
    print(f"Max           : {np.max(all_depth_differences)}")
    print(f"Mean          : {np.mean(all_depth_differences)}")
    print(f"Median        : {np.median(all_depth_differences)}")
    print(f"Std Dev       : {np.std(all_depth_differences)}")
    print(f"Variance      : {np.var(all_depth_differences)}")
    print(f"25th Percentile (Q1): {np.percentile(all_depth_differences, 25)}")
    print(f"75th Percentile (Q3): {np.percentile(all_depth_differences, 75)}")
    print(f"IQR           : {np.percentile(all_depth_differences, 75) - np.percentile(all_depth_differences, 25)}")
    print(f"the index of max diff is at: {np.argmax(all_depth_differences)}")

def height(pressure, latitude):
        return -gsw.conversions.z_from_p(pressure, latitude)

# try:
#     with h5py.File(full_path, 'r') as f:
#         def read_var(varname):
#             return np.array(f[varname]).reshape(-1)

#         pr_filt = read_var('pr_filt')
#         sa_cor = read_var('sa_cor')
#         lat = read_var("latitude")

#         valid_mask = ~np.isnan(pr_filt)
#         invalid_mask = np.isnan(pr_filt)
#         for val in invalid_mask:
#             if val:
#                 print("Has NaN value!")
#         pr_filt = pr_filt[valid_mask]

#         depth = height(pr_filt, lat)

#         # Filter depth to only between 600 and 200
#         in_range_mask = (depth >= 200) & (depth <= 600)

#         depth = depth[in_range_mask]
#         print(in_range_mask)
#         sa_cor = sa_cor[in_range_mask]

#         depth_sort = np.sort(depth)
#         print(depth_sort[-2])
#         print(depth_sort[-1])
#         depth_diff = np.diff(depth)
#         statHelper(depth_diff)
#         plt.plot(sa_cor, depth, marker='o',linestyle='dashed',linewidth=2, markersize=5)
#         plt.grid(True)
#         plt.gca().invert_yaxis()
#         plt.show()
#         # plotHelper(sa_cor,depth, "Salinity", "Depth", 112, 1,"made up time")
# except Exception:
#     print("Error Reading Data!")
#     traceback.print_exc()


# try:
#     with h5py.File(full_path, 'r') as f:
#         def read_var(varname):
#             data = np.array(f[varname])
#             if data.dtype == 'uint16': 
#                 return data.tobytes().decode('utf-16-le')
#             return data.reshape(-1)

#         pr_filt = read_var('pr_filt')
#         lat = read_var("latitude")
#         lon = read_var("longitude")
#         psdate = read_var("psdate")
#         pstart = read_var("pstart")
#         pedate = read_var("pedate")
#         pstop = read_var("pstop")

#         te_cor = read_var("te_cor")
#         sa_cor = read_var("sa_cor")

#         valid_mask = ~np.isnan(pr_filt)
#         pr_filt = pr_filt[valid_mask]
#         te_cor = te_cor[valid_mask]
#         sa_adj = sa_cor[valid_mask]

#         # Step 1: Sort by depth
#         depth = height(pr_filt, lat)
#         sorted_indices = np.argsort(depth)
#         depths_sorted = depth[sorted_indices]
#         temperatures_sorted = te_cor[sorted_indices]
#         salinity_sorted = sa_adj[sorted_indices]

#         # Step 2: Create regular depth grid (every 0.25 m)
#         regular_depths = np.arange(depths_sorted.min(), depths_sorted.max(), 0.25)
#         print(len(regular_depths))

#         # Step 3: Interpolate each variable
#         temp_interp = interp1d(depths_sorted, temperatures_sorted, kind='linear', bounds_error=False, fill_value='extrapolate')
#         sal_interp = interp1d(depths_sorted, salinity_sorted, kind='linear', bounds_error=False, fill_value='extrapolate')
#         interpolated_temperatures = temp_interp(regular_depths)
#         print(len(interpolated_temperatures))
#         interpolated_salinity = sal_interp(regular_depths)
#         print(len(interpolated_salinity))
#         plt.plot(interpolated_temperatures, regular_depths, marker='o',linestyle='dashed',linewidth=2, markersize=5)
#         plt.grid(True)
#         plt.gca().invert_yaxis()
#         plt.show()
#         # plotHelper(sa_cor,depth, "Salinity", "Depth", 112, 1,"made up time")
# except Exception:
#     print("Error Reading Data!")
#     traceback.print_exc()