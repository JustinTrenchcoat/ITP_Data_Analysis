import matplotlib.pyplot as plt
import numpy as np
import argparse
import h5py
import traceback
import gsw

# set up file path
full_path = r'D:\EOAS\ITP_Data_Analysis\datasets\itp112cormat\cor0002.mat'

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


try:
    with h5py.File(full_path, 'r') as f:
        def read_var(varname):
            return np.array(f[varname]).reshape(-1)

        pr_filt = read_var('pr_filt')
        lat = read_var("latitude")
        te_cor = read_var("te_cor")
        sa_adj = read_var("sa_adj")

        valid_mask = ~np.isnan(pr_filt)
        pr_filt = pr_filt[valid_mask]
        te_cor = te_cor[valid_mask]
        sa_adj = sa_adj[valid_mask]

        depth = height(pr_filt, lat)

        # Filter depth to only between 600 and 200
        in_range_mask = (depth >= 200) & (depth <= 600)

        depth = depth[in_range_mask]
        print(in_range_mask)
        te_cor = te_cor[in_range_mask]
        plt.plot(te_cor, depth, marker='o',linestyle='dashed',linewidth=2, markersize=5)
        plt.grid(True)
        plt.gca().invert_yaxis()
        plt.show()
        # plotHelper(sa_cor,depth, "Salinity", "Depth", 112, 1,"made up time")
except Exception:
    print("Error Reading Data!")
    traceback.print_exc()