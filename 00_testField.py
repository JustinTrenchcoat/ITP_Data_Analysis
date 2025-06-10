full_path = r'rawData\itp65cormat\cor0813.mat'
import pandas as pd
from scipy.io import savemat
import re
import os
from helper import *

# def count(full_path, file_name, folder_name):
#     return 1

grid_dir = "gridData"
countData(grid_dir)
# traverse_datasets(grid_dir, count)
# folder_path = r'D:\EOAS\ITP_Data_Analysis\gridDataMat\itp100cormat'

# mat_files = [f for f in os.listdir(folder_path) if f.endswith(".mat")]
# profile_count = len(mat_files)

# if profile_count == 0:
#      print("test 1")
# else:
#     print(f"{profile_count} profiles")


# bad_profile = []
# filename = "test"

# try:
#     with h5py.File(full_path, 'r') as f:
#         # read variables from single file for later reference.
#         pr_filt = read_var(f, 'pr_filt')
#         te_adj = read_var(f, 'te_adj')
#         sa_adj = read_var(f, "sa_adj")
#         lat = read_var(f, "latitude")
#         lon = read_var(f, "longitude")

#         # Filter out NaNs
#         valid_mask = ~np.isnan(te_adj) & ~np.isnan(pr_filt) & ~np.isnan(sa_adj)
#         pr_filt = pr_filt[valid_mask]
#         te_adj  = te_adj[valid_mask]
                

#         # check for empty values
#         if (pr_filt.size == 0) or (te_adj.size == 0):
#             bad_profile.append(filename)


#         # calculate depth
#         depth = height(pr_filt, lat)
#         dep_max = np.max(depth)
#         # filter out profiles that only gets to 200m max.
#         if dep_max <= 200:
#             bad_profile.append(filename)

#         # criteria for gold standard dataset:
#         # 1. Collected in Beaufort Gyre
#         # 2. The T_Max occured 300m below the sea surface
#         # 3. There are measurements below the T_Max for at least 10m

#         # Crit.1:
#         is_Beaufort = (73 <= lat <= 81) and (-160 <= lon <= -130)

#         # Crit 2:
#         # depth_index has all indeces of depth array who's below 200m
#         depth_index = np.where(depth >= 200)[0]
#         # temp_idx_max is the index of max value of all te_adj values measured below 200 m
#         depth_index_temp_max_idx = np.argmax(te_adj[depth_index])
#         # temp_max_depth_idx is the index value of depth, for the depth with max temperature below 200 m
#         temp_max_depth_idx = depth_index[depth_index_temp_max_idx]
#         # temp_max_depth is the value of depth at the max value of te_cor values that are under 200m
#         temp_max_depth = depth[temp_max_depth_idx]
#         max_deep = (temp_max_depth >= 250)

#         # Crit 3:
#         # the greatest depth measurement has to be 10 m deeper than temp_max_depth
#         deep_enough = (dep_max >= temp_max_depth+10)
#         helPlot(te_adj,depth)


#         # if the depth of temp_max beyond 200m, then it is good profile:
#         if (deep_enough and is_Beaufort and max_deep):
#             # --- COPY GOOD FILE TO goldData ---
#             print("good Profile")

#         else:
#             print("bad_profile")

# except Exception:
#     bad_profile.append(filename)
