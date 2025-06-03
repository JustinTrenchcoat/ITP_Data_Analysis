import matplotlib.pyplot as plt
import numpy as np
import h5py
import traceback
from scipy.interpolate import interp1d
from helper import height

# this script looks through each profile extract depth, temperature and every other measurement, interpolate the values by depth grid of 0.25m
# should it look like a dataframe or?

# set up file path
# full_path = r'D:\EOAS\ITP_Data_Analysis\datasets\itp62cormat\cor0008.mat'
full_path = r'D:\EOAS\ITP_Data_Analysis\datasets\itp1cormat\cor0002.mat'

def read_var(f, varname):
        data = np.array(f[varname])
        if data.dtype == "uint16":
            return data.tobytes().decode('utf-16-le')
        return data.reshape(-1)
        

try:
    with h5py.File(full_path, 'r') as f:
        # Print all root level object names (aka keys) 
        # these can be group or dataset names 
        print("Keys: %s" % f.keys()) 

        pr_filt = read_var(f, 'pr_filt')
        sa_adj = read_var(f, "sa_adj")
        te_adj = read_var(f, 'te_adj')
        lat = read_var(f, "latitude")
        lon = read_var(f, "longitude")

        # Filter out NaNs
        valid_mask = ~np.isnan(sa_adj) & ~np.isnan(pr_filt) & ~np.isnan(te_adj)
        # sa_cor = sa_cor[valid_mask]
        pr_filt = pr_filt[valid_mask]
        te_adj  = te_adj[valid_mask]



        print(np.max(te_adj))

        # calculate depth
        depth = height(pr_filt, lat)
        dep_max = np.max(depth)
        print(dep_max)

        # depth_index has all indeces of depth array who's value is above 200m
        depth_index = np.where(depth >= 200)[0]

        # temp_max_idx is the index of max value of all te_cor values within the range of depth_index
        temp_idx_max = np.argmax(te_adj[depth_index])
        # temp_max_depth is the value of depth at the max value of te_cor values that are under 200m
        temp_max_depth_idx = depth_index[temp_idx_max]
        print(f"Max temperature below 5 m is at {depth[temp_max_depth_idx]}m, is {te_adj[temp_max_depth_idx]}")

        temp_max_depth = depth[temp_max_depth_idx]

        # if the depth of temp_max beyond 200m, then it is good profile:

        if ((dep_max >= (temp_max_depth+2)) and (73 <= lat <= 81) and (-160 <= lon <= -130)):
            print("test1")
        else:
            print("test2")


        print(f"End Time: {read_var(f, "pedate")}, {read_var(f, "pstop")}")
        print(f"Start Time: {read_var(f, "psdate")}, {read_var(f, "pstart")}")


except Exception as e:
    print(f"Error processing file: {full_path}")
    traceback.print_exc()