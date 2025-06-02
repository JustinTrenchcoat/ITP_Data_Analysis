import matplotlib.pyplot as plt
import numpy as np
import h5py
import traceback
from scipy.interpolate import interp1d
from helper import height

# this script looks through each profile extract depth, temperature and every other measurement, interpolate the values by depth grid of 0.25m
# should it look like a dataframe or?

# set up file path
full_path = r'D:\EOAS\ITP_Data_Analysis\datasets\itp112cormat\cor0002.mat'

new_depth =[]
new_temp = []

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

        # Step 1: Sort by depth
        depth = height(pr_filt, lat)
        sorted_indices = np.argsort(depth)
        depths_sorted = depth[sorted_indices]
        temperatures_sorted = te_cor[sorted_indices]
        salinity_sorted = sa_adj[sorted_indices]

         # Step 2: Create regular depth grid (every 0.25 m)
        regular_depths = np.arange(depths_sorted.min(), depths_sorted.max(), 0.25)

        # Step 3: Interpolate temperature onto regular grid
        f_interp = interp1d(depths_sorted, temperatures_sorted, kind='linear', bounds_error=False, fill_value='extrapolate')
        interpolated_temperatures = f_interp(regular_depths)
       
        # Filter depth to only between 600 and 200
        print("test 1")
        in_range_mask = (regular_depths >= 200) & (regular_depths <= 600)
        depth_in_range = regular_depths[in_range_mask]
        temp_in_range = interpolated_temperatures[in_range_mask]


        plt.plot(temp_in_range, depth_in_range, marker='o',linestyle='dashed',linewidth=2, markersize=12)
        plt.xlabel("Temperature")
        plt.ylabel("Depth")
        plt.title(f"Temperature vs depth (interpolated), System#112 Profile#2")
        plt.grid(True)
        plt.gca().invert_yaxis()
        # Optional: Rotate date labels for clarity
        plt.xticks(rotation=45)
        plt.show()


        # Inside your for-loop where the exception occurs:
except Exception as e:
    print(f"Error processing file: {full_path}")
    traceback.print_exc()