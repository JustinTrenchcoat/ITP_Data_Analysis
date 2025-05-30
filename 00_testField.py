import matplotlib.pyplot as plt
import numpy as np
import argparse
import h5py
from helper import *
import traceback

# set up file path
full_path = r'D:\EOAS\ITP_Data_Analysis\datasets\itp41cormat\cor1391.mat'

def smartPlot(plotType, sample, sysNum, profNum):
    if plotType == "tvd":
        depth = Profile.depth(sample)
        temp = Profile.potential_temperature(sample)
        time = Profile.python_datetime(sample)
        time = time.strftime("%Y-%m-%d %H:%M:%S")

        plotHelper(temp, depth, "Temperature", "Depth", sysNum,profNum,time)

        plt.savefig(f"plots/temp_vs_depth_sys_{sysNum}_prof_{profNum}.png")
        plt.show()
    elif plotType == "svd":
        depth = Profile.depth(sample)
        salinity = Profile.absolute_salinity(sample)
        time = Profile.python_datetime(sample)
        time = time.strftime("%Y-%m-%d %H:%M:%S")

        plotHelper(salinity, depth, "Salinity", "Depth", sysNum,profNum,time)
        plt.savefig(f"plots/sal_vs_depth_sys_{sysNum}_prof_{profNum}.png")
        plt.show()
 

def plotHelper(x,y, xlabel, ylabel, sysNum, profNum,time):
    # Filter y between 200 and 500
    # mask = (y >= 200) & (y <= 500)
    # x_filtered = x[mask]
    # y_filtered = y[mask]
    plt.plot(x, y, marker='o',linestyle='dashed',linewidth=2, markersize=5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{xlabel} vs {ylabel}, System# {sysNum} Profile# {profNum}, Time {time}")
    plt.grid(True)
    plt.gca().invert_yaxis()
    # Optional: Rotate date labels for clarity
    plt.xticks(rotation=45)
    plt.show()

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



# try:
#     with h5py.File(full_path, 'r') as f:
#         def read_var(varname):
#             return np.array(f[varname]).reshape(-1)
#         sa_cor = read_var('sa_cor')
#         pr_filt = read_var('pr_filt')
#         lat = read_var("latitude")
#         lon = read_var("longitude")

#         # Filter out NaNs
#         valid_mask = ~np.isnan(sa_cor) & ~np.isnan(pr_filt)
#         # valid_mask = ~np.isnan(pr_filt)
#         sa_cor = sa_cor[valid_mask]
#         pr_filt = pr_filt[valid_mask]

#         depth = height(pr_filt, lat)
#         depth = np.sort(depth)
#         depth_diff = np.diff(depth)
#         statHelper(depth_diff)
        
#         # # Plot the histogram
#         # plt.figure(figsize=(10, 6))
#         # plt.hist(depth_diff, bins='auto', edgecolor='black')  # adjust bin count or bin edges here
#         # plt.title(f"Histogram of Depth Differences in file {full_path}")
#         # plt.xlabel("Depth Difference")
#         # plt.ylabel("Frequency")
#         # plt.grid(True)
#         # plt.tight_layout()
#         # plt.show()
#         plotHelper(sa_cor,depth, "Salinity", "Depth", 112, 1,"made up time")
# except Exception:
#     print("Error Reading Data!")


try:
   with h5py.File(full_path, 'r') as f:
        def read_var(varname):
            return np.array(f[varname]).reshape(-1)

        # read variables from single file for later reference.
        pr_filt = read_var('pr_filt')
        te_cor = read_var('te_cor')
        lat = read_var("latitude")
        lon = read_var("longitude")

        # Filter out NaNs
        # valid_mask = ~np.isnan(sa_cor) & ~np.isnan(pr_filt)
        valid_mask = ~np.isnan(pr_filt)
        # sa_cor = sa_cor[valid_mask]
        pr_filt = pr_filt[valid_mask]
        te_cor  = te_cor[valid_mask]

        # calculate depth
        depth = height(pr_filt, lat)
        dep_max = max(depth)

        # depth_index has all indeces of depth array who's value is above 200m
        depth_index = np.where(depth >= 200)[0]
        # temp_max_val is the max value of all te_cor values within the range of depth_index
        temp_max_val = np.max(te_cor[depth_index])
        # temp_max_idx is the index of max value of all te_cor values within the range of depth_index
        temp_max_idx = np.argmax(te_cor[depth_index])

        temp_max_depth = depth[temp_max_idx]
        if temp_max_idx <0:
            print("test?")

        # depth_sort = np.sort(depth)
        # print(depth_sort[1528])
        # print(depth_sort[1529])
        # depth_diff = np.diff(depth)
        # statHelper(depth_diff)
        # plt.plot(sa_cor, depth, marker='o',linestyle='dashed',linewidth=2, markersize=5)
        # plt.grid(True)
        # plt.gca().invert_yaxis()
        # plt.show()
        # plotHelper(sa_cor,depth, "Salinity", "Depth", 112, 1,"made up time")
except Exception:
    print("Error Reading Data!")
    traceback.print_exc()