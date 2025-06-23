import matplotlib.pyplot as plt
import numpy as np
import argparse
import h5py
import pickle
from helper import *

# set up file path
full_path = r'D:\EOAS\ITP_Data_Analysis\datasets\itp112cormat\cor0001.mat'




def plotHelper(x,y, xlabel, ylabel, sysNum, profNum,time):
    # Filter y between 200 and 500
    mask = (y >= 200) & (y <= 500)
    x_filtered = x[mask]
    y_filtered = y[mask]
    plt.plot(x_filtered, y_filtered, marker='o',linestyle='dashed',linewidth=2, markersize=12)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{xlabel} vs {ylabel}, System# {sysNum} Profile# {profNum}, Time {time}")
    plt.grid(True)
    plt.gca().invert_yaxis()
    # Optional: Rotate date labels for clarity
    plt.xticks(rotation=45)
    plt.show()

def histogramFrequency():
    # Load the depth differences
    with open("depth_differences.pkl", "rb") as f:
        all_depth_differences = pickle.load(f)

    # Convert to NumPy array in case it's a list
    all_depth_differences = np.array(all_depth_differences)

    # Basic statistics
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



    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(all_depth_differences, bins=100, range=(0,0.6), edgecolor='black')  # adjust bin count or bin edges here
    plt.title("Histogram of Depth Differences of All profiles in Beaufort Gyre")
    plt.xlabel("Depth Difference")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# histogramFrequency()
