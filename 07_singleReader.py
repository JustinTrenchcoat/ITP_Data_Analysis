import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
import traceback
from scipy.interpolate import CubicSpline
from helper import *
import re
from scipy.io import savemat

# this script looks through each profile extract depth, temperature and every other measurement, interpolate the values by depth grid of 0.25m
# should it look like a dataframe or?

# set up file path
# full_path = r'D:\EOAS\ITP_Data_Analysis\datasets\itp62cormat\cor0008.mat'
full_path = r'D:\EOAS\ITP_Data_Analysis\goldData\itp100cormat\cor0001.mat'

new_dir = "testData"
folder_name = "test"



def clean(x):
    # Convert to string
    x_str = str(x)
    # Remove brackets, whitespace
    x_str = x_str.strip("[] ").replace(" ", "")
    # Replace any non-alphanumeric or underscore/dash/dot characters with underscore
    x_str = re.sub(r'[^\w\-.]', '_', x_str)
    return x_str

def plot(x, y):
    plt.plot(x, y, marker='o',linestyle='dashed',linewidth=2, markersize=12)
    plt.xlabel("test x")
    plt.ylabel("test y")
    plt.title("test Plot")
    plt.grid(True)
    plt.gca().invert_yaxis()
    # Optional: Rotate date labels for clarity
    plt.xticks(rotation=45)
    plt.show()




try:
    with h5py.File(full_path, 'r') as f:
        pr_filt = read_var(f, 'pr_filt')
        lat = read_var(f, "latitude")
        lon = read_var(f, "longitude")
        psdate = read_var(f, "psdate")
        pstart = read_var(f, "pstart")
        pedate = read_var(f, "pedate")
        pstop = read_var(f, "pstop")
        te_adj = read_var(f, "te_adj")
        sa_adj = read_var(f, "sa_adj")

        # helPlot(te_adj, pr_filt)
        print(f'length of profile: {len(pr_filt)}')
        print(f'length of profile without nan:{len(~np.isnan(pr_filt))}')

        depth = height(pr_filt, lat)
        print(f'pressure: {pr_filt}')
        pressure_cal = pressure(depth, lat)
        print(f'function check:{pressure_cal}')

        lat = lat[0]
        print(lat)
        print(type(lat))
        print(lon)
        psdate = pd.to_datetime(psdate, format="%m/%d/%y").date()
        print(psdate)
        print(type(psdate))


except Exception as e:
    print(f"Error processing file: {full_path}")
    traceback.print_exc()