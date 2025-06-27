import argparse
import h5py
import pickle
from helper import *
import pandas as pd
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import gsw
import datetime
import re
import seaborn as sns
from matplotlib.colors import SymLogNorm
import matplotlib.colors as colors
import matplotlib.dates as mdates
from matplotlib.colors import Normalize
from scipy.ndimage import uniform_filter1d
import math

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


final_df = pd.read_pickle("final.pkl")
test_df = final_df[final_df['itpNum'] == 65].copy()

def transView(df, values, window_size):
    ls = []
    # test if the loop for every profile works
    unique_profNum = df['profileNumber'].unique()
    for i in unique_profNum:
        df_on_fly = df[df['profileNumber'] == i].copy()
        temp_smooth = uniform_filter1d(df_on_fly['temp'], size=window_size, mode='nearest')
        salt_smooth = uniform_filter1d(df_on_fly['salinity'], size=window_size, mode='nearest')
        pres_smooth = uniform_filter1d(df_on_fly['pressure'], size=window_size, mode='nearest')
        depth = df_on_fly['depth']
        # add new cols:
        df_on_fly['dT/dZ'] = np.gradient(temp_smooth, depth)
        df_on_fly['dS/dZ'] = np.gradient(salt_smooth, depth)
        n_sq = gsw.Nsquared(salt_smooth, temp_smooth, pres_smooth, df_on_fly['lat'])[0]
        # padding for last value as the function returns only N-1 values
        n_sq_padded = np.append(n_sq, np.nan)
        df_on_fly['smooth_n_sq'] = n_sq_padded
        # turner angle and R_rho
        [turner_angle, R_rho,p_mid] = gsw.Turner_Rsubrho(salt_smooth, temp_smooth, pres_smooth)
        df_on_fly['smooth_turner_angle'] = np.append(turner_angle,np.nan)
        df_on_fly['smooth_R_rho'] = np.append(R_rho,np.nan)
        ls.append(df_on_fly)
    fin_df = pd.concat(ls, ignore_index=True)

    n_plots = len(values)
    n_cols = 2
    n_rows = math.ceil(n_plots / n_cols)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axs = axs.flatten() 

    for i, value in enumerate(values):
        ax = axs[i] if len(values) > 1 else axs  # handle case with 1 subplot

        mask = np.isfinite(fin_df[value]) & (fin_df[value] != 0)
        masked_data = fin_df[value][mask]
        if value == 'smooth_R_rho':
            masked_data = np.log10(np.abs(masked_data))

        # Compute vmin and vmax for color scaling, ignoring outliers
        vmin = np.quantile(masked_data, 0.01)
        vmax = np.quantile(masked_data, 0.99)
        bounds = np.linspace(vmin, vmax, 100)
        norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
        if value == 'smooth_R_rho':
            max_abs = max(abs(vmin), abs(vmax))
            norm = colors.TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)
            sc = ax.scatter(
                fin_df['date'][mask], fin_df['depth'][mask],
                c=masked_data,  # or fin_df[value][mask] if not logging
                cmap='RdBu_r',
                norm=norm,
                s=10, alpha=0.8, marker=(4, 0, 45)
            )
        else: 
            sc = ax.scatter(
                fin_df['date'][mask], fin_df['depth'][mask],
                c=masked_data,  # or fin_df[value][mask] if not logging
                cmap='plasma',
                norm=norm,
                s=10, alpha=0.8, marker=(4, 0, 45)
            )
        fig.colorbar(sc, ax=ax, label=f'log {value}')

        first_marker = True  # flag to add just one legend entry
        grouped = fin_df.groupby('profileNumber')
        for prof_id, group in grouped:
            if group['temp'].isna().all():
                continue  
            temp_max_idx = group['temp'].idxmax()
            temp_max_date = fin_df.loc[temp_max_idx, 'date']
            temp_max_depth = fin_df.loc[temp_max_idx, 'depth']
            # temp_max_val = fin_df.loc[temp_max_idx, 'temp']

            temp_min_idx = group['temp'].idxmin()
            temp_min_date = fin_df.loc[temp_min_idx, 'date']
            temp_min_depth = fin_df.loc[temp_min_idx, 'depth']

            nsq_max_idx = group['smooth_n_sq'].idxmax()
            nsq_max_date = fin_df.loc[nsq_max_idx, 'date']
            nsq_max_depth = fin_df.loc[nsq_max_idx, 'depth']


            ax.plot(temp_min_date, temp_min_depth, "k*", markersize=3, label='Min Temp' if first_marker else "")
            ax.plot(temp_max_date, temp_max_depth, 'y*', markersize=3,label = 'Max Temp' if first_marker else "")
            ax.plot(nsq_max_date, nsq_max_depth, 'r*', markersize=3, label='Max N^2' if first_marker else "")
            first_marker = False

        # ax.annotate(f'{temp_max_val:.2f}°C',
        #             (temp_max_date, temp_max_depth),
        #             textcoords="offset points", xytext=(0, -10), ha='center',
        #             fontsize=8, color='red')

        ax.legend(loc='upper right')

        ax.set_title(f'Logged {value}, ITP 65, window size: {window_size*0.25}m')
        ax.set_xlabel('Date')
        ax.set_ylabel('Depth (m)')
        ax.invert_yaxis()

        ax.xaxis.set_major_locator(mdates.DayLocator(interval=30))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax.get_xticklabels(), rotation=45)  # rotate date labels
    plt.tight_layout()
    # plt.savefig('test.png')
    plt.show()
transView(test_df, ["dT/dZ", 'dS/dZ','smooth_n_sq','smooth_R_rho','smooth_turner_angle'],80)