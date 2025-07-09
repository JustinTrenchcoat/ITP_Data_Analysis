import pickle
from helper import *
import pandas as pd
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import matplotlib.colors as colors
import matplotlib.dates as mdates
from scipy.ndimage import gaussian_filter1d
import math
import seaborn as sns

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

def depthDiffHist():
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

# final_df = pd.read_pickle("final.pkl")
# experiment_df = final_df[final_df['itpNum'].isin([62, 65, 68])].copy()

def transView(df, values, window_size):

    n_plots = len(values)
    n_cols = 2
    n_rows = math.ceil(n_plots / n_cols)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axs = axs.flatten() 

    for i, value in enumerate(values):
        ax = axs[i] if len(values) > 1 else axs  # handle case with 1 subplot

        mask = np.isfinite(df[value]) & (df[value] != 0)
        masked_data = df[value][mask]
        if value == 'R_rho':
            masked_data = np.log10(np.abs(masked_data))

        # Compute vmin and vmax for color scaling, ignoring outliers
        vmin = np.quantile(masked_data, 0.01)
        vmax = np.quantile(masked_data, 0.99)
        bounds = np.linspace(vmin, vmax, 100)
        norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
        if value == 'R_rho':
            max_abs = max(abs(vmin), abs(vmax))
            norm = colors.TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)
            sc = ax.scatter(
                df['date'][mask], df['depth'][mask],
                c=masked_data,  # or fin_df[value][mask] if not logging
                cmap='RdBu_r',
                norm=norm,
                s=10, alpha=0.8, marker=(4, 0, 45)
            )
        else: 
            sc = ax.scatter(
                df['date'][mask], df['depth'][mask],
                c=masked_data,  # or fin_df[value][mask] if not logging
                cmap='plasma',
                norm=norm,
                s=10, alpha=0.8, marker=(4, 0, 45)
            )
        fig.colorbar(sc, ax=ax, label=f'log {value}')

        first_marker = True  # flag to add just one legend entry
        grouped = df.groupby('profileNumber')
        for prof_id, group in grouped:
            if group['temp'].isna().all():
                continue  
            temp_max_idx = group['temp'].idxmax()
            temp_max_date = df.loc[temp_max_idx, 'date']
            temp_max_depth = df.loc[temp_max_idx, 'depth']
            # temp_max_val = fin_df.loc[temp_max_idx, 'temp']

            temp_min_idx = group['temp'].idxmin()
            temp_min_date = df.loc[temp_min_idx, 'date']
            temp_min_depth = df.loc[temp_min_idx, 'depth']

            nsq_max_idx = group['n_sq'].idxmax()
            nsq_max_date = df.loc[nsq_max_idx, 'date']
            nsq_max_depth = df.loc[nsq_max_idx, 'depth']


            ax.plot(temp_min_date, temp_min_depth, "k*", markersize=3, label='Min Temp' if first_marker else "")
            ax.plot(temp_max_date, temp_max_depth, 'y*', markersize=3,label = 'Max Temp' if first_marker else "")
            ax.plot(nsq_max_date, nsq_max_depth, 'r*', markersize=3, label='Max N^2' if first_marker else "")
            first_marker = False

        ax.legend(loc='upper right')

        ax.set_title(f'Logged {value}, window size: {window_size*0.25}m')
        ax.set_xlabel('Date')
        ax.set_ylabel('Depth (m)')
        ax.invert_yaxis()

        ax.xaxis.set_major_locator(mdates.DayLocator(interval=30))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax.get_xticklabels(), rotation=45)  # rotate date labels
    plt.tight_layout()
    plt.show()
#####################################################################################    
# transView(experiment_df, ["dT/dZ", 'dS/dZ','n_sq','R_rho','turner_angle'],80)
def dataSummary(data, name, year):
    # Basic statistics
    print(f"Statistical Summary of {name}:")
    print(f"Count         : {len(data)}")
    print(f"Min           : {np.min(data)}")
    print(f"Max           : {np.max(data)}")
    print(f"Mean          : {np.mean(data)}")
    print(f"Median        : {np.median(data)}")
    print(f"Std Dev       : {np.std(data)}")
    print(f"Variance      : {np.var(data)}")
    print(f"25th Percentile (Q1): {np.percentile(data, 25)}")
    print(f"75th Percentile (Q3): {np.percentile(data, 75)}")
    print(f"IQR           : {np.percentile(data, 75) - np.percentile(data, 25)}")

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    counts, edges, bars = plt.hist(data, bins=50, log=(name == 'n_sq' or name == 'R_rho'), edgecolor='black')  # adjust bin count or bin edges here
    plt.bar_label(bars)
    plt.title(f"Histogram of {'logged ' + name if name in ['n_sq', 'R_rho'] else name} in {year}")
    plt.xlabel(f"{'logged ' + name if name in ['n_sq', 'R_rho'] else name}")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
###########################################################################################
def processDataHist(variable, year):
    try:
        # Load the depth differences
        with open("final.pkl", "rb") as f:
            df = pickle.load(f)
            year_df = df[df['date'].apply(lambda d: d.year) == year].copy()
            dataSummary(year_df[variable], variable, year)
    except Exception as e:
        traceback.print_exc()

# for year in years_list:
#     processDataHist("dT/dZ", year)
###############################################################################################
def overlapPlot(variable, yearStart, yearEnd):
    try:
        with open("test.pkl", "rb") as f:
            df = pickle.load(f)
            yearS_df = df[df['date'].apply(lambda d: d.year) == yearStart].copy()

            yearE_df = df[df['date'].apply(lambda d: d.year) == yearEnd].copy()

            # Group by depth and compute average dtdz
            avgStart = yearS_df.groupby('depth')[variable].agg(["mean", "std"])
            avgEnd = yearE_df.groupby('depth')[variable].agg(["mean", "std"])


            depthStart = avgStart.index.to_numpy()
            # 2660648
            print(f"length of depthStart:{len(depthStart)}")
            meanStart = avgStart['mean'].to_numpy()
            print(f"length of meanStart:{len(meanStart)}")
            stdStart= avgStart['std'].to_numpy()

            depthEnd = avgEnd.index.to_numpy()
            meanEnd = avgEnd['mean'].to_numpy()
            stdEnd= avgEnd['std'].to_numpy()

            # sns.lineplot(data=avgStart, x="mean", y="depth", errorbar=('sd', 1))
            # sns.lineplot(data=avgEnd, x="mean", y="depth", errorbar=('sd', 1))
            plt.errorbar(meanStart, depthStart, xerr=stdStart, fmt='-o', alpha = 0.2, capsize=3, label=f'{yearStart}')
            plt.errorbar(meanEnd, depthEnd, xerr=stdEnd, fmt='-o', alpha=0.2, capsize=3, label=f'{yearEnd}')

            plt.gca().invert_yaxis()  # optional: invert depth axis if needed
            plt.xlabel(f'Average {variable}')
            plt.title(f"Comparison of {variable} between {yearStart} and {yearEnd}")
            plt.ylabel('Depth')
            plt.legend()
            plt.show()


    except Exception as e:
        traceback.print_exc()

overlapPlot('n_sq', 2007, 2015)