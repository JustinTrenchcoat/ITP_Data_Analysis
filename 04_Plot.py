import pickle
from helper import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
import matplotlib.dates as mdates
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

            # Group by depth and compute mean, std, and count
            avgStart = yearS_df.groupby('depth')[variable].agg(["mean", "std", "count"])
            avgEnd = yearE_df.groupby('depth')[variable].agg(["mean", "std", "count"])

            # Monitor depths with large number of observations
            
            depthStart = avgStart.index.to_numpy()
            meanStart = avgStart['mean'].to_numpy()
            stdStart = avgStart['std'].to_numpy()

            depthEnd = avgEnd.index.to_numpy()
            meanEnd = avgEnd['mean'].to_numpy()
            stdEnd = avgEnd['std'].to_numpy()

            # # # Main plot: mean profiles
            plt.errorbar(meanStart, depthStart, xerr=stdStart, fmt='-o', alpha=0.2, capsize=3, label=f'{yearStart}')
            plt.errorbar(meanEnd, depthEnd, xerr=stdEnd, fmt='-o', alpha=0.2, capsize=3, label=f'{yearEnd}')
            plt.gca().invert_yaxis()  
            plt.xlabel(f'Average {variable}')
            plt.title(f"Comparison of {variable} between {yearStart} and {yearEnd}")
            plt.ylabel('Depth')
            plt.legend()
            plt.show()

            # Plot depth vs. count (histogram-like bar plot)
            plt.figure(figsize=(10, 6))
            plt.bar(avgStart.index, len(depthStart), width=0.8, alpha=0.5, color='red', label=f'{yearStart} Counts')
            plt.bar(avgEnd.index, len(depthEnd), width=0.8, alpha=0.5, color='blue', label=f'{yearEnd} Counts')
            plt.xlabel('Depth')
            plt.ylabel('Number of Observations')
            plt.title('Number of Observations per Depth')
            plt.legend()
            plt.show()

            # Plot histogram of minimum depth per profile
            # Get index (row number) of max temperature per profile
            idx_max_temp_start = yearS_df.groupby('profileNum')['temp'].idxmax()
            idx_max_temp_end = yearE_df.groupby('profileNum')['temp'].idxmax()

            idx_min_temp_start = yearS_df.groupby('profileNum')['temp'].idxmin()
            idx_min_temp_end = yearE_df.groupby('profileNum')['temp'].idxmin()

            # Access corresponding depths at max temperature:
            max_temp_depth_start = yearS_df.loc[idx_max_temp_start, 'depth'].values
            max_temp_depth_end = yearE_df.loc[idx_max_temp_end, 'depth'].values

            min_temp_depth_start = yearS_df.loc[idx_min_temp_start, 'depth'].values
            min_temp_depth_end = yearE_df.loc[idx_min_temp_end, 'depth'].values

            plt.figure(figsize=(10, 6))
            counts_start, _, patches_start =plt.hist(min_temp_depth_start, bins=30, alpha=0.5, color='red', 
                                    label=f'{yearStart} Depth at Tmin per Profile, total {len(min_temp_depth_start)} observations')
            counts_end, _, patches_end =plt.hist(min_temp_depth_end, bins=30, alpha=0.5, color='blue', 
                                    label=f'{yearEnd} Depth at Tmin per Profile, total {len(min_temp_depth_end)} observations')
            for count, patch in zip(counts_start, patches_start):
                # omit 0 so the plot looks better
                if count != 0:
                    plt.text(patch.get_x() + patch.get_width()/2, count, int(count),
                    ha='center', va='bottom', fontsize=8)
            for count, patch in zip(counts_end, patches_end):
                if count != 0:
                    plt.text(patch.get_x() + patch.get_width()/2, count, int(count),
                    ha='center', va='bottom', fontsize=8)
            plt.xlabel('Depth at Tmin per Profile')
            plt.ylabel('Frequency')
            plt.title('Histogram of Depth at Tmin per Profile')
            plt.legend()
            plt.show()

            plt.figure(figsize=(10, 6))
            counts_start, _, patches_start =plt.hist(max_temp_depth_start, bins=30, alpha=0.5, color='red', 
                                    label=f'{yearStart} Depth at Tmax per Profile, total {len(max_temp_depth_start)} observations')
            counts_end, _, patches_end =plt.hist(max_temp_depth_end, bins=30, alpha=0.5, color='blue', 
                                    label=f'{yearEnd} Depth at Tmax per Profile, total {len(max_temp_depth_end)} observations')
            for count, patch in zip(counts_start, patches_start):
                # omit 0 so the plot looks better
                if count != 0:
                    plt.text(patch.get_x() + patch.get_width()/2, count, int(count),
                    ha='center', va='bottom', fontsize=8)
            for count, patch in zip(counts_end, patches_end):
                if count != 0:
                    plt.text(patch.get_x() + patch.get_width()/2, count, int(count),
                    ha='center', va='bottom', fontsize=8)
            plt.xlabel('Depth at Tmax per Profile')
            plt.ylabel('Frequency')
            plt.title('Histogram of Depth at Tmax per Profile')
            plt.legend()
            plt.show()


            # value of Tmin and Tmax:
            max_temp_start = yearS_df.loc[idx_max_temp_start, 'temp'].values
            max_temp_end = yearE_df.loc[idx_max_temp_end, 'temp'].values

            min_temp_start = yearS_df.loc[idx_min_temp_start, 'temp'].values
            min_temp_end = yearE_df.loc[idx_min_temp_end, 'temp'].values

            plt.figure(figsize=(10, 6))
            counts_start, _, patches_start =plt.hist(min_temp_start, bins=30, alpha=0.5, color='red', 
                                                     label=f'{yearStart} Tmin per Profile, total {len(min_temp_start)} observations')
            counts_end, _, patches_end =plt.hist(min_temp_end, bins=30, alpha=0.5, color='blue', 
                                                 label=f'{yearEnd} Tmin per Profile, total {len(min_temp_end)} observations')
            for count, patch in zip(counts_start, patches_start):
                # omit 0 so the plot looks better
                if count != 0:
                    plt.text(patch.get_x() + patch.get_width()/2, count, int(count),
                    ha='center', va='bottom', fontsize=8)
            for count, patch in zip(counts_end, patches_end):
                if count != 0:
                    plt.text(patch.get_x() + patch.get_width()/2, count, int(count),
                    ha='center', va='bottom', fontsize=8)
            plt.xlabel('Tmin per Profile')
            plt.ylabel('Frequency')
            plt.title('Histogram of Tmin per Profile')
            plt.legend()
            plt.show()

            plt.figure(figsize=(10, 6))
            counts_start, _, patches_start =plt.hist(max_temp_start, bins=30, alpha=0.5, color='red', 
                                                     label=f'{yearStart} Tmax per Profile, total {len(max_temp_start)} observations')
            counts_end, _, patches_end =plt.hist(max_temp_end, bins=30, alpha=0.5, color='blue', 
                                                 label=f'{yearEnd} Tmax per Profile, total {len(max_temp_end)} observations')
            for count, patch in zip(counts_start, patches_start):
                # omit 0 so the plot looks better
                if count != 0:
                    plt.text(patch.get_x() + patch.get_width()/2, count, int(count),
                    ha='center', va='bottom', fontsize=8)
            for count, patch in zip(counts_end, patches_end):
                if count != 0:
                    plt.text(patch.get_x() + patch.get_width()/2, count, int(count),
                    ha='center', va='bottom', fontsize=8)
            plt.xlabel('Tmax per Profile')
            plt.ylabel('Frequency')
            plt.title('Histogram of Tmax per Profile')
            plt.legend()
            plt.show()

            temp_diff_start = max_temp_start-min_temp_start
            temp_diff_end = max_temp_end-min_temp_end

            depth_diff_start = max_temp_depth_start-min_temp_depth_start
            depth_diff_end = max_temp_depth_end-min_temp_depth_end

            gradient_start = (temp_diff_start)/(depth_diff_start)
            gradient_end =  (temp_diff_end)/(depth_diff_end)

            plt.figure(figsize=(10, 6))
            counts_start, _, patches_start = plt.hist(gradient_start, bins=30, alpha=0.5, color='red', 
                    label=f'{yearStart} thermocline-scale temperature gradient per Profile, total {len(gradient_start)} observations')
            counts_end, _, patches_end = plt.hist(gradient_end, bins=30, alpha=0.5, color='blue', 
                    label=f'{yearEnd} thermocline-scale temperature gradient per Profile, total {len(gradient_end)} observations')
            for count, patch in zip(counts_start, patches_start):
                # omit 0 so the plot looks better
                if count != 0:
                    plt.text(patch.get_x() + patch.get_width()/2, count, int(count),
                    ha='center', va='bottom', fontsize=8)
            for count, patch in zip(counts_end, patches_end):
                if count != 0:
                    plt.text(patch.get_x() + patch.get_width()/2, count, int(count),
                    ha='center', va='bottom', fontsize=8)
            plt.xlabel('thermocline-scale temperature gradient values')
            plt.ylabel('Frequency')
            plt.title('Histogram of thermocline-scale temperature gradient per Profile')
            plt.legend()
            plt.show()


            rhoStart = yearS_df.groupby('profileNum')['R_rho'].agg(lambda x: np.mean(np.log10(x)))
            R_rho_start = rhoStart.to_numpy()
            R_rho_start_count = len(R_rho_start)

            rhoEnd = yearE_df.groupby('profileNum')['R_rho'].agg(lambda x: np.mean(np.log10(x)))
            R_rho_end = rhoEnd.to_numpy()
            R_rho_end_count = len(R_rho_end)

            plt.figure(figsize=(10, 6))
            counts_start, _, patches_start = plt.hist(R_rho_start, bins=100, alpha=0.5,color='red', 
                                                    label=f'{yearStart} average log10(R_rho) per Profile, total {R_rho_start_count} observations')
            counts_end,_, patches_end = plt.hist(R_rho_end, bins=100, alpha=0.5, color='blue', 
                                                label=f'{yearEnd} average log10(R_rho) per Profile, total {R_rho_end_count} observations')

            for count, patch in zip(counts_start, patches_start):
                # omit 0 so the plot looks better
                if count != 0:
                    plt.text(patch.get_x() + patch.get_width()/2, count, int(count),
                    ha='center', va='bottom', fontsize=8)
            for count, patch in zip(counts_end, patches_end):
                if count != 0:
                    plt.text(patch.get_x() + patch.get_width()/2, count, int(count),
                    ha='center', va='bottom', fontsize=8)
            plt.xlabel('average log10(R_rho) values')
            plt.ylabel('Frequency')
            plt.title('Histogram of average log10(R_rho) per Profile')
            plt.legend()
            plt.show()

    except Exception as e:
        traceback.print_exc()
# overlapPlot('n_sq', 2007, 2015)
def profileChecker():
    try:
        with open("test.pkl", "rb") as f:
            df = pickle.load(f)
            years = df['date'].apply(lambda d: d.year).unique()
            years_sorted = np.sort(years)
            print(years_sorted)
            
            year_counts = []
            for year in years_sorted:
                year_df = df[df['date'].apply(lambda d: d.year) == year].copy()
                profiles = year_df['profileNum'].unique()
                print(f'year {year} has {len(profiles)} profiles')
                year_counts.append(len(profiles))
            
            # Bar plot with notations
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(years_sorted, year_counts, edgecolor='black')
            ax.set_title("Number of Profiles per Year")
            ax.set_xlabel("Year")
            ax.set_ylabel("Number of Profiles")
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            ax.set_xticks(years_sorted)
            
            # Add count labels above each bar
            ax.bar_label(bars, padding=3)
            
            plt.tight_layout()
            plt.show()
            
    except Exception as e:
        traceback.print_exc()
# profileChecker()
#################################################################################################
# data density plot

with open('grouped.pkl', 'rb') as f:
    groupedYears = pickle.load(f)

# Group and count
def simpleDF(df):
    df_with_counts = (
        df.groupby(['date', 'profileNum', 'lat', 'lon'])
        .size()
        .reset_index(name='count')
    )
    df_with_counts['year'] = df_with_counts['date'].apply(lambda d: d.year)
    return df_with_counts

group_zero = simpleDF(groupedYears[0])

# print(group_zero.head())
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
def plot_density_map(df, groupNum, bins=50, log_scale=True,cmap='Spectral_r'):
    """
    Plots a density map of observations using lat/lon and count.
    Rotates polar projection to center over your data region.
    """
    # Determine central meridian if not given

    central_longitude = np.median(df['lon'])
    central_latitude = np.median(df["lat"])

    # Define projection centered on your data
    projection = ccrs.LambertConformal(central_longitude=central_longitude, central_latitude=central_latitude)

    # Bin edges
    # lon_bins = np.linspace(df['lon'].min(), df['lon'].max(), bins if isinstance(bins, int) else bins[0])
    lon_bins = np.linspace(-160, -130, bins)

    lat_bins = np.linspace(73, 81, bins)

    # Prepare weights
    values = df["count"].values
    if log_scale:
        values = np.where(values > 0, np.log10(values), 0)

    # 2D histogram
    hist, lon_edges, lat_edges = np.histogram2d(
        df['lon'], df["lat"], bins=[lon_bins, lat_bins], weights=values
    )

    # Grid centers
    lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    lon_grid, lat_grid = np.meshgrid(lon_centers, lat_centers)

    # Plot
    fig = plt.figure(figsize=(6, 6))
    ax = plt.axes(projection=projection)
    ax.set_extent([df['lon'].min(), df['lon'].max(),
                   df["lat"].min(), df["lat"].max()], crs=ccrs.PlateCarree())

    ax.coastlines(resolution='50m', linewidth=0.5)

    pcm = ax.pcolormesh(lon_grid, lat_grid, hist.T, cmap=cmap, shading='auto',
                        transform=ccrs.PlateCarree())

    cbar = plt.colorbar(pcm, ax=ax, shrink=0.7, pad=0.05)
    cbar.set_label("Log10 Number of Observations" if log_scale else "Number of Observations")

    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5)
    gl.top_labels = True
    gl.right_labels = False

    plt.title(f'Observation Density Map (Log10 Count) of group {groupNum}')
    plt.tight_layout()
    plt.savefig(f"plots/heatmap/G{groupNum}")
    plt.show()


for i in range (5):
    df = simpleDF(groupedYears[i])
    plot_density_map(df, i)
