import pandas as pd
from helper import *
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


final_df = pd.read_pickle("final.pkl")
test_df = final_df[final_df['itpNum'] == 100].copy()

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
        logged_value = np.log10(np.abs(masked_data))

        # Compute vmin and vmax for color scaling, ignoring outliers
        vmin = np.quantile(logged_value, 0.01)
        vmax = np.quantile(logged_value, 0.99)
        bounds = np.linspace(vmin, vmax, 100)
        norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)

        sc = ax.scatter(
            fin_df['date'][mask], fin_df['depth'][mask],
            c=logged_value,  # or fin_df[value][mask] if not logging
            cmap='RdBu_r',
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
            ax.plot(nsq_max_date, nsq_max_depth, '*',  color='orange', markersize=3, label='Max N^2' if first_marker else "")
            first_marker = False

        ax.legend(loc='upper right')

        ax.set_title(f'Logged {value}, ITP 100, window size: {window_size}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Depth (m)')
        ax.invert_yaxis()

        ax.xaxis.set_major_locator(mdates.DayLocator(interval=30))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax.get_xticklabels(), rotation=45)  # rotate date labels
    plt.tight_layout()
    # plt.savefig('test.png')
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Create sample grid data with some missing values
dates_unique = test_df['date'].unique()
print(type(dates_unique[0]))
# need to reorder
depth_unique = test_df['depth'].unique()

X, Y = np.meshgrid(dates_unique, depth_unique)
# print(X[0])
# get index for where x and y are nan??
# print(np.where(np.isnan(X)))
# Z = np.sin(X) + np.cos(Y)

# # Introduce missing values (NaN) at specific locations
# Z[5:8, 10:13] = np.nan
# Z[15:17, 2:5] = np.nan

# # Create the plot
# plt.figure(figsize=(8, 6))

# # Plot the data using pcolormesh or contourf for grid data
# # pcolormesh is good for visualizing the grid cells
# plt.pcolormesh(X, Y, Z, shading='auto', cmap='viridis')

# # Add a colorbar to indicate the values
# plt.colorbar(label='Z values')

# # Add grid lines for better visualization of the grid
# plt.grid(True, linestyle='--', alpha=0.7)

# # Add titles and labels
# plt.title('Grid Data with Missing Values')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')

# # Show the plot
# plt.show()