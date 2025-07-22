import numpy as np
import cartopy.crs as ccrs
import pickle
from helper import *
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.dates as mdates
import matplotlib
import math
import cartopy.crs as ccrs
#################################################################################################
# data density plot

with open('grouped.pkl', 'rb') as f:
    groupedYears = pickle.load(f)

# Group and count
def simpleDF(df):
    df_with_counts = (
        df.groupby(['lat', 'lon'])
        .size()
        .reset_index(name='count')
    )
    df_with_counts['count'] = 1
    return df_with_counts

group_zero = simpleDF(groupedYears[0])

# print(group_zero.head())

def plot_density_map(df, groupNum, bins=50, log_scale=False,cmap='Spectral_r'):
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
    # ax.set_extent([df['lon'].min(), df['lon'].max(),
    #                df["lat"].min(), df["lat"].max()], crs=ccrs.PlateCarree())
    ax.set_extent([-160, -130,
                   72, 81], crs=ccrs.PlateCarree())
    # print(f"longitude range is: {df['lon'].min()}, {df['lon'].max()}")
    # print(f"latitude range is: {df['lat'].min()}, {df['lat'].max()}")

    ax.coastlines(resolution='50m', linewidth=0.5)

    pcm = ax.pcolormesh(lon_grid, lat_grid, hist.T, cmap=cmap, shading='auto',
                        transform=ccrs.PlateCarree())

    cbar = plt.colorbar(pcm, ax=ax, shrink=0.7, pad=0.05)
    cbar.set_label("Log 10 scaled Number of profiles" if log_scale else "Number of Profiles")

    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5)
    gl.top_labels = True
    gl.right_labels = False

    plt.title(f'Profile Density Map of group {groupNum}, N= {len(df["count"])}')
    plt.tight_layout()
    plt.savefig(f"plots/heatmap/ProfileG{groupNum}")
    # plt.show()
    plt.close()


# for i in range (5):
#     df = simpleDF(groupedYears[i])
#     print(f"processing group{i}")
#     plot_density_map(df, i)



def dataDistribution(df, groupNum):
    df_with_counts = df.copy()
    df_with_counts['year'] = df_with_counts['date'].apply(lambda d: d.year)
    
    df_with_counts = (
        df_with_counts.groupby(["year", "systemNum", "profileNum"])
        .size()
        .reset_index(name='count')
    )
    years = df_with_counts['year'].unique()
    counts = []
    for year in years:
        counts.append(len(df_with_counts[df_with_counts["year"]==year]))
    
    print(np.sum(counts))
    years = years.astype(str)


    # Plot
    fig, ax = plt.subplots()
    bar_colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange'] 
    bar_colors = bar_colors[:len(years)] 
    
    bars = ax.bar(years, counts, color=bar_colors)

    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 5, str(height),
                ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Number of profiles')
    ax.set_title(f'Number of profiles per year, Group {groupNum}')
    ax.set_xlabel('Year')
    ax.set_ylim(0, max(counts)*1.1)
    plt.savefig(f"plots/heatmap/ProfileNumG{groupNum}")
    # plt.show()
    plt.close()



# for i in range (5):
#     print(f"processing group{i}")
#     dataDistribution(groupedYears[i], i)


def traceDF(df):
    df_with_counts = df.copy()
    df_with_counts['year'] = df_with_counts['date'].apply(lambda d: d.year)
    df_with_counts = (
        df_with_counts.groupby(["lat", "lon", "year", "systemNum", "profileNum"])
        .size()
        .reset_index(name='count')
        )
    df_with_counts['count'] = 1
    return df_with_counts

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.cm as cm

def dataTrace(df, groupNum, log_scale=False, cmap='Set1'):
    """
    Plots a density scatter map of observations using lat/lon and count,
    colored by discrete years.
    """
    df = traceDF(df)
    central_longitude = np.median(df['lon'])
    central_latitude = np.median(df["lat"])

    projection = ccrs.LambertConformal(central_longitude=central_longitude, central_latitude=central_latitude)

    fig = plt.figure(figsize=(7, 7))
    ax = plt.axes(projection=projection)
    ax.set_extent([-160, -130, 72, 81], crs=ccrs.PlateCarree())

    ax.coastlines(resolution='50m', linewidth=0.5)
    # ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    # ax.add_feature(cfeature.LAND, facecolor='lightgray')
    # ax.add_feature(cfeature.OCEAN, facecolor='white')

    # Prepare color map
    unique_years = sorted(df['year'].unique())
    n_years = len(unique_years)
    colors = matplotlib.colormaps['Set1'].resampled(n_years)

    # Plot each year separately
    for i, year in enumerate(unique_years):
        subset = df[df['year'] == year]
        sizes = 15
        ax.scatter(
            subset['lon'], subset['lat'],
            transform=ccrs.PlateCarree(),
            color=colors(i),
            label=str(year),
            s=sizes,
            alpha=1,
            edgecolor='k',
            linewidth=0.2
        )

    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5)
    gl.top_labels = True
    gl.right_labels = False

    ax.legend(title="Year", loc="lower left", fontsize="small")
    plt.title(f'Profile Distribution by Year, Group {groupNum}, N = {len(df)}')
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"plots/heatmap/dataTraceG{groupNum}")
    plt.close()

for i in range (5):
    print(f"processing group{i}")
    dataTrace(groupedYears[i], i)
