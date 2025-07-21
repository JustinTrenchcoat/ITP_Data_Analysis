import numpy as np
import cartopy.crs as ccrs
import pickle
from helper import *
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.dates as mdates
import math
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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
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


import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import pandas as pd

def dataTrace(df, groupNum, cmap='Spectral_r'):
    """
    Plot trajectory of ITP systems colored by year on a map.
    Assumes input df has: systemNum, profileNum, year, lon, lat
    """
    df_with_counts = df.copy()
    df_with_counts['year'] = df_with_counts['date'].apply(lambda d: d.year)

    # Determine map center
    central_longitude = np.median(df_with_counts['lon'])
    central_latitude = np.median(df_with_counts['lat'])

    # Lambert Conformal Projection
    projection = ccrs.LambertConformal(
        central_longitude=central_longitude,
        central_latitude=central_latitude
    )

    # Prepare figure and axes
    fig = plt.figure(figsize=(8, 6))
    ax = plt.axes(projection=projection)
    ax.set_extent([-160, -130, 72, 81], crs=ccrs.PlateCarree())
    ax.coastlines(resolution='50m', linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4)
    ax.add_feature(cfeature.LAND, zorder=0, facecolor='lightgray')

    # Get unique years and assign a colormap
    years = sorted(df_with_counts['year'].unique())
    norm = plt.Normalize(min(years), max(years))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    # Plot each trajectory
    for system in df_with_counts['systemNum'].unique():
        sub = df_with_counts[df_with_counts['systemNum'] == system].sort_values(['year', 'profileNum'])

        ax.plot(
            sub['lon'], sub['lat'],
            transform=ccrs.PlateCarree(),
            color=sm.to_rgba(sub['year'].iloc[0]),
            linewidth=1.2,
            label=f'ITP {system}' if system == df_with_counts['systemNum'].unique()[0] else None  # avoid legend spam
        )

    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5)
    gl.top_labels = True
    gl.right_labels = False

    # Add colorbar
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', shrink=0.7, pad=0.05)
    cbar.set_label('Year')

    # Title and save
    plt.title(f'Trajectory of ITP Systems (Colored by Year), Group {groupNum}')
    plt.tight_layout()
    # plt.savefig(f"plots/trajectory/TrajectoryG{groupNum}")
    plt.show()
    plt.close()



for i in range (5):
    print(f"processing group{i}")
    dataTrace(groupedYears[i], i)
