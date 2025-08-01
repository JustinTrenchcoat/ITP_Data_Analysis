import pickle
from helper import *
import pandas as pd
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
from scipy.ndimage import gaussian_filter1d
import math
import seaborn as sns

import cartopy.crs as ccrs
import pandas as pd
import matplotlib.colors as colors
import matplotlib.dates as mdates
import matplotlib

import cartopy.feature as cfeature
import matplotlib.cm as cm

from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap


# vertical plots

with open('grouped.pkl', 'rb') as f:
    groupedYears = pickle.load(f)


colorscale = LinearSegmentedColormap.from_list("year_trend", [
    "#4575b4",  # blue
    "#91bfdb",
    "#12F012",
    "#fc8d59",
    "#d73027"   # red-orange
])


years = ["2004-2007", '2008-2011', '2012-2015', '2016-2019', '2020-2023']
colors = colorscale(np.linspace(0, 1, len(years)))

temp_min_idx = []
temp_max_idx = []
nsq_max = []

#######################################################################################################################
# making mean, mean+std, mean-std for each group,
def vertPlot(df_list, variable, path, type):
    for i, df_group in enumerate(df_list):
        print(f'-------Processing Group {i}------------------')
        df_copy = df_group.copy()
        # df_copy['year'] = df_copy['date'].apply(lambda d: d.year)
        avg = df_copy.groupby('depth')[variable].agg(["mean", "std", "count"])
        count  = avg['count'].to_numpy()
        count_mask = count > 50
            
        depth = avg.index.to_numpy()
        depth = depth[count_mask]
        mean = avg['mean'].to_numpy()
        mean = mean[count_mask]
        std = avg['std'].to_numpy()
        std = std[count_mask]

        if variable == "temp":
            temp_min_idx.append(mean.argmin())
            temp_max_idx.append(mean.argmax())
        if variable == "n_sq":
             nsq_max.append(mean.argmax())

        
        if variable in ("R_rho","dT/dZ","dS/dZ"):
            plt.axhline(depth[nsq_max[i]], color='k', linestyle='dotted')
        

        # print(depth[mean.argmin()])
        if type == "origin":
            plt.scatter(mean, depth, label = f"Group {i}", color = colors[i], s=5,edgecolors='none')
            plt.axhline(depth[temp_max_idx[i]], color=colors[i], linestyle='--')
            plt.axhline(depth[temp_min_idx[i]], color=colors[i], linestyle='--')
        elif type == "plus":
            plt.scatter(mean+std, depth, alpha=0.1,label = f"Group {i}", color = colors[i])
        else:
            plt.scatter(mean-std,depth, alpha=0.1,label = f"Group {i}", color = colors[i])



    plt.gca().invert_yaxis()
    if type == "origin":
        if variable == "temp":
            plt.xlabel('average temperature (\u00B0C)',fontsize=18)
        elif variable == "salinity":
            plt.xlabel('average salinity (g/kg)',fontsize=18)
        elif variable == "dT/dZ":
            plt.xlabel("average temperature gradient (\u00B0C/m)",fontsize=18)
        elif variable == "dS/dZ":
            plt.xlabel("average salinity gradient((g/kg)/m)",fontsize=18)
        elif variable == "R_rho":
            plt.xlabel("average density gredient ratio",fontsize=18) 
        else:
            plt.xlabel(f"average {variable}",fontsize=18)           
    elif type == "plus":
            plt.xlabel(f'Average {variable}+1 standard deviation')
            plt.title(f"Average of {variable}+std")
    else:
            plt.xlabel(f'Average {variable}-1 standard deviation')
            plt.title(f"Average of {variable}-std")

    plt.ylabel('depth (m)',fontsize=18)
    if variable == "R_rho":
        plt.axvline(x=1, color = 'k',linestyle='dashdot')
        plt.axvline(x=10, color = 'k',linestyle='dashdot')

        
    # # Legend: color squares instead of dots
    # legend_patches = [
    #     Patch(facecolor=colors[i], edgecolor='black', label=str(years[i]))
    #     for i in range(len(years))
    # ]   
    plt.tight_layout()
    plt.savefig(f"plots/presentPlot/{path}{type}")
    plt.show()
    plt.close()
##################################################################################################################
# # legend plot
def plot_legend_only(years, colors, filename, legend_title="Year"):
    fig, ax = plt.subplots()

    # Create legend handles
    legend_patches = [
        Patch(facecolor=colors[i], edgecolor='black', label=str(years[i]))
        for i in range(len(years))
    ]

    # Place the legend in center of new figure
    legend = ax.legend(
        handles=legend_patches,
        title=legend_title,
        loc='center',
        frameon=False,
        fontsize=12,
        title_fontsize=13
    )

    # Hide axes for clean legend-only figure
    ax.axis('off')
    
    # Resize figure to fit legend
    fig.set_size_inches(2.5, len(years) * 0.35 + 1)
    plt.savefig(f"plots/presentPlot/{filename}")
    plt.show()
    plt.close()


vertPlot(groupedYears, "temp", "temp", "origin")
# plot_legend_only(years, colors, "legend")
# vertPlot(groupedYears, "turner_angle", "turner", "origin")
vertPlot(groupedYears, "salinity", "sal", "origin")
vertPlot(groupedYears, "n_sq", "nSq", "origin")
vertPlot(groupedYears, "dT/dZ" , "dTdZ", "origin")
vertPlot(groupedYears, "dS/dZ", "dSdZ", "origin")
vertPlot(groupedYears, "R_rho", "rho", "origin")
################################################################################################################
# # season comparison:
# def seasonSelect(df_list, monthRange):
#     print(f'seasonSelect: working...\nSelecting for month {monthRange}')
#     new_list = []
#     for i, df_group in enumerate(df_list):
#         print(f'-------Processing Group {i}------------------')
#         df_copy = df_group.copy()
#         df_copy['month'] = df_copy['date'].apply(lambda d: d.month)
#         df_select = df_copy[df_copy['month'].isin(monthRange)].copy()
#         new_list.append(df_select)
#         # print(df_select.head())
#     return new_list

# # winter:
# winterlist = seasonSelect(groupedYears, [12,1,2])

# # summer:
# summerList = seasonSelect(groupedYears, [6,7,8])

# trickPlot = [winterlist[4], summerList[4]]

# vertPlot(trickPlot, "temp", "temp", "origin")

# vertPlot(trickPlot, "R_rho", "rho", "origin")

##############################################################################
# # number of profiles per year:
def profileChecker():
    try:
        with open("test.pkl", "rb") as f:
            df = pickle.load(f)
            df_with_counts = df.copy()
            df_with_counts['year'] = df_with_counts['date'].apply(lambda d: d.year)
            # print(df_with_counts.shape)
    
            df_with_counts = (
                df_with_counts.groupby(["year", "systemNum", "profileNum"])
                .size()
            .reset_index(name='count')
            )
            # print(df_with_counts.head())
            years = df_with_counts['year'].unique()
            # print(years)
            counts = []
            for year in years:
                counts.append(len(df_with_counts[df_with_counts["year"]==year]))
    
            # print(np.sum(counts))
            years = years.astype(str)
            
            # Bar plot with notations
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(years, counts, edgecolor='black')
            ax.set_title("Number of Profiles per Year")
            ax.set_xlabel("Year",fontsize=14)
            ax.set_ylabel("Number of Profiles",fontsize=14)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            ax.set_xticks(years[::4])
            ax.set_xticklabels(years[::4], rotation=45)
            
            # Add count labels above each bar
            ax.bar_label(bars, padding=3,fontsize=14)
            plt.savefig(f"plots/presentPlot/yearProfNum")
            plt.tight_layout()
            plt.show()
            plt.close()
            
    except Exception as e:
        traceback.print_exc()
# profileChecker()
###############################################################################################################
# number of profiles in group:
def groupChecker(df):
    try:
        counts = []
        for group in df: 
            df_with_counts = group.copy()
            df_with_counts = (
                df_with_counts.groupby([ 'lat', 'lon',"systemNum", "profileNum"])
                .size().reset_index(name='count'))
            values = len(df_with_counts["count"])
            counts.append(values)
        groupList = ['1', '2', '3', '4', '5']
        # Bar plot with notations
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(['1', '2', '3', '4', '5'], counts, edgecolor='black', color = colors)
        ax.set_title("Number of Profiles per Group", fontsize=14)
        ax.set_xlabel("Group", fontsize=14)
        ax.set_ylabel("Number of Profiles", fontsize=14)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_xticks(groupList)
            
        # Add count labels above each bar
        ax.bar_label(bars, padding=3, fontsize=14)
            
        plt.tight_layout()
        plt.savefig(f"plots/presentPlot/groupProfNum")
        plt.show()
        plt.close()
            
    except Exception as e:
        traceback.print_exc()
# groupChecker(groupedYears)
#####################################################################################################################
# total distribution map
# Group and count
# ITP system trace:
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

def dataTrace(df):
    """
    Plots a density scatter map of observations using lat/lon and count,
    colored by discrete years.
    """
    fig = plt.figure(figsize=(10, 10))
    sizes = 15
    projection = ccrs.LambertConformal(central_longitude=-145, central_latitude=76.5)
    ax = plt.axes(projection=projection)
    ax.set_extent([-160, -130, 72, 81], crs=ccrs.PlateCarree())
    ax.coastlines(resolution='50m', linewidth=0.5)
         

    for i, group in enumerate(df):
         df = traceDF(group)
        #  central_longitude = np.median(df['lon'])
        #  central_latitude = np.median(df["lat"])
        #  projection = ccrs.LambertConformal(central_longitude=central_longitude, central_latitude=central_latitude)
         ax.scatter(
            df['lon'], df['lat'],
            transform=ccrs.PlateCarree(),
            # color = 'grey',
            color=colors[i],
            label=str(i),
            s=sizes,
            alpha=1,
            edgecolor='k',
            linewidth=0.2
        )

    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5)
    gl.top_labels = True
    gl.right_labels = False

    # ax.legend(title="Year", loc="lower left", fontsize="small")
    plt.title(f'Profile Distribution')
    plt.tight_layout()
    # plt.savefig(f"plots/presentPlot/dataTraceTotalBW")
    plt.savefig(f"plots/presentPlot/dataTraceTotal")
    plt.show()
    plt.close()

# dataTrace(groupedYears)
