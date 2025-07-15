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

'''
 1. histogram of all estimates of property Z for time period Y.  If easy to do, I suggest a grid of Z rows and Y columns.  Y=5.  
 Z to be determined below. Mark on each distribution the median value with a horizontal line.
 2. a figure of X rows that show time series of “box and whisker plots” 
 3. overlapping Y=5 vertical mean profiles with STD envelopes vertical profiles: 1 for each X. 
 Make the 5 different profiles different colours and add a legend that explains which colour corresponds to which time period.

 X:
 T inside the AW thermocline
S inside the AW thermocline
rho inside the AW thermocline
dT/dz inside the AW thermocline
dS/dz inside the AW thermocline
N^2 inside the AW thermocline
log10(R_rho) inside the AW thermocline
'''

# read file:
with open('grouped.pkl', 'rb') as f:
    groupedYears = pickle.load(f)


def singleGroupFinePlot(df, variable, title, xlabel, groupNum, filename, log = False):
    df['year'] = df['date'].apply(lambda d: d.year)
    # takes in the df, check every profile:
    data = df[variable].values
    data = data[np.isfinite(data)]
    if log:
        data = np.log10(data)
        data = data[np.isfinite(data)]

    # Histogram:
    plt.figure(figsize=(10, 6))
    counts, _, patches =plt.hist(data, bins=70, alpha=0.5, color='red', 
                                    label=f'{title} in year Group {groupNum}')
    plt.axvline(np.median(data), color='k', linestyle='dashed', linewidth=1)
    plt.text(np.median(data), plt.ylim()[1]*0.9, f'Median: {np.median(data):.2f}', va='top', ha='right')

    for count, patch in zip(counts, patches):
                # omit 0 so the plot looks better
                if count != 0:
                    plt.text(patch.get_x() + patch.get_width()/2, count, int(count),
                    ha='center', va='bottom', fontsize=5)
    plt.xlabel(f'{title} in year group {groupNum} ({xlabel})')
    plt.ylabel(f'Frequency')
    plt.title(f'Histogram of Fine Scale {title}')
    plt.legend()
    plt.savefig(f"plots/FineHist{filename}G{groupNum}")
    plt.close()
    # Boxplot:
    plt.figure(figsize=(10, 6))
    plt.boxplot(data)
    plt.title(f"Boxplot of Fine Scale {title}, group{i}")
    plt.xlabel(f'{title} in year group {groupNum} ({xlabel})')
    plt.ylabel(f'{xlabel}')   
    plt.savefig(f"plots/FineBox{filename}G{groupNum}")
    plt.close()


# # T inside the AW thermocline:
# for i in range(5):
#     singleGroupFinePlot(groupedYears[i], 'temp', "Temperature", "Celcius", i, "Temp" )

# #  S inside the AW thermocline:
# for i in range(5):
#     singleGroupFinePlot(groupedYears[i], 'salinity', "Salinity", "g/kg", i, "Sal" )

# #  rho inside the AW thermocline:
# for i in range(5):
#     singleGroupFinePlot(groupedYears[i], 'R_rho', "Density Ratio", " ", i, "Rho" )

# #  dT/dZ inside the AW thermocline:
# for i in range(5):
#     singleGroupFinePlot(groupedYears[i], 'dT/dZ', "Temperature Gradient", "Celcius/m", i, "TempGradient" )

# #  dS/dZ inside the AW thermocline:
# for i in range(5):
#     singleGroupFinePlot(groupedYears[i], 'dS/dZ', "Salinity Gradient", "g/kg/m", i, "SalGradient" )

# # n_sq
# for i in range(5):
#     singleGroupFinePlot(groupedYears[i], 'n_sq', "N^2", " ", i, "NSq" )

# for i in range(5):
#     singleGroupFinePlot(groupedYears[i], 'R_rho', "log10(Density Ratio)", " ", i, "RhoLog", log = True)

# we need a singleErrorbarPlot(), and loop it in the below function
def singleErrorBarPlot(df_list, variable,save_path=None):

    for i, df_group in enumerate(df_list):
        print(f'-------Processing Group {i}------------------')
        df_copy = df_group.copy()
        # df_copy['year'] = df_copy['date'].apply(lambda d: d.year)
        avg = df_copy.groupby('depth')[variable].agg(["mean", "std", "count"])
            
        depth = avg.index.to_numpy()
        mean = avg['mean'].to_numpy()
        std = avg['std'].to_numpy()
        plt.errorbar(mean, depth, xerr=std, fmt='-o', alpha=0.1, capsize=3, label=f'Group {i}')

    plt.gca().invert_yaxis()  
    plt.xlabel(f'Average {variable}')
    plt.title(f"Comparison of {variable}")
    plt.ylabel('Depth')
    plt.legend()
    plt.show()


    plt.tight_layout()
    if save_path:
        plt.savefig(f"plots/{save_path}")


singleErrorBarPlot(groupedYears, "temp")


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

    except Exception as e:
        traceback.print_exc()
