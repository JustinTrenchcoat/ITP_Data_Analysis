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
FIRST FOR BULK SCALE PROPERTIES (i.e. properties over the scale of the AW thermocline as a whole; there is one bulk-scale property per profile)
1. first the histogram of all estimates of property X for time period Y.  If easy to do, I suggest a grid of X rows and Y columns.  Y=5.  
X to be determined below. Mark on each distribution the median value with a horizontal line.
2. second a figure of X rows that show time series of “box and whisker plots” 
that summarize the distributions for each time period and shows how these distributions vary over the 20-year record (there will be Y=5 “box and whisker” items per time series)

The list of X (the variables of interest):
1. depth of Tmin (the depth of the top of the AW thermocline)
2. depth of Tmax (the depth of the bottom of the AW thermocline)
3. DZ=depth of Tmax - depth of Tmin (the thickness of the AW thermocline)
4. T at the depth of the Tmin
5. T at the depth of the Tmax
6. DT = bulk-scale temperature change over the scale of the AW thermocline
7. S at the depth of the Tmin
8. S at the depth of the Tmax
9. DS = bulk-scale salinity change over the scale of the AW thermocline
10. rho at the depth of the Tmin
11. rho at the depth of the Tmax
12. Drho = bulk-scale salinity change over the scale of the AW thermocline
13. DT/DZ = bulk-scale temperature gradient over the scale of the AW thermocline
14. DS/DZ = bulk-scale salinity gradient over the scale of the AW thermocline
15. Drho/DZ = bulk-scale density gradient over the scale of the AW thermocline
16. log10(R_rho_BULK)= log10[ (beta DS/DZ) / (alpha DT/DZ) ]
'''
################################################################
def dfGrouper():
    try:
        with open("test.pkl", "rb") as f:
            dfList = []
            df = pickle.load(f)
            # years = df['date'].apply(lambda d: d.year).unique()
            # years_sorted = np.sort(years)
            # print(years_sorted)

            # yearStart = years_sorted.min()
            # yearLast = years_sorted.max()
            # we skipped the above part since ran it once.
            yearStart, yearLast = 2004, 2023
            # make list of years for grouping
            yearList = np.arange(yearStart, yearLast+1)
            for i in range (0, len(yearList), 4):
                yearGroup = yearList[i:i+4]
                print(yearGroup)
                yeardf = df[df['date'].apply(lambda d: d.year).isin(yearGroup)].copy()
                dfList.append(yeardf)
            return dfList
    except Exception as e:
        traceback.print_exc()
#########################################################
# # Run this only once to save works:
# yearDF = dfGrouper()
# with open("grouped.pkl", 'wb') as f:
#     pickle.dump(yearDF,f)
#########################################################
with open('grouped.pkl', 'rb') as f:
    groupedYears = pickle.load(f)

'''
'''

def histogramPlot(data, groupNum, description, filename):
    plt.figure(figsize=(10, 6))
    counts, _, patches =plt.hist(data, bins=70, alpha=0.5, color='red', 
                                    label=f'{description} per Profile in year Group {groupNum},  total {len(data)} observations')
    plt.axvline(np.median(data), color='k', linestyle='dashed', linewidth=1)
    plt.text(np.median(data), plt.ylim()[1]*0.9, f'Median: {np.median(data):.2f}', va='top', ha='right')

    for count, patch in zip(counts, patches):
                # omit 0 so the plot looks better
                if count != 0:
                    plt.text(patch.get_x() + patch.get_width()/2, count, int(count),
                    ha='center', va='bottom', fontsize=8)
    plt.xlabel(f'{description} per Profile')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {description} per Profile')
    plt.legend()
    plt.savefig(f"plots/{filename}G{groupNum}")

def singleGroupBulkPlot(df, groupNum):
    print(f'-------Processing Group{groupNum}------------------')
    df['year'] = df['date'].apply(lambda d: d.year)
    # index min temp for every profile in every year:
    idx_min_temp = df.groupby(['year','profileNum'])['temp'].idxmin()
    # index max temp for every profile in every year
    idx_max_temp = df.groupby(['year','profileNum'])['temp'].idxmax()
    
    # depth at Tmin:
    min_temp_depth = df.loc[idx_min_temp, 'depth'].values
    histogramPlot(min_temp_depth, groupNum, "Depth at Tmin", "TminDepth")

    # depth at Tmax:
    max_temp_depth = df.loc[idx_max_temp, 'depth'].values
    # histogramPlot(max_temp_depth, groupNum,"Depth at Tmax", "TmaxDepth" )

    # DZ:
    dz = max_temp_depth - min_temp_depth
    # plt.figure(figsize=(10, 6))
    # counts, _, patches =plt.hist(dz, bins=70, alpha=0.5, color='red', 
    #                                 label=f'Thickness of the AW thermocline in Group {groupNum},  total {len(dz)} observations')
    # plt.axvline(np.median(dz), color='k', linestyle='dashed', linewidth=1)
    # plt.text(np.median(dz), plt.ylim()[1]*0.9, f'Median: {np.median(dz):.2f}', va='top', ha='right')

    # for count, patch in zip(counts, patches):
    #             # omit 0 so the plot looks better
    #             if count != 0:
    #                 plt.text(patch.get_x() + patch.get_width()/2, count, int(count),
    #                 ha='center', va='bottom', fontsize=8)
    # plt.xlabel('Thickness of the AW thermocline')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Thickness of the AW thermocline per Profile')
    # plt.legend()
    # plt.savefig(f"plots/ThicknessG{groupNum}")

    # T at the depth of the Tmin:
    min_temp = df.loc[idx_min_temp, 'temp'].values
    # plt.figure(figsize=(10, 6))
    # counts, _, patches =plt.hist(min_temp, bins=70, alpha=0.5, color='red', 
    #                                 label=f'Tmin in Group {groupNum},  total {len(min_temp)} observations')
    # plt.axvline(np.median(min_temp), color='k', linestyle='dashed', linewidth=1)
    # plt.text(np.median(min_temp), plt.ylim()[1]*0.9, f'Median: {np.median(min_temp):.2f}', va='top', ha='right')

    # for count, patch in zip(counts, patches):
    #             # omit 0 so the plot looks better
    #             if count != 0:
    #                 plt.text(patch.get_x() + patch.get_width()/2, count, int(count),
    #                 ha='center', va='bottom', fontsize=8)
    # plt.xlabel('Minimum Temperature')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Tmin per Profile')
    # plt.legend()
    # plt.savefig(f"plots/TminG{groupNum}")

    # T at the depth of the Tmax:
    max_temp = df.loc[idx_max_temp, 'temp'].values
    # plt.figure(figsize=(10, 6))
    # counts, _, patches =plt.hist(max_temp, bins=70, alpha=0.5, color='red', 
    #                                 label=f'Tmax in Group {groupNum},  total {len(max_temp)} observations')
    # plt.axvline(np.median(max_temp), color='k', linestyle='dashed', linewidth=1)
    # plt.text(np.median(max_temp), plt.ylim()[1]*0.9, f'Median: {np.median(max_temp):.2f}', va='top', ha='right')

    # for count, patch in zip(counts, patches):
    #             # omit 0 so the plot looks better
    #             if count != 0:
    #                 plt.text(patch.get_x() + patch.get_width()/2, count, int(count),
    #                 ha='center', va='bottom', fontsize=8)
    # plt.xlabel('Maximum Temperature')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Tmax per Profile')
    # plt.legend()
    # plt.savefig(f"plots/TmaxG{groupNum}")

    # DT:
    dt = max_temp - min_temp
    # plt.figure(figsize=(10, 6))
    # counts, _, patches =plt.hist(dt, bins=70, alpha=0.5, color='red', 
    #                                 label=f'dT in Group {groupNum},  total {len(dt)} observations')
    # plt.axvline(np.median(dt), color='k', linestyle='dashed', linewidth=1)
    # plt.text(np.median(dt), plt.ylim()[1]*0.9, f'Median: {np.median(dt):.2f}', va='top', ha='right')

    # for count, patch in zip(counts, patches):
    #             # omit 0 so the plot looks better
    #             if count != 0:
    #                 plt.text(patch.get_x() + patch.get_width()/2, count, int(count),
    #                 ha='center', va='bottom', fontsize=8)
    # plt.xlabel('Bulk-scale temperature change')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Bulk-scale Temperature Change per Profile')
    # plt.legend()
    # plt.savefig(f"plots/dTG{groupNum}")    

    #Salinity at Tmin:
    min_temp_sal = df.loc[idx_min_temp, 'salinity'].values
    # plt.figure(figsize=(10, 6))
    # counts, _, patches =plt.hist(min_temp_sal, bins=70, alpha=0.5, color='red', 
    #                                 label=f'Tmin Salinity in Group {groupNum},  total {len(min_temp_sal)} observations')
    # plt.axvline(np.median(min_temp_sal), color='k', linestyle='dashed', linewidth=1)
    # plt.text(np.median(min_temp_sal), plt.ylim()[1]*0.9, f'Median: {np.median(min_temp_sal):.2f}', va='top', ha='right')

    # for count, patch in zip(counts, patches):
    #             # omit 0 so the plot looks better
    #             if count != 0:
    #                 plt.text(patch.get_x() + patch.get_width()/2, count, int(count),
    #                 ha='center', va='bottom', fontsize=8)
    # plt.xlabel('Salinity at Tmin')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Salinity at Tmin per Profile')
    # plt.legend()
    # plt.savefig(f"plots/TminSalG{groupNum}")  

    #Salinity at Tmin:
    max_temp_sal = df.loc[idx_max_temp, 'salinity'].values
    # plt.figure(figsize=(10, 6))
    # counts, _, patches =plt.hist(max_temp_sal, bins=70, alpha=0.5, color='red', 
    #                                 label=f'Tmax Salinity in Group {groupNum},  total {len(max_temp_sal)} observations')
    # plt.axvline(np.median(max_temp_sal), color='k', linestyle='dashed', linewidth=1)
    # plt.text(np.median(max_temp_sal), plt.ylim()[1]*0.9, f'Median: {np.median(max_temp_sal):.2f}', va='top', ha='right')

    # for count, patch in zip(counts, patches):
    #             # omit 0 so the plot looks better
    #             if count != 0:
    #                 plt.text(patch.get_x() + patch.get_width()/2, count, int(count),
    #                 ha='center', va='bottom', fontsize=8)
    # plt.xlabel('Salinity at Tmax')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Salinity at Tmax per Profile')
    # plt.legend()
    # plt.savefig(f"plots/TmaxSalG{groupNum}")  


    # bulk-scale salinity change over the scale of the AW thermocline
    ds = max_temp_sal-min_temp_sal
    # plt.figure(figsize=(10, 6))
    # counts, _, patches =plt.hist(ds, bins=70, alpha=0.5, color='red', 
    #                                 label=f'Bulk-scale Salinity Change in Group {groupNum},  total {len(ds)} observations')
    # plt.axvline(np.median(ds), color='k', linestyle='dashed', linewidth=1)
    # plt.text(np.median(ds), plt.ylim()[1]*0.9, f'Median: {np.median(ds):.2f}', va='top', ha='right')

    # for count, patch in zip(counts, patches):
    #             # omit 0 so the plot looks better
    #             if count != 0:
    #                 plt.text(patch.get_x() + patch.get_width()/2, count, int(count),
    #                 ha='center', va='bottom', fontsize=8)
    # plt.xlabel('Bulk-scale Salinity Change')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Bulk-scale Salinity Change per Profile')
    # plt.legend()
    # plt.savefig(f"plots/dSG{groupNum}")     


    # rho at Tmin:
    min_temp_rho = df.loc[idx_min_temp, 'R_rho'].values
    # plt.figure(figsize=(10, 6))
    # counts, _, patches =plt.hist(min_temp_rho, bins=100, alpha=0.5, color='red', 
    #                                 label=f'Tmin Density Ratio in Group {groupNum},  total {len(min_temp_rho)} observations')
    # plt.axvline(np.median(min_temp_rho), color='k', linestyle='dashed', linewidth=1)
    # plt.text(np.median(min_temp_rho), plt.ylim()[1]*0.9, f'Median: {np.median(min_temp_rho):.2f}', va='top', ha='right')

    # for count, patch in zip(counts, patches):
    #             # omit 0 so the plot looks better
    #             if count != 0:
    #                 plt.text(patch.get_x() + patch.get_width()/2, count, int(count),
    #                 ha='center', va='bottom', fontsize=8)
    # plt.xlabel('R_rho at Tmin')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of R_rho at Tmin per Profile')
    # plt.legend()
    # plt.savefig(f"plots/TminRhoG{groupNum}")  
    

    # rho at Tmax:
    max_temp_rho = df.loc[idx_max_temp, 'R_rho'].values
    max_temp_rho_clean = max_temp_rho[np.isfinite(max_temp_rho)]
    # plt.figure(figsize=(10, 6))
    # counts, _, patches =plt.hist(max_temp_rho_clean, bins=100, alpha=0.5, color='red', 
    #                                 label=f'Tmax Density Ratio in Group {groupNum},  total {len(max_temp_rho_clean)} observations')
    # plt.axvline(np.median(max_temp_rho_clean), color='k', linestyle='dashed', linewidth=1)
    # plt.text(np.median(max_temp_rho_clean), plt.ylim()[1]*0.9, f'Median: {np.median(max_temp_rho_clean):.2f}', va='top', ha='right')
    # for count, patch in zip(counts, patches):
    #             # omit 0 so the plot looks better
    #             if count != 0:
    #                 plt.text(patch.get_x() + patch.get_width()/2, count, int(count),
    #                 ha='center', va='bottom', fontsize=8)
    # plt.xlabel('R_rho at Tmax')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of R_rho at Tmax per Profile')
    # plt.legend()
    # plt.savefig(f"plots/TmaxRhoG{groupNum}")  

    
    # filter for rhos
    mask = np.isfinite(max_temp_rho) & np.isfinite(min_temp_rho)
    max_temp_rho_clean = max_temp_rho[mask]
    min_temp_rho_clean = min_temp_rho[mask]

    # DRho:
    dRho = max_temp_rho_clean - min_temp_rho_clean
    # plt.figure(figsize=(10, 6))
    # counts, _, patches =plt.hist(dRho, bins=100, alpha=0.5, color='red', 
    #                                 label=f'Bulk-scale Density Ratio Change in Group {groupNum},  total {len(dRho)} observations')
    # plt.axvline(np.median(dRho), color='k', linestyle='dashed', linewidth=1)
    # plt.text(np.median(dRho), plt.ylim()[1]*0.9, f'Median: {np.median(dRho):.2f}', va='top', ha='right')
    # for count, patch in zip(counts, patches):
    #             # omit 0 so the plot looks better
    #             if count != 0:
    #                 plt.text(patch.get_x() + patch.get_width()/2, count, int(count),
    #                 ha='center', va='bottom', fontsize=8)
    # plt.xlabel('Bulk-scale Density Ratio Change')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Bulk-scale Density Ratio Change per Profile')
    # plt.legend()
    # plt.savefig(f"plots/dRhoG{groupNum}")  

    #dT/dZ:
    temp_gradient = dt/dz
    # plt.figure(figsize=(10, 6))
    # counts, _, patches =plt.hist(temp_gradient, bins=100, alpha=0.5, color='red', 
    #                                 label=f'Bulk-scale Temperature Gredient in Group {groupNum},  total {len(temp_gradient)} observations')
    # plt.axvline(np.median(temp_gradient), color='k', linestyle='dashed', linewidth=1)
    # plt.text(np.median(temp_gradient), plt.ylim()[1]*0.9, f'Median: {np.median(temp_gradient):.2f}', va='top', ha='right')
    # for count, patch in zip(counts, patches):
    #             # omit 0 so the plot looks better
    #             if count != 0:
    #                 plt.text(patch.get_x() + patch.get_width()/2, count, int(count),
    #                 ha='center', va='bottom', fontsize=8)
    # plt.xlabel('Bulk-scale Temperature Gredient')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Bulk-scale Temperature Gredient per Profile')
    # plt.legend()
    # plt.savefig(f"plots/tempGradG{groupNum}")  


    sal_gradient = ds/dz
    # plt.figure(figsize=(10, 6))
    # counts, _, patches =plt.hist(sal_gradient, bins=100, alpha=0.5, color='red', 
    #                                 label=f'Bulk-scale Salinity Gredient in Group {groupNum},  total {len(sal_gradient)} observations')
    # plt.axvline(np.median(sal_gradient), color='k', linestyle='dashed', linewidth=1)
    # plt.text(np.median(sal_gradient), plt.ylim()[1]*0.9, f'Median: {np.median(sal_gradient):.2f}', va='top', ha='right')
    # for count, patch in zip(counts, patches):
    #             # omit 0 so the plot looks better
    #             if count != 0:
    #                 plt.text(patch.get_x() + patch.get_width()/2, count, int(count),
    #                 ha='center', va='bottom', fontsize=8)
    # plt.xlabel('Bulk-scale Salinity Gredient')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Bulk-scale Salinity Gredient per Profile')
    # plt.legend()
    # plt.savefig(f"plots/salGradG{groupNum}")  

    dz_clean = dz[mask]
    # rho_gradient = dRho/dz_clean
    # plt.figure(figsize=(10, 6))
    # counts, _, patches =plt.hist(rho_gradient, bins=100, alpha=0.5, color='red', 
    #                                 label=f'Bulk-scale R_rho Gredient in Group {groupNum},  total {len(rho_gradient)} observations')
    # plt.axvline(np.median(rho_gradient), color='k', linestyle='dashed', linewidth=1)
    # plt.text(np.median(rho_gradient), plt.ylim()[1]*0.9, f'Median: {np.median(rho_gradient):.2f}', va='top', ha='right')
    # for count, patch in zip(counts, patches):
    #             # omit 0 so the plot looks better
    #             if count != 0:
    #                 plt.text(patch.get_x() + patch.get_width()/2, count, int(count),
    #                 ha='center', va='bottom', fontsize=8)
    # plt.xlabel('Bulk-scale R_rho Gredient')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Bulk-scale R_rho Gredient per Profile')
    # plt.legend()
    # plt.savefig(f"plots/rhoGradG{groupNum}")  

    # Bulk Rho:
#####################################################################################
for i in range(5):
     singleGroupBulkPlot(groupedYears[i], i)

def boxPlots(df):
    n_cols = 5
    n_rows = 1

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axs = axs.flatten() 

    all_depths = []  # To store depths for global min/max

    # First loop: Collect depths
    for i in range(5):
        df_copy = df[i].copy()
        df_copy['year'] = df_copy['date'].apply(lambda d: d.year)
        idx_min_temp = df_copy.groupby(['year','profileNum'])['temp'].idxmin()
        min_temp_depth = df_copy.loc[idx_min_temp, 'depth'].values
        all_depths.append(min_temp_depth)

    # Compute global min and max
    global_min_depth = min(np.nanmin(depths) for depths in all_depths)
    global_max_depth = max(np.nanmax(depths) for depths in all_depths)

    # Second loop: Plot boxplots with same y-axis limits
    for i in range(5):
        ax = axs[i]
        ax.boxplot(all_depths[i])
        ax.set_title(f"Boxplot for depth at Tmin, group{i}")
        ax.set_xlabel('Group Number')
        ax.set_ylabel('Depth (m)')
        ax.set_ylim(global_max_depth, global_min_depth)  # Invert y-axis (depth increases downward)

    plt.tight_layout()
    plt.show()


# boxPlots(groupedYears)
