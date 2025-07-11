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
def singleGroupHandeler(df, groupNum):
    print(f'-------Processing Group{groupNum}------------------')
    df['year'] = df['date'].apply(lambda d: d.year)
    # index min temp for every profile in every year:
    idx_min_temp = df.groupby(['year','profileNum'])['temp'].idxmin()
    # index max temp for every profile in every year
    idx_max_temp = df.groupby(['year','profileNum'])['temp'].idxmax()
    
    # depth at Tmin:
    min_temp_depth = df.loc[idx_min_temp, 'depth'].values
    # plt.figure(figsize=(10, 6))
    # counts, _, patches =plt.hist(min_temp_depth, bins=70, alpha=0.5, color='red', 
    #                                 label=f'Depth at Tmin per Profile in year Group {groupNum}, total {len(min_temp_depth)} observations')
    # plt.axvline(np.median(min_temp_depth), color='k', linestyle='dashed', linewidth=1)
    # plt.text(np.median(min_temp_depth), plt.ylim()[1]*0.9, f'Median: {np.median(min_temp_depth):.2f}', va='top', ha='right')

    # for count, patch in zip(counts, patches):
    #             # omit 0 so the plot looks better
    #             if count != 0:
    #                 plt.text(patch.get_x() + patch.get_width()/2, count, int(count),
    #                 ha='center', va='bottom', fontsize=8)

    # plt.xlabel('Depth at Tmin per Profile')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Depth at Tmin per Profile')
    # plt.legend()
    # # plt.savefig(f"plots/TminG{groupNum}")


    # depth at Tmax:
    max_temp_depth = df.loc[idx_max_temp, 'depth'].values
    # plt.figure(figsize=(10, 6))
    # counts, _, patches =plt.hist(max_temp_depth, bins=70, alpha=0.5, color='red', 
    #                                 label=f'Depth at Tmax per Profile in year Group {groupNum},  total {len(max_temp_depth)} observations')
    # plt.axvline(np.median(max_temp_depth), color='k', linestyle='dashed', linewidth=1)
    # plt.text(np.median(max_temp_depth), plt.ylim()[1]*0.9, f'Median: {np.median(max_temp_depth):.2f}', va='top', ha='right')

    # for count, patch in zip(counts, patches):
    #             # omit 0 so the plot looks better
    #             if count != 0:
    #                 plt.text(patch.get_x() + patch.get_width()/2, count, int(count),
    #                 ha='center', va='bottom', fontsize=8)
    # plt.xlabel('Depth at Tmax per Profile')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Depth at Tmax per Profile')
    # plt.legend()
    # # plt.savefig(f"plots/TmaxG{groupNum}")

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
    min_temp_depth = df.loc[idx_min_temp, 'temp'].values
    # plt.figure(figsize=(10, 6))
    # counts, _, patches =plt.hist(min_temp_depth, bins=70, alpha=0.5, color='red', 
    #                                 label=f'Tmin in Group {groupNum},  total {len(min_temp_depth)} observations')
    # plt.axvline(np.median(min_temp_depth), color='k', linestyle='dashed', linewidth=1)
    # plt.text(np.median(min_temp_depth), plt.ylim()[1]*0.9, f'Median: {np.median(min_temp_depth):.2f}', va='top', ha='right')

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




for i in range(5):
     singleGroupHandeler(groupedYears[i], i)
