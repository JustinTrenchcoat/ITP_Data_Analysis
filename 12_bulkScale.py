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
BULK SCALE PROPERTIES (i.e. properties over the scale of the AW thermocline as a whole; there is one bulk-scale property per profile)
1. the histogram of all estimates of property X for time period Y. 
2. a figure of X rows that show time series of “box and whisker plots” 
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
# Run this only once to save works:
# yearDF = dfGrouper()
# with open("grouped.pkl", 'wb') as f:
#     pickle.dump(yearDF,f)
#########################################################
# with open('grouped.pkl', 'rb') as f:
#     groupedYears = pickle.load(f)
'''
Tiny Helper function(s?) for saving space
'''
############################################################################################################################################
def histogramPlot(data, groupNum, description, filename):
    plt.figure(figsize=(10, 6))
    counts, _, patches =plt.hist(data, bins=70, alpha=0.5, color='red', 
                                    label=f'{description} per Profile in year Group {groupNum},  total {len(data)} profiles')
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
    plt.savefig(f"plots/bulk/Histograms/{filename}G{groupNum}")
    plt.close()
##########################################################################################################################################
def singleGroupBulkPlot(df, groupNum):
    print(f'-------Processing Group{groupNum}------------------')
    # index min temp for every profile in every year:
    idx_min_temp = df.groupby(["date",'systemNum','profileNum'])['temp'].idxmin()
    # index max temp for every profile in every year
    idx_max_temp = df.groupby(["date",'systemNum','profileNum'])['temp'].idxmax()
    
    # depth at Tmin:
    min_temp_depth = df.loc[idx_min_temp, 'depth'].values
    # histogramPlot(min_temp_depth, groupNum, "Depth at Tmin", "TminDepth")

    # depth at Tmax:
    max_temp_depth = df.loc[idx_max_temp, 'depth'].values
    # histogramPlot(max_temp_depth, groupNum,"Depth at Tmax", "TmaxDepth" )

    # DZ:
    dz = max_temp_depth - min_temp_depth
    # histogramPlot(dz, groupNum, "Thickness of the AW thermocline", "Thickness")

    # T at the depth of the Tmin:
    min_temp = df.loc[idx_min_temp, 'temp'].values
    # histogramPlot(min_temp, groupNum, "Minimum Temperature", "Tmin" )

    # T at the depth of the Tmax:
    max_temp = df.loc[idx_max_temp, 'temp'].values
    # histogramPlot(max_temp, groupNum, "Maximum Temperature", "Tmax")

    # DT:
    dt = max_temp - min_temp
    histogramPlot(dt, groupNum, "Bulk-scale temperature change", "dT")   

    # Salinity at Tmin:
    min_temp_sal = df.loc[idx_min_temp, 'salinity'].values
    histogramPlot(min_temp_sal, groupNum, "Salinity at Tmin", "TminSal")

    #Salinity at Tmin:
    max_temp_sal = df.loc[idx_max_temp, 'salinity'].values
    histogramPlot(max_temp_sal, groupNum, "Salinity at Tmax", "TmaxSal")

    # bulk-scale salinity change over the scale of the AW thermocline
    ds = max_temp_sal-min_temp_sal
    histogramPlot(ds, groupNum, "Bulk-scale Salinity Change", "dS")     


    # rho at Tmin:
    min_temp_rho = df.loc[idx_min_temp, 'R_rho'].values
    max_temp_rho = df.loc[idx_max_temp, 'R_rho'].values

    mask = np.isfinite(max_temp_rho) & np.isfinite(min_temp_rho)
    max_temp_rho_clean = max_temp_rho[mask]
    min_temp_rho_clean = min_temp_rho[mask]

    # max_temp_rho_clean = np.log10(max_temp_rho_clean)
    # min_temp_rho_clean = np.log10(min_temp_rho_clean)
    histogramPlot(min_temp_rho_clean, groupNum, "Tmin Stability Ratio", "TminRho")
    histogramPlot(max_temp_rho_clean, groupNum, "Tmax Stability Ratio", "TmaxRho")

    # DRho:
    dRho = max_temp_rho_clean - min_temp_rho_clean
    histogramPlot(dRho, groupNum, "Bulk-scale Stability Ratio Change", "dRho")

    #dT/dZ:
    temp_gradient = dt/dz
    histogramPlot(temp_gradient, groupNum, "Bulk-scale Temperature Gredient", "tempGrad")


    sal_gradient = ds/dz
    histogramPlot(sal_gradient, groupNum, "Bulk-scale Salinity Gredient", "salGrad")  

    dz_clean = dz[mask]
    rho_gradient = dRho/dz_clean
    histogramPlot(rho_gradient, groupNum, "Bulk-scale R_rho Gredient", "rhoGrad")

    # Bulk Rho:
    beta = []
    alpha = []
    temps = [group.values for _, group in df.groupby(['date',"systemNum",'profileNum'])['temp']]
    sals = [group.values for _, group in df.groupby(['date',"systemNum",'profileNum'])['salinity']]
    pres = [group.values for _, group in df.groupby(['date',"systemNum",'profileNum'])['pressure']]

    for i in range(len(temps)):
        beta.append(np.mean(gsw.beta(sals[i], temps[i], pres[i])))
        alpha.append(np.mean(gsw.alpha(sals[i], temps[i], pres[i])))
    
    rho_bulk = np.log10((beta*np.absolute(sal_gradient))/(alpha*np.absolute(temp_gradient)))
    rho_bulk = rho_bulk[np.isfinite(rho_bulk)]
    histogramPlot(rho_bulk, groupNum, "Bulk-scale Stability Ratio", "rho")
# for i in range(5):
#      singleGroupBulkPlot(groupedYears[i], i)
#############################################################################################################################################
def singleBoxplot(df_list, compute_depth_fn, title_prefix, y_label, y_limits, save_path):

    all_values = []
    all_groups = []

    for i, df_group in enumerate(df_list):
        print(f'-------Processing Group {i}------------------')
        df_copy = df_group.copy()
        values = compute_depth_fn(df_copy)
        all_values.extend(values)
        all_groups.extend([i] * len(values))  # i represents group number

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.boxplot(
        [ [val for val, g in zip(all_values, all_groups) if g == i] for i in sorted(set(all_groups)) ],
        labels=sorted(set(all_groups))
    )

    plt.title(f"Boxplot of {title_prefix} by Group")
    plt.xlabel('Group Number')
    plt.ylabel(y_label)
    plt.ylim(y_limits)
    plt.tight_layout()
    plt.savefig(f"plots/bulk/Boxplots/{save_path}")
    plt.close()
#######################################################################################################################
def boxPlots(df):
    # First loop: Collect depths
    def TminD(df):
        idx_min_temp = df.groupby(['date',"systemNum",'profileNum'])['temp'].idxmin()
        min_temp_depth = df.loc[idx_min_temp, 'depth'].values
        return min_temp_depth
         
    # singleBoxplot(df, TminD, "depth at Tmin", "Depth (m)", (275,75),"TminDepthBox")

    def TmaxD(df):
        idx_max_temp = df.groupby(['date',"systemNum",'profileNum'])['temp'].idxmax()
        max_temp_depth = df.loc[idx_max_temp, 'depth'].values
        return max_temp_depth
    # singleBoxplot(df, TmaxD, "depth at Tmax","Depth (m)", (575,325), "TmaxDepthBox")

    def dZ(df):
        dz = TmaxD(df) - TminD(df)
        return dz
    # singleBoxplot(df, dZ, "Thickness of Thermocline", "Thickness (m)", (150,375), "ThicknessBox")

    def minTemp(df):
        idx_min_temp = df.groupby(['date',"systemNum",'profileNum'])['temp'].idxmin()
        min_temp = df.loc[idx_min_temp, 'temp'].values
        return min_temp
    # singleBoxplot(df, minTemp, "Minimum Temperature", "Celcius",(-1.7,-1.3), "TminBox")

    def maxTemp(df):
         idx_max_temp = df.groupby(['date',"systemNum",'profileNum'])['temp'].idxmax()
         max_temp = df.loc[idx_max_temp, 'temp'].values
         return max_temp
    # singleBoxplot(df, maxTemp, "Maximum Temperature", "Celcius", (0.55,1.15), "TmaxBox")

    def dT(df):
        dt = maxTemp(df) - minTemp(df)
        return dt
    # singleBoxplot(df, dT, "Bulk-scale temperature change", "Celcius", (2,2.7), "dTBox")

    def minSal(df):
        idx_min_temp = df.groupby(['date',"systemNum",'profileNum'])['temp'].idxmin()
        min_sal = df.loc[idx_min_temp, "salinity"].values
        return min_sal
    # singleBoxplot(df, minSal, "Salinity at Minimum Temperature", "g/kg",(31.75,34.3), "TminSalBox")

    def maxSal(df):
        idx_max_temp = df.groupby(['date',"systemNum",'profileNum'])['temp'].idxmax()
        max_sal = df.loc[idx_max_temp, 'salinity'].values
        return max_sal     
    # singleBoxplot(df, maxSal, "Salinity at Maximum Temperature", "g/kg", (34.96,35.04), "TmaxSalBox")  

    def dS(df):
        ds = maxSal(df) - minSal(df)
        return ds
    # singleBoxplot(df, dS, "Bulk-scale Salinity Change", "g/kg", (0.75,3.25), "dSBox")

    def minRho(df):
        idx_min_temp = df.groupby(['date',"systemNum",'profileNum'])['temp'].idxmin()

        min_temp_rho = df.loc[idx_min_temp, 'R_rho'].values
        
        return min_temp_rho
    singleBoxplot(df, minRho, "Stability Ratio at Tmin", " ", (-5,40), "TminRhoBox")

    def maxRho(df):
        idx_max_temp = df.groupby(['date',"systemNum",'profileNum'])['temp'].idxmax()

        max_temp_rho = df.loc[idx_max_temp, 'R_rho'].values

        return max_temp_rho
    singleBoxplot(df, maxRho, "Stability Ratio at Tmax", " ", (-2.5,7.5), "TmaxRhoBox")

    def dRho(df):
        idx_min_temp = df.groupby(['date',"systemNum",'profileNum'])['temp'].idxmin()
        idx_max_temp = df.groupby(['date',"systemNum",'profileNum'])['temp'].idxmax()

        min_temp_rho = df.loc[idx_min_temp, 'R_rho'].values
        max_temp_rho = df.loc[idx_max_temp, 'R_rho'].values

        mask = np.isfinite(max_temp_rho) & np.isfinite(min_temp_rho)
        min_temp_rho_clean = min_temp_rho[mask]
        max_temp_rho_clean = max_temp_rho[mask]

        return max_temp_rho_clean - min_temp_rho_clean
    singleBoxplot(df, dRho, "Bulk-scale Stability Ratio Change","", (-40,10), "dRhoBox" )

    def dTdZ(df):
        return dT(df)/dZ(df)
    # singleBoxplot(df, dTdZ, "Bulk-Scale Temperature Gradient", "Celcius/m", (0.005,0.014),"dTdZBox")

    def dSdZ(df):
        return dS(df)/dZ(df)
    # singleBoxplot(df, dSdZ, "Bulk-Scale Salinity Gradient", "(g/kg)/m", (0.004,0.011), "dSdZBox")

    def dRhodZ(df):
        return dRho(df)/dZ(df)
    singleBoxplot(df, dRhodZ, "Bulk-Scale Stability Ratio Gradient", "", (-0.2,0.075),"dRhodZBox")

    def bulkRho(df):
        # Bulk Rho:
        beta = []
        alpha = []
        temps = [group.values for _, group in df.groupby(['date',"systemNum",'profileNum'])['temp']]
        # print(f"# of profiles:{len(temps)}")
        sals = [group.values for _, group in df.groupby(['date',"systemNum",'profileNum'])['salinity']]
        pres = [group.values for _, group in df.groupby(['date',"systemNum",'profileNum'])['pressure']]
        
        for i in range(len(temps)):
            beta.append(np.mean(gsw.beta(sals[i], temps[i], pres[i])))
            alpha.append(np.mean(gsw.alpha(sals[i], temps[i], pres[i])))
        # added absolute to ensure log10 process is not broken
        rho_bulk = np.log10((beta*np.absolute(dS(df)))/(alpha*np.absolute(dT(df))))
        rho_bulk = rho_bulk[np.isfinite(rho_bulk)]
        return rho_bulk
    singleBoxplot(df,bulkRho, "Bulk-scale Stability Ratio", "", (0.8,1.35), "bulkRhoBox")


# boxPlots(groupedYears)


# def tester(df, groupNum):
#     print(f'-------Processing Group{groupNum}------------------')
#     df['year'] = df['date'].apply(lambda d: d.year)
#     # index min temp for every profile in every year:
#     idx_min_temp = df.groupby(['date','profileNum'])['temp'].idxmin()
#     print(len(idx_min_temp))
# for i in range(5):
#      tester(groupedYears[i], i)