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
with open('grouped.pkl', 'rb') as f:
    groupedYears = pickle.load(f)

'''
Tiny Helper function(s?) for saving space
'''

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
    plt.savefig(f"plots/Histograms/{filename}G{groupNum}")
##########################################################################################################################################
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
    histogramPlot(max_temp_depth, groupNum,"Depth at Tmax", "TmaxDepth" )

    # DZ:
    dz = max_temp_depth - min_temp_depth
    histogramPlot(dz, groupNum, "Thickness of the AW thermocline", "Thickness")

    # T at the depth of the Tmin:
    min_temp = df.loc[idx_min_temp, 'temp'].values
    histogramPlot(min_temp, groupNum, "Minimum Temperature", "Tmin" )

    # T at the depth of the Tmax:
    max_temp = df.loc[idx_max_temp, 'temp'].values
    histogramPlot(max_temp, groupNum, "Maximum Temperature", "Tmax")

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

    max_temp_rho_clean = np.log10(max_temp_rho_clean)
    min_temp_rho_clean = np.log10(min_temp_rho_clean)
    histogramPlot(min_temp_rho_clean, groupNum, "Tmin Density Ratio", "TminRho")
    histogramPlot(max_temp_rho_clean, groupNum, "Tmax Density Ratio", "TmaxRho")

    # DRho:
    dRho = max_temp_rho_clean - min_temp_rho_clean
    histogramPlot(dRho, groupNum, "Bulk-scale Density Ratio Change", "dRho")

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
    temps = [group.values for _, group in df.groupby(['year','profileNum'])['temp']]
    sals = [group.values for _, group in df.groupby(['year','profileNum'])['salinity']]
    pres = [group.values for _, group in df.groupby(['year','profileNum'])['pressure']]

    for i in range(len(temps)):
        beta.append(np.mean(gsw.beta(sals[i], temps[i], pres[i])))
        alpha.append(np.mean(gsw.alpha(sals[i], temps[i], pres[i])))
    
    rho_bulk = np.log10((beta*sal_gradient)/(alpha*temp_gradient))
    rho_bulk = rho_bulk[np.isfinite(rho_bulk)]
    histogramPlot(rho_bulk, groupNum, "Bulk-scale Density Ratio", "rho")
##########################################################################################################################################
# for i in range(5):
#      singleGroupBulkPlot(groupedYears[i], i)



def singleBoxplot(df_list, compute_depth_fn, title_prefix, y_label, y_limits, save_path):
    n_cols = 5
    n_rows = 1
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axs = axs.flatten() 

    for i, df_group in enumerate(df_list):
        print(f'-------Processing Group {i}------------------')
        df_copy = df_group.copy()
        df_copy['year'] = df_copy['date'].apply(lambda d: d.year)

        values = compute_depth_fn(df_copy)

        ax = axs[i]
        ax.boxplot(values)
        ax.set_title(f"Boxplot for {title_prefix}, group{i}")
        ax.set_xlabel('Group Number')
        ax.set_ylabel(f'{y_label}')
        ax.set_ylim(y_limits)

    plt.tight_layout()
    plt.savefig(f"plots/Boxplots/{save_path}")
    plt.close()


def boxPlots(df):
    # First loop: Collect depths
    def TminD(df):
        idx_min_temp = df.groupby(['year','profileNum'])['temp'].idxmin()
        min_temp_depth = df.loc[idx_min_temp, 'depth'].values
        return min_temp_depth
         
    singleBoxplot(df, TminD, "depth at Tmin", "Depth (m)", (650,0),"TminDepthBox")

    def TmaxD(df):
        idx_max_temp = df.groupby(['year','profileNum'])['temp'].idxmax()
        max_temp_depth = df.loc[idx_max_temp, 'depth'].values
        return max_temp_depth
    singleBoxplot(df, TmaxD, "depth at Tmax","Depth (m)", (650,300), "TmaxDepthBox")

    def dZ(df):
        dz = TmaxD(df) - TminD(df)
        return dz
    singleBoxplot(df, dZ, "Thickness of Thermocline", "Thickness (m)", (450,0), "ThicknessBox")

    def minTemp(df):
        idx_min_temp = df.groupby(['year','profileNum'])['temp'].idxmin()
        min_temp = df.loc[idx_min_temp, 'temp'].values
        return min_temp
    singleBoxplot(df, minTemp, "Minimum Temperature", "Celcius",(2,-4), "TminBox")

    def maxTemp(df):
         idx_max_temp = df.groupby(['year','profileNum'])['temp'].idxmax()
         max_temp = df.loc[idx_max_temp, 'temp'].values
         return max_temp
    singleBoxplot(df, maxTemp, "Maximum Temperature", "Celcius", (2,0), "TmaxBox")

    def dT(df):
        dt = maxTemp(df) - minTemp(df)
        return dt
    singleBoxplot(df, dT, "Bulk-scale temperature change", "Celcius", (4,-1), "dTBox")

    def minSal(df):
        idx_min_temp = df.groupby(['year','profileNum'])['temp'].idxmin()
        min_sal = df.loc[idx_min_temp, "salinity"].values
        return min_sal
    singleBoxplot(df, minSal, "Salinity at Minimum Temperature", "g/kg",(40,30), "TminSalBox")

    def maxSal(df):
        idx_max_temp = df.groupby(['year','profileNum'])['temp'].idxmax()
        max_sal = df.loc[idx_max_temp, 'salinity'].values
        return max_sal     
    singleBoxplot(df, maxSal, "Salinity at Maximum Temperature", "g/kg", (45, 30), "TmaxSalBox")  

    def dS(df):
        ds = maxSal(df) - minSal(df)
        return ds
    singleBoxplot(df, dS, "Bulk-scale Salinity Change", "g/kg", (5,0), "dSBox")

    def minRho(df):
        idx_min_temp = df.groupby(['year','profileNum'])['temp'].idxmin()
        idx_max_temp = df.groupby(['year','profileNum'])['temp'].idxmax()

        min_temp_rho = df.loc[idx_min_temp, 'R_rho'].values
        max_temp_rho = df.loc[idx_max_temp, 'R_rho'].values

        mask = np.isfinite(max_temp_rho) & np.isfinite(min_temp_rho)
        min_temp_rho_clean = min_temp_rho[mask]
        return min_temp_rho_clean
    singleBoxplot(df, minRho, "Density Gradient at Tmin", "(g/kg)/m", (5,0), "TminRhoBox")

    def maxRho(df):
        idx_min_temp = df.groupby(['year','profileNum'])['temp'].idxmin()
        idx_max_temp = df.groupby(['year','profileNum'])['temp'].idxmax()

        min_temp_rho = df.loc[idx_min_temp, 'R_rho'].values
        max_temp_rho = df.loc[idx_max_temp, 'R_rho'].values

        mask = np.isfinite(max_temp_rho) & np.isfinite(min_temp_rho)
        max_temp_rho_clean = max_temp_rho[mask]
        return max_temp_rho_clean
    singleBoxplot(df, maxRho, "Density Gradient at Tmax", "(g/kg)/m", (5,0), "TmaxRhoBox")

    def dRho(df):
        idx_min_temp = df.groupby(['year','profileNum'])['temp'].idxmin()
        idx_max_temp = df.groupby(['year','profileNum'])['temp'].idxmax()

        min_temp_rho = df.loc[idx_min_temp, 'R_rho'].values
        max_temp_rho = df.loc[idx_max_temp, 'R_rho'].values

        mask = np.isfinite(max_temp_rho) & np.isfinite(min_temp_rho)
        min_temp_rho_clean = min_temp_rho[mask]
        max_temp_rho_clean = max_temp_rho[mask]

        return max_temp_rho_clean - min_temp_rho_clean
    singleBoxplot(df, dRho, "Bulk-scale Density Ratio Change","(g/kg)/m", (5,0), "dRhoBox" )

    def dTdZ(df):
        return dT(df)/dZ(df)
    singleBoxplot(df, dTdZ, "Bulk-Scale Temperature Gradient", "Celcius/m", (1,-1),"dTdZBox")

    def dSdZ(df):
        return dS(df)/dZ(df)
    singleBoxplot(df, dSdZ, "Bulk-Scale Salinity Gradient", "(g/kg)/m", (1,-1), "dSdZBox")

    def dRhodZ(df):
        return dRho(df)/dZ(df)
    singleBoxplot(df, dRhodZ, "Bulk-Scale Density Ratio Gradient", "(g/kg)/m^2", (1,-1),"dRhodZBox")

    def bulkRho(df):
        # Bulk Rho:
        beta = []
        alpha = []
        temps = [group.values for _, group in df.groupby(['year','profileNum'])['temp']]
        # print(f"# of profiles:{len(temps)}")
        sals = [group.values for _, group in df.groupby(['year','profileNum'])['salinity']]
        pres = [group.values for _, group in df.groupby(['year','profileNum'])['pressure']]
        
        for i in range(len(temps)):
            beta.append(np.mean(gsw.beta(sals[i], temps[i], pres[i])))
            alpha.append(np.mean(gsw.alpha(sals[i], temps[i], pres[i])))
    
        rho_bulk = np.log10((beta*dS(df))/(alpha*dT(df)))
        rho_bulk = rho_bulk[np.isfinite(rho_bulk)]
        return rho_bulk
    singleBoxplot(df,bulkRho, "Bulk-scale Density Ratio", "(g/kg)/m", (2,-1), "bulkRhoBox")


boxPlots(groupedYears)
