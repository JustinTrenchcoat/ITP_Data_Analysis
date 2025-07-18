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
            for i in range (0, len(yearList), 5):
                yearGroup = yearList[i:i+5]
                print(yearGroup)
                yeardf = df[df['date'].apply(lambda d: d.year).isin(yearGroup)].copy()
                dfList.append(yeardf)
            return dfList
    except Exception as e:
        traceback.print_exc()


def test(df_list):
    count_dfs = []
    for i, df_group in enumerate(df_list):
        print(f'-------Processing Group {i}------------------')
        # number of rows for each depth
        observation = df_group.groupby("depth").size()
        obs = df_group.groupby(['date', 'profileNum']).size()
        print(len(obs))
testDF = dfGrouper()
test(testDF)
