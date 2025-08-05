import numpy as np
import pandas as pd
import traceback
from helper import *
from scipy.io import loadmat

# this script would serve as a profile checker for you, the user, to check what went wrong when there is a warning message

# set up file path, subject to change
full_path = r'D:\EOAS\ITP_Data_Analysis\gridDataMat\itp103cormat\cor1207.mat'

# print out stats of the profile and make basic plots for you to inspect
try:
    data = loadmat(full_path)
    
    # add or take out variables depend on your need
    depth = data['Depth'].squeeze()
    lat = data['lat'].squeeze()
    lon = data['lon'].squeeze()
    psdate = data['startDate'].squeeze()
    date = pd.to_datetime(psdate, format='%m/%d/%y')
    print(date)
    date = date.date
    temp = data['Temperature'].squeeze()
    salinity = data['Salinity'].squeeze()
    pres = pressure(depth, lat)

    assert not np.isnan(depth).any(), "depth contains NaN values"
    assert not np.isnan(temp).any(), "temp contains NaN values"
    assert not np.isnan(salinity).any(), "salinity contains NaN values"
    assert not np.isnan(pres).any(), "pres contains NaN values"

    # change the variable based on your need.
    simplePlot(temp, depth)
    printBasicStat(depth)

except Exception as e:
    print(f"Error processing file: {full_path}")
    traceback.print_exc()
