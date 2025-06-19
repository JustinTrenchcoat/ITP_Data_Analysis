import pandas as pd
from helper import *
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import xarray as xr

'''
In the dataset:
every row is one observation from one profile.
It has to be ordered in time so that time series analysis would work.



thermal expansion coef: alpha

haline contraction coef: beta

Nsquared


Numerical features:
Depth
Temperature
Salinity


Categorical features:
Mixed layer
interface layer


Target?
Staircase types: Sharp Mushy SuperMushy


Time analysis variable:
Date
(I doubt if the exact time would be a variable??)
##############################################
background features equations:

density:
density stratification (N2)
desnity gradient ratio (R_rho)


'''

numeric_features = [
    "Temp",
    "Salinity",
    "Depth",
    "Density",
    "Density_N2",
    "R_rho",
]
categorical_features = [
    "Mixed",
    "Interface",
]
time_feature = ["Date"]
target = ["StaircaseType"]

# Read in the data:


# Configuration
file_path = 'itp100cormat.nc' 
# this has shape of 189, 8000
# meaning that each first dimension row is a profile, with 8000 measurements
# 
ds = nc.Dataset(file_path)
# print(ds.dimensions)
# print(ds.variables)


with ds as dataset:
    # extract variables:
    # the prof is not profile number, but index. FloatID true profile number from each ITP system
    # this would not be used as an input variable
    profN = dataset.variables['FloatID'][:]
    # the nc file mistakenly wrote pressure instead of depth
    depth = dataset.variables['pressure'][:]
    temp = dataset.variables["ct"][:]
    connect_layer_mask = dataset.variables['mask_cl'][:]
    dates = dataset.variables["dates"][:]
date = pd.to_datetime(dates, unit = 's')

# how to store such that every entry is an array for depth, a dictionary? for temp
ocean_df = pd.DataFrame({
    "profile_number" : profN,
    'depth' : [d for d in depth],
    'temp' : [t for t in temp],
    'date' : date.date
})
print(ocean_df.shape)
print(ocean_df.loc[0:10])
test_frame = ocean_df.loc[0]['temp']
print(len(test_frame))
print(len(ocean_df.loc[0]['depth']))



data_dir = 'stairs'
ocean_df = pd.DataFrame()
# read data and put everything into ocean_df
def read_data(full_path, file_name, folder_name):
    profile_df = pd.read_csv(full_path)
    ocean_df= pd.concat([ocean_df, profile_df], ignore_index=True)

traverse_datasets(data_dir, read_data)
# Now ocean_df should have everything

# we now sort the dataframe in order of time:
ocean_sorted_df = ocean_df.sort_values(by='startDate')







