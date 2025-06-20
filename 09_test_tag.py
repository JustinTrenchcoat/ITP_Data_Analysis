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

file_path = 'tagData/itp100cormat.nc' 
# 
ds = nc.Dataset(file_path)
# print(ds.dimensions)
# print(ds.variables)
ocean_df = pd.DataFrame()
def readNC(full_path,ls):
    ds = nc.Dataset(full_path)
    with ds as dataset:
        # extract variables:
        # the prof is not profile number, but index. FloatID true profile number from each ITP system
        # this would not be used as an input variable
        profN = dataset.variables['FloatID'][:]
        # the nc file mistakenly wrote pressure instead of depth
        depth = dataset.variables['pressure'][:]
        temp = dataset.variables["ct"][:]
        salinity = dataset.variables["sa"][:]
        connect_layer_mask = dataset.variables['mask_cl'][:]
        interface_layer_mask = dataset.variables['mask_int'][:]
        mixed_layer_mask = dataset.variables["mask_ml"][:]
        staircase_mask = dataset.variables["mask_sc"][:]
        dates = dataset.variables["dates"][:]
        lon = dataset.variables["lon"][:]
        lat = dataset.variables["lat"][:]
        date = pd.to_datetime(dates, unit = 's')
        date = date.date
        for i in range(len(date)):
            mask_cl = connect_layer_mask[i]
            mask_int = interface_layer_mask[i]
            mask_ml = mixed_layer_mask[i]
            mask_sc = staircase_mask[i]
            new_df = pd.DataFrame({
                "profileNumber" : profN[i],
                "depth" : depth[i],
                'temp' : temp[i],
                'date' : date[i],
                "salinity" : salinity[i],
                'mask_cl' : mask_cl,
                'mask_int' : mask_int,
                'mask_ml' : mask_ml,
                "mask_sc" : mask_sc,
                "lon" : lon[i],
                "lat" : lat[i]
            })
            ls.append(new_df)
    return ls

# we now sort the dataframe in order of time:
tagData_dir = 'tagData'
df_list = []
for fileName in tqdm(sorted(os.listdir(tagData_dir)), desc="Processing files"):
    full_path = os.path.join(tagData_dir, fileName)
    df_list = readNC(full_path, df_list)


final_df = pd.concat(df_list, ignore_index=True)
# add a save to pickle?
print(final_df.head())
filtered_df = final_df[final_df['mask_cl'].notna()]
print(filtered_df)
ocean_sorted_df = final_df.sort_values(by='date')
print(f'sorted DF: \n{ocean_sorted_df.head()}')


# data_dir = 'stairs'
# ocean_df = pd.DataFrame()
# # read data and put everything into ocean_df
# def read_data(full_path, file_name, folder_name):
#     profile_df = pd.read_csv(full_path)
#     ocean_df= pd.concat([ocean_df, profile_df], ignore_index=True)

# traverse_datasets(data_dir, readNC)
# # Now ocean_df should have everything

# # we now sort the dataframe in order of time:
# ocean_sorted_df = ocean_df.sort_values(by='startDate')







