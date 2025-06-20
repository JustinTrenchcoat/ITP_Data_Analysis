import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import xarray as xr
import pandas as pd
from helper import *

# Configuration
file_path = 'itp100cormat.nc' 
# 
ds = nc.Dataset(file_path)
# print(ds.dimensions)
# print(ds.variables)

ocean_df = pd.DataFrame()
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
    for i in range(2):
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
        ocean_df = pd.concat([ocean_df, new_df])
        print(f"test dataframe shape: \n{ocean_df.shape}")
        print(f'test dateframe: \n{new_df.head()}')
        # helPlot(new_df["temp"], new_df["depth"])

    
# how to store such that every entry is an array for depth, a dictionary? for temp


# we now sort the dataframe in order of time:
ocean_sorted_df = ocean_df.sort_values(by='date')
print(f'sorted DF: \n{ocean_sorted_df.head()}')


# itp = 'ITP65'  # Update with your ITP name if needed

# prof_no = 492              # Profile index to plot

# profile_ids = ds.variables['FloatID'][:]  # 1D array of profile IDs
# if prof_no in profile_ids:
#     prof_idx = int(np.where(profile_ids == prof_no)[0][0])
# else:
#     raise ValueError(f"Profile ID {prof_no} not found; available IDs: {profile_ids}")

# mask_ml_all = ds.variables['mask_ml'][:].astype(bool)

# # Determine which profiles have any mixed-layer points
# profiles_with_ml = profile_ids[np.any(mask_ml_all, axis=1)]
# print(f"Profiles with mixed layer detected: {profiles_with_ml.tolist()}")

# # Dates
# time_var  = ds.variables['dates']  # dates in create_netcdf.py are Gregorian
# times_all = nc.num2date(time_var[:], units=time_var.units, calendar='gregorian')

# # Print dimensions
# print("=== Dimensions ===")
# for dim_name, dim in ds.dimensions.items():
#     print(f"{dim_name}: {len(dim)}")

# # Print variables and their shapes
# print("\n=== Variables ===")
# for var_name, var in ds.variables.items():
#     print(f"{var_name}: shape = {var.shape}, dtype = {var.dtype}")

# # Load dataset
# pressure = ds.variables['pressure'][:]
# ct = ds.variables['ct'][prof_idx, :]
# ct_full = ds.variables['ct'][prof_idx, :].filled(np.nan)
# mask_ml = ds.variables['mask_ml'][prof_idx, :].astype(bool)
# mask_int = ds.variables['mask_int'][prof_idx, :].astype(bool)
# mask_cl = ds.variables['mask_cl'][prof_idx, :].astype(bool)    

# # Mask invalid temperature
# # ct = np.ma.masked_invalid(ct_full)

# # Basic assertions 
# assert np.any(mask_ml), "No mixed layer mask found in the data."
# # assert np.any(mask_int), "No interface mask found in the data."

# # # 1. get the two extreme depths for this profile
# # dmax = ds.variables['depth_max_T'][prof_idx]
# # dmin = ds.variables['depth_min_T'][prof_idx]

# # # 2. interpolate CT at those exact depths
# # #    (assumes pressure and ct_raw are 1D arrays, and pressure is monotonic)
# # ct_at_dmax = np.interp(dmax, pressure, ct)
# # ct_at_dmin = np.interp(dmin, pressure, ct)

# # Get profile date string
# timestamp = times_all[prof_idx]
# date_str  = timestamp.strftime('%Y-%m-%d')

# # # Plot temperature profile
# # plt.figure(figsize=(6, 8))
# # plt.plot(ct, pressure, linewidth=1, label='Temperature')

# # # Scatter masks: ml, int, cl
# # plt.scatter(ct[mask_ml], pressure[mask_ml], s=20, label='Mixed Layer (ml)')
# # plt.scatter(ct[mask_int], pressure[mask_int], s=20, label='Interface (int)')
# # plt.scatter(ct[mask_cl], pressure[mask_cl], s=20, label='Connection Layer (cl)')

# # # Aesthetics
# # plt.gca().invert_yaxis()
# # plt.xlabel('Conservative Temperature (°C)')
# # plt.ylabel('Pressure (m)')
# # plt.title(f'Temperature Profile {prof_no} of {itp} on {date_str} with masks')
# # plt.legend()
# # plt.grid(True)
# # plt.tight_layout()

# # plt.show()