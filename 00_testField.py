import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import xarray as xr
import pandas as pd
from helper import *

# Configuration
# file_path = 'itp100cormat.nc' 
# # 
# ds = nc.Dataset(file_path)
# # print(ds.dimensions)
# print(ds.variables)
ocean_df = pd.DataFrame()
def readNC(full_path, ls, itp_num):
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
        for i in range(len(profN)):
            mask_cl = connect_layer_mask[i]
            mask_int = interface_layer_mask[i]
            mask_ml = mixed_layer_mask[i]
            mask_sc = staircase_mask[i]
            for j in range(len(depth[i])):
                if np.isnan(depth[i][j]):
                    print(f"Warning! index {i} has NaN value")
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
            new_df['pressure'] = pressure(depth[i],lat[i])
            new_df['itpNum'] = itp_num

            # background infos, do it here because we calculate it per-profile:
            # N Sqaured:
            n_sq = gsw.Nsquared(salinity[i], temp[i], new_df['pressure'], lat[i])[0]
            # padding for last value as the function returns only N-1 values
            n_sq_padded = np.append(n_sq, np.nan)
            new_df['n_sq'] = n_sq_padded
            # turner angle and R_rho
            [turner_angle, R_rho,p_mid] = gsw.Turner_Rsubrho(salinity[i], temp[i], new_df['pressure'])
            new_df['turner_angle'] = np.append(turner_angle,np.nan)
            new_df['R_rho'] = np.append(R_rho,np.nan)
            ####################
            ls.append(new_df)
    return ls

    
tagData_dir = 'tagData'
test_list = []
test_list = readNC(r"D:\EOAS\ITP_Data_Analysis\tagData\itp100cormat.nc", test_list, itp_num=100)
final_df = pd.concat(test_list, ignore_index=True)
print(final_df.shape)


test_df = final_df[final_df['itpNum'] == 100].copy()
print(test_df.shape)
test_df = test_df[test_df['depth'].notna()]
print(test_df.shape)

# df_list = []
# for fileName in tqdm(sorted(os.listdir(tagData_dir)), desc="Processing files"):
#     match = re.search(r'itp(\d+)cormat\.nc', fileName)
#     if match:
#             itp_num = int(match.group(1))
#             full_path = os.path.join(tagData_dir, fileName)
#             df_list = readNC(full_path, df_list, itp_num)
#             final_df = pd.concat(df_list, ignore_index=True)
#             final_df.to_pickle("final.pkl")