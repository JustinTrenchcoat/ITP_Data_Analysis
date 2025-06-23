import pandas as pd
from helper import *
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import gsw
import datetime
import seaborn as sns
from matplotlib.colors import SymLogNorm

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

# print(ds.dimensions)
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

#######################################################
# Read Data and save, Only run once                   #
#######################################################
# tagData_dir = 'tagData'
# df_list = []
# for fileName in tqdm(sorted(os.listdir(tagData_dir)), desc="Processing files"):
#     match = re.search(r'itp(\d+)cormat\.nc', fileName)
#     if match:
#             itp_num = int(match.group(1))
#             full_path = os.path.join(tagData_dir, fileName)
#             df_list = readNC(full_path, df_list, itp_num)
#             final_df = pd.concat(df_list, ignore_index=True)
#             final_df.to_pickle("final.pkl")
#######################################################
#                                                     #
#######################################################


final_df = pd.read_pickle("final.pkl")
print(final_df.shape)
# print(final_df.head())


test_df = final_df[final_df['itpNum'] == 100].copy()
print(test_df.shape)
test_df = test_df[test_df['mask_cl'].notna()]
print(test_df.shape)

depth_col = test_df['depth']
n_sq_col = test_df['n_sq']
def printBasicStat(column):
    # Basic statistics
    print("Statistical Summary of Depth Differences:")
    print(f"Count         : {len(column)}")
    print(f"Min           : {np.min(column)}")
    print(f"Max           : {np.max(column)}")
    print(f"Mean          : {np.mean(column)}")
    print(f"Median        : {np.median(column)}")
    print(f"Std Dev       : {np.std(column)}")
    print(f"Variance      : {np.var(column)}")
    print(f"25th Percentile (Q1): {np.percentile(column, 25)}")
    print(f"75th Percentile (Q3): {np.percentile(column, 75)}")
    print(f"IQR           : {np.percentile(column, 75) - np.percentile(column, 25)}")
printBasicStat(depth_col)
printBasicStat(n_sq_col)

depth_index = np.where(np.isnan(depth_col))[0]
print(depth_index)

n_sq_index = np.where(np.isnan(n_sq_col))[0]
print(n_sq_index)


# # print(test_df['n_sq'].min(), test_df['n_sq'].max(), test_df['n_sq'].describe())
import matplotlib.colors as colors

# def transectPlot(df):
#     vmin = df['n_sq'].quantile(0.01)  # ignore extreme low outliers
#     vmax = df['n_sq'].quantile(0.99)  # ignore extreme high outliers
#     bounds = np.linspace(vmin, vmax, 100)

#     norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)

#     sc = plt.scatter(
#         df['date'], df['depth'],
#         c=df['n_sq'], cmap='viridis',
#         norm=norm,
#         s=5, alpha=0.8
#     )

#     plt.colorbar(sc, label='N^2')


#     plt.title('Transect View: N_sq over Time and Depth for ITP 100')
#     plt.xlabel('Date')
#     plt.ylabel('Depth (m)')
#     plt.gca().invert_yaxis()
#     plt.tight_layout()
#     plt.show()

# transectPlot(test_df)

# import matplotlib.dates as mdates
# def contourPlot(df):
#     df = df.copy()

#     # Ensure datetime and binning (in case not already done)
#     df['date_bin'] = pd.to_datetime(df['date']).dt.floor('D')

#     # Pivot to 2D grid: rows = depth, cols = date
#     pivot_df = df.pivot_table(
#         index='depth',
#         columns='date_bin',
#         values='n_sq',
#         aggfunc='mean'
#     )

#     # Create grid: X = dates, Y = depths, Z = n_sq
#     X, Y = np.meshgrid(pivot_df.columns, pivot_df.index)
#     Z = pivot_df.values
#     Z_masked = np.ma.masked_invalid(Z)



#     print("Z shape:", Z.shape)
#     print("Z_masked mask sum (invalid points):", np.sum(Z_masked.mask))
#     print("Any valid data points?:", np.any(~Z_masked.mask))
#     print("Z min/max (valid):", Z_masked.min(), Z_masked.max())

#     plt.figure(figsize=(14, 6))

#     norm = SymLogNorm(
#         linthresh=1e-5,   # Linear between -1e-5 and +1e-5
#         linscale=1.0,
#         vmin=-0.08,
#         vmax=0.08,
#         base=10
#     )

#     cp = plt.contourf(X, Y, Z_masked, levels=100, cmap='seismic', norm=norm)

#     # Plot
#     # cp = plt.contourf(X, Y, Z, levels=100, cmap='viridis', norm=LogNorm(vmin=1e-7, vmax=1e-4))
#     plt.gca().invert_yaxis()

#     # Format x-axis ticks
#     plt.xticks(rotation=45)
#     plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

#     # Add colorbar and labels
#     plt.colorbar(cp, label='N_sq')
#     plt.title('Contour Plot of N^2 over Time and Depth')
#     plt.xlabel('Date')
#     plt.ylabel('Depth (m)')
#     plt.tight_layout()
#     plt.show()
# # contourPlot(test_df)





# testing phase
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates

# def transectPlot(df):
#     # Bin depth to 1 m to reduce resolution
#     df = df.copy()
#     # df = df.dropna(subset=['depth'])
#     # df = df[np.isfinite(df['depth'])]
#     # df['depth_bin'] = (df['depth'] // 1).astype(int)

    
#     # Pivot to get 2D grid: depth x date, with mean N²
#     pivot_df = df.pivot_table(
#         index='depth',
#         columns='date',
#         values='n_sq',
#         aggfunc='mean'
#     )
    
#     # Prepare data for plotting
#     X, Y = np.meshgrid(
#         pivot_df.columns,
#         pivot_df.index
#     )
#     Z = pivot_df.values
    
#     # Plot using pcolormesh for better control
#     plt.figure(figsize=(14, 6))
    
#     # Use quantiles to clip color scale, for better contrast
#     vmin = np.nanquantile(Z, 0.01)
#     vmax = np.nanquantile(Z, 0.99)
#     print(f'vmin is {vmin}')
#     print(f'vmax is {vmax}')
    
#     bounds = np.linspace(vmin, vmax, 100)

#     norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)

#     pcm = plt.pcolormesh(
#         X, Y, Z,
#         shading='auto',
#         cmap='viridis',
#         norm=norm
#     )
    
#     plt.colorbar(pcm, label=r'$N^2$ (s$^{-2}$)')
    
#     plt.gca().invert_yaxis()
#     plt.xlabel('Date')
#     plt.ylabel('Depth (m)')
#     plt.title('Transect View of $N^2$ over Time and Depth in ITP 100')
    
#     plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=3))
#     plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
#     plt.xticks(rotation=45)
    
#     plt.tight_layout()
#     plt.show()

# # Usage
# transectPlot(test_df)








# print(test_df.head())
# print(test_df.shape)
# filtered_df = final_df[final_df['mask_cl'].notna()]
# print(filtered_df)




# ocean_sorted_df = final_df.sort_values(by='profileNumber')
# print(f'sorted DF: \n{ocean_sorted_df.tail()}')

# Now ocean_df should have everything

# we now sort the dataframe in order of time:
# ocean_sorted_df = ocean_df.sort_values(by='startDate')







