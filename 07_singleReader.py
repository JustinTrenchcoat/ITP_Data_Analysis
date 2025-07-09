import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
import traceback
from scipy.interpolate import CubicSpline
from helper import *
import re
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d


# this script looks through each profile extract depth, temperature and every other measurement, interpolate the values by depth grid of 0.25m
# should it look like a dataframe or?

# set up file path
# full_path = r'D:\EOAS\ITP_Data_Analysis\datasets\itp62cormat\cor0008.mat'
full_path = r'D:\EOAS\ITP_Data_Analysis\gridDataMat\itp101cormat\cor0320.mat'

new_dir = "testData"
folder_name = "test"



def clean(x):
    # Convert to string
    x_str = str(x)
    # Remove brackets, whitespace
    x_str = x_str.strip("[] ").replace(" ", "")
    # Replace any non-alphanumeric or underscore/dash/dot characters with underscore
    x_str = re.sub(r'[^\w\-.]', '_', x_str)
    return x_str

def plot(x, y):
    plt.plot(x, y, marker='o',linestyle='dashed',linewidth=2, markersize=12)
    plt.xlabel("test x")
    plt.ylabel("test y")
    plt.title("test Plot")
    plt.grid(True)
    plt.gca().invert_yaxis()
    # Optional: Rotate date labels for clarity
    plt.xticks(rotation=45)
    plt.show()

try:
    data = loadmat(full_path)
    
    # Adjust according to actual variable names in your .mat file
    depth = data['Depth'].squeeze()
    lat = data['lat'].squeeze()
    lon = data['lon'].squeeze()
    psdate = data['startDate'].squeeze()
    date = pd.to_datetime(psdate, format='%m/%d/%y')
    date = date.date
    temp = data['Temperature'].squeeze()
    salinity = data['Salinity'].squeeze()
    pres = pressure(depth, lat)

    assert not np.isnan(depth).any(), "depth contains NaN values"
    assert not np.isnan(temp).any(), "temp contains NaN values"
    assert not np.isnan(salinity).any(), "salinity contains NaN values"
    assert not np.isnan(pres).any(), "pres contains NaN values"

    # extra lines for debugging#####################
    print("SA min/max:", np.min(salinity), np.max(salinity))
    print("temp min/max:", np.min(temp), np.max(temp))
    print("pres min/max:", np.min(pres), np.max(pres))
    for i, (sa_i, t_i, p_i) in enumerate(zip(salinity, temp, pres)):
        try:
            gsw.CT_from_t(np.array([sa_i]), np.array([t_i]), np.array([p_i]))
        except Exception as e:
            print(f"Invalid input at index {i}: SA={sa_i}, t={t_i}, p={p_i}")



    ###############################################################
    new_df = pd.DataFrame({
            "depth" : depth,
            'temp' : temp,
            'date' : date,
            "salinity" : salinity,
            "lon" : lon,
            "lat" : lat,
            "pressure" : pres
        })

    # add new cols:
    new_df['dT/dZ'] = gaussian_filter1d(np.gradient(temp, depth),sigma=80, mode='nearest')
    new_df['dS/dZ'] = gaussian_filter1d(np.gradient(salinity, depth),sigma=80, mode='nearest')

    n_sq = gaussian_filter1d(gsw.Nsquared(salinity, temp, pres, lat)[0], sigma=80,mode="nearest")
    # padding for last value as the function returns only N-1 values
    n_sq_padded = np.append(n_sq, np.nan)
    new_df['n_sq'] = n_sq_padded
    # turner angle and R_rho
    [turner_angle, R_rho,p_mid] = gsw.Turner_Rsubrho(salinity, temp, pres)
    turner_angle = gaussian_filter1d(turner_angle, sigma=80, mode="nearest")
    R_rho = gaussian_filter1d(R_rho, sigma=80, mode="nearest")
    new_df['turner_angle'] = np.append(turner_angle,np.nan)
    new_df['R_rho'] = np.append(R_rho,np.nan)
    plot(new_df["dT/dZ"], new_df['depth'])


except Exception as e:
    print(f"Error processing file: {full_path}")
    traceback.print_exc()
