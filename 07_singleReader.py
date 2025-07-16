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
from scipy.interpolate import interp1d



# this script looks through each profile extract depth, temperature and every other measurement, interpolate the values by depth grid of 0.25m
# should it look like a dataframe or?

# set up file path
# full_path = r'D:\EOAS\ITP_Data_Analysis\datasets\itp62cormat\cor0008.mat'
full_path = r'D:\EOAS\ITP_Data_Analysis\gridDataMat\itp103cormat\cor1207.mat'

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
    print(date)
    date = date.date
    temp = data['Temperature'].squeeze()
    salinity = data['Salinity'].squeeze()
    pres = pressure(depth, lat)

    assert not np.isnan(depth).any(), "depth contains NaN values"
    assert not np.isnan(temp).any(), "temp contains NaN values"
    assert not np.isnan(salinity).any(), "salinity contains NaN values"
    assert not np.isnan(pres).any(), "pres contains NaN values"

    # extra lines for debugging#####################
    print(lat)
    print("SA min/max:", np.min(salinity), np.max(salinity))
    print("temp min/max:", np.min(temp), np.max(temp))
    print("pres min/max:", np.min(pres), np.max(pres))
    print("depth min/max", np.min(depth), np.max(depth))
    [n_sq,p_mid] = gsw.Nsquared(salinity, temp, pres, lat)
    print("n_sq min/max", np.min(n_sq), np.max(n_sq))
    [turner_angle, R_rho, p_mid_2] = gsw.Turner_Rsubrho(salinity, temp, pres)
    R_rho = 1/R_rho
    print(np.array_equal(p_mid, p_mid_2))
    print("rho min/max", np.min(R_rho), np.max(R_rho))
    print("after conversion:")
    print("############################")
    salinity = gsw.SA_from_SP(salinity, pres, lon, lat)
    temp = gsw.CT_from_t(salinity, temp, pres)
    print("SA min/max:", np.min(salinity), np.max(salinity))
    print(len(salinity))
    print("temp min/max:", np.min(temp), np.max(temp))
    [n_sq,p_mid] = gsw.Nsquared(salinity, temp, pres, lat)
    print("n_sq min/max", np.min(n_sq), np.max(n_sq))
    [turner_angle, R_rho, p_mid_2] = gsw.Turner_Rsubrho(salinity, temp, pres)
    print("Tu min/max", np.min(turner_angle), np.max(turner_angle))

    print("p_mid min/max:", np.min(p_mid_2), np.max(p_mid_2))
    indexPmin = np.argmin(p_mid_2)
    indexPMax = np.argmax(p_mid_2)
    R_rho_min = R_rho[indexPmin]
    R_rho_max = R_rho[indexPMax]
    print(f"Rho at PMin is {R_rho_min}, Rho at PMax is {R_rho_max}")
    depth_mid = height(p_mid, lat)

    # R_rho = np.reciprocal(R_rho)
    print(len(R_rho))
    plot(R_rho, depth_mid)
    print("rho min/max", np.min(R_rho), np.max(R_rho))

    R_rho_interp = interp1d(depth_mid, R_rho,kind='linear', fill_value="extrapolate")
    interpolated_R_rho = R_rho_interp(depth)
    print("rho interp min/max", np.min(interpolated_R_rho), np.max(interpolated_R_rho))
#########################
    R_rho_smooth = gaussian_filter1d(interpolated_R_rho, sigma=40, mode='nearest')
    # R_rho_smooth = np.reciprocal(R_rho_smooth)
    print("rho smooth min/max", np.min(R_rho_smooth), np.max(R_rho_smooth))

    plot(R_rho_smooth, depth)








    # plot(temp, depth)

   

except Exception as e:
    print(f"Error processing file: {full_path}")
    traceback.print_exc()
