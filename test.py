from mpl_toolkits.basemap import Basemap
from itp.itp_query import ItpQuery
from itp.profile import Profile
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
from netCDF4 import Dataset
import numpy as np

path = r'D:\EOAS\ITP_Data_Analysis\itp_data\itp_final_2025_05_07.db'
# query = ItpQuery(path, system=[1])
# results = query.fetch()
# Load saved file
# with open("profiles.pkl", "rb") as f:
#     profiles = pickle.load(f)
# size = len(profiles)
# for idx, p in tqdm(enumerate(profiles)):
#     print(f"{idx}th profile's sample size  is: {type(Profile.depth(p))}")
#     if idx == 10:
#         break
import xarray as xr
import pandas as pd
from helper import *


# Load the file
ds = xr.open_dataset("profiles.nc")

# Convert relevant variables to DataFrame columns
df = pd.DataFrame({
    "prof_Num": ds.profile_number.values,
    "sys_Num": ds.system_number.values,
    "source": ds.source.values,
    "date": ds.date_time.values,
    "lat": ds.latitude.values,
    "lon": ds.longitude.values,
    "temp": ds.temperature.values,
    "salinity": ds.salinity.values,
    "depth": ds.depth.values,
    "pressure": ds.pressure.values

})

print(get_data(df, 3, ["temp", "depth", "salinity"]))