# this file filters only ITP data that is collected in the Beaufort Gyre
# lon>-160 & lon<-130 & lat>73 & lat<81;

from mpl_toolkits.basemap import Basemap
from itp.itp_query import ItpQuery
import matplotlib.pyplot as plt
from itp.profile import Profile
from tqdm import tqdm
from netCDF4 import Dataset, date2num
import datetime
import numpy as np


global good_list
good_list = []
path = r'D:\EOAS\ITP_package_try\itp_data\itp_final_2025_05_07.db'
# query = ItpQuery(path, system=[1])
# results = query.fetch()

query = ItpQuery(path, latitude=[73, 81], longitude=[-160, -130])
query.set_max_results(100000)

results = query.fetch()
for p in tqdm(results):
    if max(Profile.depth(p))>=400:
        good_list.append(p)
list_size = len(good_list)

import pandas as pd
import xarray as xr
import numpy as np

# Example: build a DataFrame where each row is a profile measurement (profile_number, level, pressure, temp, salinity)
rows = []
for p in tqdm(good_list):
    n = len(p.pressure)
    for i in range(n):
        rows.append({
            "profile_number": p.profile_number,
            "date_time": p.date_time,
            "latitude": p.latitude,
            "longitude": p.longitude,
            "level": i,
            "pressure": p.pressure[i],
            "temperature": p.temperature[i],
            "salinity": p.salinity[i]
        })

df = pd.DataFrame(rows)

# Convert to xarray Dataset
ds = df.set_index(["profile_number", "level"]).to_xarray()

# Save to NetCDF
ds.to_netcdf("profiles.nc")
