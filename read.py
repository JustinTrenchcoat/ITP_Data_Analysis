import pickle

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

# Save to a pkl file.
with open("df.pkl", "wb") as f:
    pickle.dump(df, f)