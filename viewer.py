import xarray as xr
import pandas as pd

# Load the file
ds = xr.open_dataset("profile.nc")

# Convert relevant variables to DataFrame columns
df = pd.DataFrame({
    "pressure": ds.pressure.values,
    "temperature": ds.temperature.values,
    "salinity": ds.salinity.values
})

# Add scalar metadata as columns repeated for all rows (optional)
df["date_time"] = ds.attrs.get("date_time")
df["latitude"] = ds.attrs.get("latitude")
df["longitude"] = ds.attrs.get("longitude")
df["system_number"] = ds.attrs.get("system_number")
df["profile_number"] = ds.attrs.get("profile_number")
df["source"] = ds.attrs.get("source")

print(df)

