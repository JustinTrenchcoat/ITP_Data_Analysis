from netCDF4 import Dataset
import numpy as np
import pickle
from tqdm import tqdm


# Load later
with open("profiles.pkl", "rb") as f:
    good_list = pickle.load(f)

with Dataset("profiles.nc", "w", format="NETCDF4") as ncfile:
    # Create unlimited dimension for profiles (rows)
    ncfile.createDimension("obs", None)  # None means unlimited


    pressure_var = ncfile.createVariable("pressure", "f4", ("obs",))
    temperature_var = ncfile.createVariable("temperature", "f4", ("obs",))
    salinity_var = ncfile.createVariable("salinity", "f4", ("obs",))
    
    # Metadata repeated per observation
    date_time_var = ncfile.createVariable("date_time", str, ("obs",))  # store as string or bytes
    latitude_var = ncfile.createVariable("latitude", "f4", ("obs",))
    longitude_var = ncfile.createVariable("longitude", "f4", ("obs",))
    system_number_var = ncfile.createVariable("system_number", "i4", ("obs",))
    profile_number_var = ncfile.createVariable("profile_number", "i4", ("obs",))
    source_var = ncfile.createVariable("source", str, ("obs",))

    
    idx = 0
    for p in good_list:
        n = len(p.pressure)  # number of observations in this profile
        
        pressure_var[idx:idx+n] = p.pressure
        temperature_var[idx:idx+n] = p.temperature
        salinity_var[idx:idx+n] = p.salinity
        
        # Repeat metadata for each observation in profile
        date_time_var[idx:idx+n] = [p.date_time] * n
        latitude_var[idx:idx+n] = [p.latitude] * n
        longitude_var[idx:idx+n] = [p.longitude] * n
        system_number_var[idx:idx+n] = [p.system_number] * n
        profile_number_var[idx:idx+n] = [p.profile_number] * n
        source_var[idx:idx+n] = [p.source] * n
        
        idx += n
