from netCDF4 import Dataset
import numpy as np
import pickle
from tqdm import tqdm
from itp.profile import Profile

# Load saved file
with open("profiles.pkl", "rb") as f:
    profiles = pickle.load(f)
size = len(profiles)
with Dataset("profiles.nc", "w", format="NETCDF4") as ncfile:
    # we have 10 properties to store
    # date_time, latitude, longitude, system number, profile number, source, pressure, temperature, salinity depth 
    # everything beside pressure, temperature, depth and salinity should have dimension of length of the profiles

    ncfile.createDimension("profile", size)
    vlen_float64 = ncfile.createVLType(np.float64, "vlen_float64")

    # list of properties in the profiles:
    # pressure: 1*N size Numpy array, float64
    # temperature: 1*N size Numpy array, float64
    # salinity: 1*N size Numpy array, float64
    pressure_var = ncfile.createVariable("pressure", vlen_float64, ("profile",))
    temperature_var = ncfile.createVariable("temperature", vlen_float64, ("profile",))
    salinity_var = ncfile.createVariable("salinity", vlen_float64, ("profile",))
    depth_var = ncfile.createVariable("depth",vlen_float64, ("profile",))
    
    # date_time: ISO 8601 string
    # latitude: latitude where the profile began, float
    # longitude: longitude where the profile began, float
    # system number: ISP system number of this profile, int
    # profile number: profile number, int
    # source: original filename of this profile, string
    date_time_var = ncfile.createVariable("date_time", str, ("profile",))  # store as string or bytes
    latitude_var = ncfile.createVariable("latitude", "f8", ("profile",))
    longitude_var = ncfile.createVariable("longitude", "f8", ("profile",))
    system_number_var = ncfile.createVariable("system_number", "i8", ("profile",))
    profile_number_var = ncfile.createVariable("profile_number", "i8", ("profile",))
    source_var = ncfile.createVariable("source", str, ("profile",))

    # cluster number: anexperiment variable, for the clustering method to assign the cluster property
    # default value is 0, indicating that thedata point has not been assigned to any cluster
    # cluster_var = ncfile.createVariable("cluster","i8",("pressure","temperature","salinity",
    #                                                     "depth","date", "lat", "lon", 
    #                                                     "sysNum", "profNum",
    #                                                     "source",))
    # write in data to ncfile
    for idx, p in tqdm(enumerate(profiles)):
        date_time_var[idx] = p.date_time
        latitude_var[idx] = p.latitude
        longitude_var[idx] = p.longitude
        system_number_var[idx] = p.system_number
        profile_number_var[idx] = p.profile_number
        source_var[idx] = p.source
        temperature_var[idx] = Profile.conservative_temperature(p)
        pressure_var[idx] = p.pressure
        salinity_var[idx] = Profile.absolute_salinity(p)
        depth_var[idx] = Profile.depth(p)
