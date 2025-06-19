import pandas as pd
from helper import *

'''
In the dataset:
every row is one observation from one profile.
It has to be ordered in time so that time series analysis would work.




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






data_dir = 'stairs'
ocean_df = pd.DataFrame()
# read data and put everything into ocean_df
def read_data(full_path, file_name, folder_name):
    profile_df = pd.read_csv(full_path)
    ocean_df= pd.concat([ocean_df, profile_df], ignore_index=True)

traverse_datasets(data_dir, read_data)
# Now ocean_df should have everything

# we now sort the dataframe in order of time:
ocean_sorted_df = ocean_df.sort_values(by='startDate')







