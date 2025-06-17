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
Staircase


Dates:
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
    "Sharp",
    "Mushy",
    "SuperMushy",
    "Staircase",
]
drop_features = ["Date"]
target = ["RainTomorrow"]