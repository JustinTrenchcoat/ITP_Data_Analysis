import matplotlib.pyplot as plt
import numpy as np
import h5py
import traceback
from scipy.interpolate import interp1d
from helper import height

# this script looks through each profile extract depth, temperature and every other measurement, interpolate the values by depth grid of 0.25m
# should it look like a dataframe or?

# set up file path
# full_path = r'D:\EOAS\ITP_Data_Analysis\datasets\itp62cormat\cor0008.mat'
full_path = r'D:\EOAS\ITP_Data_Analysis\datasets\itp1cormat\cor0002.mat'

def read_var(f, varname):
        data = np.array(f[varname])
        if data.dtype == "uint16":
            return data.tobytes().decode('utf-16-le')
        return data
        

try:
    with h5py.File(full_path, 'r') as f:
        # Print all root level object names (aka keys) 
        # these can be group or dataset names 
        print("Keys: %s" % f.keys())

       
        print(f"End Time: {read_var(f, "pedate")}, {read_var(f, "pstop")}")
        print(f"Start Time: {read_var(f, "psdate")}, {read_var(f, "pstart")}")


except Exception as e:
    print(f"Error processing file: {full_path}")
    traceback.print_exc()