import pandas as pd
from helper import *
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import gsw
import datetime
import re
import seaborn as sns
from matplotlib.colors import SymLogNorm
import matplotlib.colors as colors
import matplotlib.dates as mdates
from matplotlib.colors import Normalize
from scipy.ndimage import uniform_filter1d
import math
import numpy as np
import scipy
from scipy.interpolate import CubicSpline
full_path  = r'D:\EOAS\ITP_Data_Analysis\goldData\itp1cormat\cor1686.mat'
try:
    with h5py.File(full_path, 'r') as f:
    # read variables from single file for later reference.
        pr_filt = read_var(f, 'pr_filt')
        te_adj = read_var(f, 'te_adj')
        sa_adj = read_var(f, "sa_adj")
        lat = read_var(f, "latitude")
        lon = read_var(f, "longitude")
        depth = height(pr_filt, lat)
        helPlot(te_adj,depth)

except Exception as e:
    print(f"Error processing file: {full_path}")
    traceback.print_exc()
