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
depth_filtered = [2.315,6.478,7.152,7.449,8.576,9.2]
start = np.floor(min(depth_filtered))
end = np.ceil(max(depth_filtered))
regular_depths = np.arange(start, end, 0.25)
print(regular_depths)
temp_filtered = [45.6,98.4,12,72,51.339,19.4468]
temp_interp = CubicSpline(depth_filtered, temp_filtered)
interpolated_temperatures = temp_interp(regular_depths)
print(f'length of interp temp: {len(interpolated_temperatures)}')
print(interpolated_temperatures)