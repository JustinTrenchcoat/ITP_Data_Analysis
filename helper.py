import xarray as xr
import pandas as pd
import numpy

# this is the helper funcion collection for visualization.ipynb

def get_data(df, row_num, var):
    results = []
    for idx,v in enumerate(var):
        value  = df.loc[row_num, v][0]
        results.append(value)
    return results