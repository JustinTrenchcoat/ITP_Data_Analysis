full_path = r'gridData\itp1cormat\cor0001_-150.13129388_78.82668839_08_16_05_06_00_01_08_16_05_06_46_41.csv'
import pandas as pd
from scipy.io import savemat
import re
import os

# Step 1: Read the CSV
df = pd.read_csv(full_path)  # replace with your CSV path

# Step 2: Convert to dictionary (MATLAB needs a dict of variable names)
mat_dict = {col: df[col].values for col in df.columns}

# Step 3: Save to .mat file
savemat('your_file.mat', mat_dict)  # replace with desired output path

