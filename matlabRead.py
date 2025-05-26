import h5py
import numpy as np
import os
import gsw
from tqdm import tqdm
from helper import *

# Folder containing your .mat files
folder_path = 'itp5cormat'

# Store results
good_profile = []
bad_profile = []
missing_profile = []

def decode_ascii(matlab_str):
    return ''.join([chr(c) for c in matlab_str])

# Delete bad profile files
def delete_file(full_path):
    try:
        if os.path.isfile(full_path):
            os.remove(full_path)
            print(f"Deleted: {filename}")
        else:
            print(f"File not found: {filename}")
    except Exception as e:
        print(f"Failed to delete {filename}: {e}")

# Loop over all expected file names
for i in tqdm(range(1, 1095)):
    filename = f"cor{i:04d}.mat"  # Formats as cor0001.mat, ..., cor0014.mat
    full_path = os.path.join(folder_path, filename)
    
    if not os.path.isfile(full_path):
        # print(f"Missing: {filename}")
        missing_profile.append(i)
        continue

    try:
        with h5py.File(full_path, 'r') as f:
            def read_var(varname):
                return np.array(f[varname]).squeeze()

            sa_cor = read_var('sa_cor')
            pr_filt = read_var('pr_filt')

            # Decode single string (e.g., one profile)
            date = decode_ascii(read_var("psdate"))
            time = decode_ascii(read_var("pstart"))
            lat = read_var("latitude")
            lon = read_var("longitude")

            # filter out NaNs or bad profiles
            valid_mask = ~np.isnan(sa_cor) & ~np.isnan(pr_filt)
            sa_cor = sa_cor[valid_mask]
            pr_filt = pr_filt[valid_mask]
            depth = height(pr_filt,lat)
            dep_max = max(height(pr_filt,lat))
            if (dep_max >= 400) and (73 <= lat <= 81) and (-160 <= lon <= -130):
                good_profile.append(i)
            else:
                bad_profile.append(i)

    except Exception as e:
        bad_profile.append(i)


# Folder containing your .mat files
folder_path = 'itp5cormat'

# Delete bad profile files
for i in bad_profile:
    filename = f"cor{i:04d}.mat"
    full_path = os.path.join(folder_path, filename)
    
    try:
        if os.path.isfile(full_path):
            os.remove(full_path)
            print(f"Deleted: {filename}")
        else:
            print(f"File not found: {filename}")
    except Exception as e:
        print(f"Failed to delete {filename}: {e}")