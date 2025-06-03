import os
import h5py
import numpy as np
from tqdm import tqdm
from datetime import datetime

from helper import *

# Global list to hold (filename, datetime) tuples
dates_list = []


def extract_date_and_store(folder, file, path):
    try:
        with h5py.File(path, 'r') as f:
            def read_var(varname):
                data = np.array(f[varname])
                data_flat = data.flatten()

                if data_flat.dtype == 'uint16':
                # UTF-16 encoded string
                    return data_flat.tobytes().decode('utf-16-le').rstrip('\x00').strip()
                elif data_flat.dtype.kind in {'U', 'S'}:
                    # Join character arrays
                    return ''.join(map(str, data_flat)).strip()
                elif data_flat.size == 1:
                    # Return scalar from 1-element numeric array
                    return data_flat.item()
                else:
                    # Return full array (for debugging or complex structures)
                    return data_flat

            # Adjust 'date_var' to your actual variable name
            pedate = read_var("pedate")
            if pedate == "nan":
                print(f"{folder}/{file}: 'pedate' is nan")
                return
            
            
            # Parse to datetime (adjust format if needed)
            date_obj = datetime.strptime(pedate, "%m/%d/%y")
            
            # Store (full filename, datetime)
            dates_list.append((f"{folder}/{file}", date_obj))
    except Exception as e:
        print(f"Error reading {folder}/{file}: {e}")

# Run the traversal
datasets_dir = "datasets"
traverse_datasets(datasets_dir, extract_date_and_store)

# After traversal, find most recent
if dates_list:
    most_recent_file, most_recent_date = max(dates_list, key=lambda x: x[1])
    print(f"Most recent date is {most_recent_date.strftime('%Y-%m-%d')} from file {most_recent_file}")
else:
    print("No dates found.")