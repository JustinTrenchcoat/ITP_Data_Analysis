'''
This file will try to utilize unsupervised learning to classify water cilumn data.
The initial idea was to use PCA to determine number of clusters, then use SOM or other methods to cluster.
'''

# from minisom import MiniSom
import numpy as np
import matplotlib.pyplot as plt
import h5py
import traceback
from helper import height, read_var

# Path to .mat file
full_path = r'D:\EOAS\ITP_Data_Analysis\goldData\itp1cormat\cor0002.mat'
# or take a piece of pkl file for testing
