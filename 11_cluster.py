'''
This section would mainly focus on making cluster analysis for staircase/non-staircase, or specific features.
It aims to extract key factors that determines the occurance of staircase/other features
'''
import pandas as pd
from helper import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
import statsmodels.api as sm


final_df = pd.read_pickle("final.pkl")

experiment_df = final_df[final_df['itpNum'].isin([62, 65, 68])].copy()
import matplotlib.pyplot as plt
import numpy as np

def plotHelper(x, y, xlabel, ylabel, third):
    plt.figure(figsize=(8, 6))
    
    # Scatter plot with coloring
    plt.scatter(x, y, c=third, s=10)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{xlabel} vs {ylabel} Colored by mask_sc")

    plt.legend(title="mask_sc")

    plt.show()

# Example usage:
plotHelper(experiment_df['n_sq'], experiment_df['R_rho'], "N_sq", "R_rho", experiment_df['mask_sc'])
