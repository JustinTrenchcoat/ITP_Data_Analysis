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

experiment_df = final_df[final_df['itpNum'].isin([62, 64,65,68,69])].copy()


import matplotlib.pyplot as plt
import numpy as np

def plotHelper(x, y, xlabel, ylabel, third):
    plt.figure(figsize=(8, 6))
    
    categories = third.unique()
    cmap = plt.get_cmap('tab10')  # You can change colormap if needed
    
    for idx, cat in enumerate(categories):
        mask = (third == cat)
        plt.scatter(x[mask], y[mask], s=10, label=str(cat), color=cmap(idx))
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{xlabel} vs {ylabel} Colored by mask_sc")
    plt.legend(title="mask_sc")
    plt.show()

# Example usage:
plotHelper(experiment_df['n_sq'], experiment_df['R_rho'], "N_sq", "R_rho", experiment_df['mask_sc'])
