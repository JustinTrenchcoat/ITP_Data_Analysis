import pickle
from helper import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
import matplotlib.dates as mdates
import math


with open('grouped.pkl', 'rb') as f:
    groupedYears = pickle.load(f)


# making mean, mean+std, mean-std for each group,
def vertPlot(df_list, variable, path, type):
    color_list = ["r","y", "g", "b","k"]
    for i, df_group in enumerate(df_list):
        print(f'-------Processing Group {i}------------------')
        df_copy = df_group.copy()
        # df_copy['year'] = df_copy['date'].apply(lambda d: d.year)
        avg = df_copy.groupby('depth')[variable].agg(["mean", "std", "count"])
        count  = avg['count'].to_numpy()
        count_mask = count > 50
            
        depth = avg.index.to_numpy()
        depth = depth[count_mask]
        mean = avg['mean'].to_numpy()
        mean = mean[count_mask]
        std = avg['std'].to_numpy()
        std = std[count_mask]

        # print(depth[mean.argmin()])
        if type == "origin":
            plt.scatter(mean, depth, label = f"Group {i}", color = color_list[i], s=1,edgecolors='none')
            plt.axhline(depth[mean.argmin()], color=color_list[i], linestyle='--')
            plt.axhline(depth[mean.argmax()], color=color_list[i], linestyle='--')
        elif type == "plus":
            plt.scatter(mean+std, depth, alpha=0.1,label = f"Group {i}", color = color_list[i])
        else:
            plt.scatter(mean-std,depth, alpha=0.1,label = f"Group {i}", color = color_list[i])

    plt.gca().invert_yaxis()
    if type == "origin":
            plt.xlabel(f'Average {variable}')
            plt.title(f"Average of {variable}")
    elif type == "plus":
            plt.xlabel(f'Average {variable}+1 standard deviation')
            plt.title(f"Average of {variable}+std")
    else:
            plt.xlabel(f'Average {variable}-1 standard deviation')
            plt.title(f"Average of {variable}-std")

    plt.ylabel('Depth')
    # plt.legend()
    plt.tight_layout()
    # plt.savefig(f"plots/fine/vertPlot/{path}{type}")
    plt.show()
    plt.close()

vertPlot(groupedYears, "temp", "temp", "origin")
# vertPlot(groupedYears, "salinity", "sal", "minus")
# vertPlot(groupedYears, "dT/dZ" , "dTdZ", "minus")
# vertPlot(groupedYears, "dS/dZ", "dSdZ", "minus")
# vertPlot(groupedYears, "n_sq", "nSq", "minus")
# vertPlot(groupedYears, "R_rho", "rho", "minus")

# def vertPlot(df_list, variable, path, type):
#     color_list = ["r","y", "g", "b","k"]

#     df_copy = df_list.copy()
#     # df_copy['year'] = df_copy['date'].apply(lambda d: d.year)
#     avg = df_copy.groupby('depth')[variable].agg(["mean", "std", "count"])
            
#     depth = avg.index.to_numpy()
#     mean = avg['mean'].to_numpy()
#     std = avg['std'].to_numpy()
#     if type == "origin":
#         plt.scatter(mean, depth, label = f"Group", color = color_list[0])
#     elif type == "plus":
#         plt.scatter(mean+std, depth, alpha=0.1,label = f"Group", color = color_list[0])
#     else:
#         plt.scatter(mean-std,depth, alpha=0.1,label = f"Group", color = color_list[0])

#     plt.gca().invert_yaxis()
#     if type == "origin":
#             plt.xlabel(f'Average {variable}')
#             plt.title(f"Average of {variable}")
#     elif type == "plus":
#             plt.xlabel(f'Average {variable}+1 standard deviation')
#             plt.title(f"Average of {variable}+std")
#     else:
#             plt.xlabel(f'Average {variable}-1 standard deviation')
#             plt.title(f"Average of {variable}-std")

#     plt.ylabel('Depth')
#     plt.legend()
#     plt.tight_layout()
#     # plt.savefig(f"plots/fine/vertPlot/{path}{type}")
#     plt.show()
#     plt.close()

# testyear = groupedYears[3].copy()
# testyear['year'] = testyear['date'].apply(lambda d: d.year)
# testyear = testyear[testyear["year"]==2017]
# vertPlot(testyear, "dT/dZ" , "dTdZ", "minus")