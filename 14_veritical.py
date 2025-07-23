import pickle
from helper import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import math
from matplotlib.colors import LinearSegmentedColormap



with open('grouped.pkl', 'rb') as f:
    groupedYears = pickle.load(f)


colorscale = LinearSegmentedColormap.from_list("year_trend", [
    "#4575b4",  # blue
    "#91bfdb",
    "#e0f3f8",
    "#fee090",
    "#fc8d59",
    "#d73027"   # red-orange
])


years = ["2004-2007", '2008-2011', '2012-2015', '2016-2019', '2020-2023']
colors = colorscale(np.linspace(0, 1, len(years)))

temp_min_idx = []
temp_max_idx = []

# making mean, mean+std, mean-std for each group,
def vertPlot(df_list, variable, path, type):
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

        if variable == "temp":
            temp_min_idx.append(mean.argmin())
            temp_max_idx.append(mean.argmax())

        # print(depth[mean.argmin()])
        if type == "origin":
            plt.scatter(mean, depth, label = f"Group {i}", color = colors[i], s=1,edgecolors='none')
            plt.axhline(depth[temp_max_idx[i]], color=colors[i], linestyle='--')
            plt.axhline(depth[temp_min_idx[i]], color=colors[i], linestyle='--')
        elif type == "plus":
            plt.scatter(mean+std, depth, alpha=0.1,label = f"Group {i}", color = colors[i])
        else:
            plt.scatter(mean-std,depth, alpha=0.1,label = f"Group {i}", color = colors[i])

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
    # # Legend: color squares instead of dots
    # legend_patches = [
    #     Patch(facecolor=colors[i], edgecolor='black', label=str(years[i]))
    #     for i in range(len(years))
    # ]   

    # plt.legend(handles=legend_patches, title='Year')    
    plt.tight_layout()
    plt.savefig(f"plots/fine/vertPlot/{path}{type}")
    plt.show()
    plt.close()

def plot_legend_only(years, colors, filename, legend_title="Year"):
    fig, ax = plt.subplots()

    # Create legend handles
    legend_patches = [
        Patch(facecolor=colors[i], edgecolor='black', label=str(years[i]))
        for i in range(len(years))
    ]

    # Place the legend in center of new figure
    legend = ax.legend(
        handles=legend_patches,
        title=legend_title,
        loc='center',
        frameon=False,
        fontsize=12,
        title_fontsize=13
    )

    # Hide axes for clean legend-only figure
    ax.axis('off')
    
    # Resize figure to fit legend
    fig.set_size_inches(2.5, len(years) * 0.35 + 1)
    plt.savefig(f"plots/fine/vertPlot/{filename}")
    plt.show()
    plt.close()


vertPlot(groupedYears, "temp", "temp", "origin")
plot_legend_only(years, colors, "legend")
vertPlot(groupedYears, "turner_angle", "turner", "origin")
vertPlot(groupedYears, "salinity", "sal", "origin")
vertPlot(groupedYears, "dT/dZ" , "dTdZ", "origin")
vertPlot(groupedYears, "dS/dZ", "dSdZ", "origin")
vertPlot(groupedYears, "n_sq", "nSq", "origin")
vertPlot(groupedYears, "R_rho", "rho", "origin")


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