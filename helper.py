import matplotlib.pyplot as plt
from itp.profile import Profile

# this is the helper funcion collection for visualization.ipynb

def get_data(df, row_num):
    value  = df.loc[row_num]
    return value


def smartPlot(plotType, sample):
    if plotType == "tvd":
        depth = sample["depth"]
        temp = sample["temp"]
        time = sample["date"]
        sysNum = sample["sys_Num"]
        profNum = sample["prof_Num"]


        plotHelper(temp, depth, "Temperature", "Depth", sysNum,profNum,time)

        # plt.savefig(f"plots/temp_vs_depth_sys_{sysNum}_prof_{profNum}.png")
        plt.show()
    elif plotType == "svd":
        depth = sample["depth"]
        salinity = sample["salinity"]
        time = sample["date"]
        sysNum = sample["sys_Num"]
        profNum = sample["prof_Num"]

        plotHelper(salinity, depth, "Salinity", "Depth", sysNum,profNum,time)
        # plt.savefig(f"plots/sal_vs_depth_sys_{sysNum}_prof_{profNum}.png")
        plt.show()
 


def plotHelper(x,y, xlabel, ylabel, sysNum, profNum,time):
    # Filter y between 200 and 500
    mask = (y >= 200) & (y <= 500)
    x_filtered = x[mask]
    y_filtered = y[mask]
    plt.plot(x_filtered, y_filtered, marker='o',linestyle='dashed',linewidth=2, markersize=12)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{xlabel} vs {ylabel}, System# {sysNum} Profile# {profNum}, Time {time}")
    plt.grid(True)
    plt.gca().invert_yaxis()
    # Optional: Rotate date labels for clarity
    plt.xticks(rotation=45)
