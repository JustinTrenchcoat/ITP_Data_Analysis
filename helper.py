import matplotlib.pyplot as plt
from itp.profile import Profile
import gsw

# this is the helper for maniulating the .mat files:
def absolute_salinity(salinity, pressure, longitude, latitude):
    return gsw.conversions.SA_from_SP(
        salinity,
        pressure,
        longitude,
        latitude
    )

# Decode ASCII arrays to strings
def decode_ascii(matlab_str):
    return ''.join([chr(c) for c in matlab_str])



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



# # Load saved file
# with open("df.pkl", "rb") as f:
#     df = pickle.load(f)
# # stair case at around depth , temp 
# # sample = get_data(df, )
# # staircase at around depth 280-400, temp 06-09
# # sample = get_data(df, 4000)
# # stair case at around depth 280-400, temp 04-08
# # sample = get_data(df, 4500)
# # stair case at around depth 340-420, temp 0.6-0.75
# # sample = get_data(df, 24500)
# # stair case at around depth 240-360 , temp .3-.9
# # sample = get_data(df, 124)
# # stair case at around depth 300-440, temp .45-.75
# # sample = get_data(df, 22000)
# # stair case at around depth 260-380, temp .3-.95
# # sample = get_data(df, 23)
# # stair case at around depth 360-440, temp .5-.8
# # sample = get_data(df, 14000)
# # stair case at around depth 300-450, temp .2-.8
# # sample = get_data(df, 13000)
# # stair case at around depth 250-375, temp .15-.8
# # sample = get_data(df, 6000)
# # stair case at around depth 260-380, temp .2-.7
# # sample = get_data(df, 3000)
# # stair case at around depth 360-385, temp .3-.9
# # sample = get_data(df, 25)
# # sample = get_data(df,14500)
# #--------------------------------------------------
# # terrible example 
# # sample = get_data(df, 26170)
# sample = get_data(df, 30000)
# # sample = get_data(df, 15000)
# # sample = get_data(df, 16000)


# smartPlot("tvd", sample)
