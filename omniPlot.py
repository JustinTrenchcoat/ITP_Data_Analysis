from itp.itp_query import ItpQuery
import matplotlib.pyplot as plt
from itp.profile import Profile
import argparse

# set up file path
path = r'D:\EOAS\ITP_Data_Analysis\itp_data\itp_final_2025_05_07.db'




def smartPlot(plotType, sample, sysNum, profNum):
    if plotType == "tvd":
        depth = Profile.depth(sample)
        temp = Profile.potential_temperature(sample)
        time = Profile.python_datetime(sample)
        time = time.strftime("%Y-%m-%d %H:%M:%S")

        plotHelper(temp, depth, "Temperature", "Depth", sysNum,profNum,time)

        plt.savefig(f"plots/temp_vs_depth_sys_{sysNum}_prof_{profNum}.png")
        plt.show()
    elif plotType == "svd":
        depth = Profile.depth(sample)
        salinity = Profile.absolute_salinity(sample)
        time = Profile.python_datetime(sample)
        time = time.strftime("%Y-%m-%d %H:%M:%S")

        plotHelper(salinity, depth, "Salinity", "Depth", sysNum,profNum,time)
        plt.savefig(f"plots/sal_vs_depth_sys_{sysNum}_prof_{profNum}.png")
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

# set up possible arguments?
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # system number argument
    parser.add_argument(
        "-s", "--sysNum", type=int,
        default=1,
        help="ITP System Number, range from 1 to 132",
    )
    # profile number
    parser.add_argument(
        "-p", "--profNum", type=int,
        default=0,
        help="profile number needed from the ITP system",
    )
    # set up type of plots needed.
    # temp vs depth plot
    parser.add_argument(
        "-tvd", "--temp_depth",
        default=False,
        help="plot the temperature vs depth plot if True",
    )
    # salinity vs depth plot
    parser.add_argument(
        "-svd", "--salinity_depth",
        default=False,
        help="plot the salinity vs depth plot if True",
    )

    args = parser.parse_args()
    # load in parameters:
    # system Number:
    sysNum = args.sysNum
    profNum = args.profNum
    tvd = args.temp_depth
    svd = args.salinity_depth

    # load dataset
    query = ItpQuery(path, system=[sysNum])
    query.set_max_results(10000)
    results = query.fetch()
    sample = results[profNum]

    if tvd:
         smartPlot("tvd", sample, sysNum, profNum)

    if svd:
         smartPlot("svd", sample, sysNum,profNum)





