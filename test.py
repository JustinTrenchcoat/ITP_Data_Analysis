import pickle
from helper import *

# Load saved file
with open("df.pkl", "rb") as f:
    df = pickle.load(f)
# stair case at around depth , temp 
# sample = get_data(df, )
# staircase at around depth 280-400, temp 06-09
# sample = get_data(df, 4000)
# stair case at around depth 280-400, temp 04-08
# sample = get_data(df, 4500)
# stair case at around depth 340-420, temp 0.6-0.75
# sample = get_data(df, 24500)
# stair case at around depth 240-360 , temp .3-.9
# sample = get_data(df, 124)
# stair case at around depth 300-440, temp .45-.75
# sample = get_data(df, 22000)
# stair case at around depth 260-380, temp .3-.95
# sample = get_data(df, 23)
# stair case at around depth 360-440, temp .5-.8
# sample = get_data(df, 14000)
# stair case at around depth 300-450, temp .2-.8
# sample = get_data(df, 13000)
# stair case at around depth 250-375, temp .15-.8
# sample = get_data(df, 6000)
# stair case at around depth 260-380, temp .2-.7
# sample = get_data(df, 3000)
# stair case at around depth 360-385, temp .3-.9
# sample = get_data(df, 25)
sample = get_data(df,14500)
#--------------------------------------------------
# terrible example 
# sample = get_data(df, 26170)
# sample = get_data(df, 30000)
# sample = get_data(df, 15000)
# sample = get_data(df, 16000)


smartPlot("tvd", sample)
