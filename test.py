from mpl_toolkits.basemap import Basemap
from itp.itp_query import ItpQuery
import matplotlib.pyplot as plt
from itp.profile import Profile
from tqdm import tqdm


global good_list
good_list = []

path = r'D:\EOAS\ITP_package_try\itp_data\itp_final_2025_05_07.db'
# query = ItpQuery(path, system=[1])
# results = query.fetch()

query = ItpQuery(path, latitude=[68, 81.5], longitude=[-170, -130])
query.set_max_results(100000)

results = query.fetch()
print(len(results))
for p in tqdm(results):
    if max(Profile.depth(p))>=400:
        good_list.append(p)
print(len(good_list))