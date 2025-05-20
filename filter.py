# this file filters only ITP data that is collected in the Beaufort Gyre
# lon>-160 & lon<-130 & lat>73 & lat<81;

from itp.itp_query import ItpQuery
from itp.profile import Profile
from tqdm import tqdm
import pickle


global good_list
good_list = []
path = r'D:\EOAS\ITP_Data_Analysis\itp_data\itp_final_2025_05_07.db'
# query = ItpQuery(path, system=[1])
# results = query.fetch()

query = ItpQuery(path, latitude=[73, 81], longitude=[-160, -130])
query.set_max_results(100000)

results = query.fetch()
for p in tqdm(results):
    if max(Profile.depth(p))>=400:
        good_list.append(p)


# Save to file
with open("profiles.pkl", "wb") as f:
    pickle.dump(good_list, f)

