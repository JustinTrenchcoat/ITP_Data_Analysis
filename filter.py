# this file filters only ITP data that is collected in the Beaufort Gyre
# lon>-160 & lon<-130 & lat>73 & lat<81;

from itp.itp_query import ItpQuery
from itp.profile import Profile
from tqdm import tqdm
import pickle


global good_list
good_list = []
path = r'D:\EOAS\ITP_Data_Analysis\itp_data\itp_final_2025_05_07.db'

# filter through their built in function.
query = ItpQuery(path, latitude=[73, 81], longitude=[-160, -130])
# set up the search limit since the result list would be huge
query.set_max_results(100000)

# result is about 60000-ish?
results = query.fetch()
# this gives you a progress check, every profile with their deepest measurement reaching 400m or beyond is put into the list.
for p in tqdm(results):
    if max(Profile.depth(p))>=400:
        good_list.append(p)


# Save to a pkl file.
with open("profiles.pkl", "wb") as f:
    pickle.dump(good_list, f)

