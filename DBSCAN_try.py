from sklearn.cluster import DBSCAN
from itp.itp_query import ItpQuery
import matplotlib.pyplot as plt
from itp.profile import Profile
import numpy as np
from sklearn.preprocessing import StandardScaler



# Sample data
path = r'D:\EOAS\ITP_package_try\itp_data\itp_final_2025_05_07.db'
query = ItpQuery(path, system=[1])
results = query.fetch()

longitude = [p.longitude for p in results]
    
latitude = [p.latitude for p in results]
    
sample = results[0]

depth = Profile.depth(sample)
temp = Profile.conservative_temperature(sample)

X = np.column_stack((depth, temp))
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

dbscan = DBSCAN(eps=0.5, min_samples=10)
dbscan.fit(X_scaled)
# Yellowbrick is designed to work with K-Means and not with DBSCAN.
# So it needs the number of clusters stored in n_clusters
# It also needs `predict` method to be implemented.
# So I'm implementing it here so that we can use Yellowbrick to show Silhouette plots.
n_clusters = len(set(dbscan.labels_))
print(n_clusters)

# Plotting
plt.plot(X_scaled[:,1], X_scaled[:,0], marker='o',linestyle='dashed',
     linewidth=2, markersize=12)
plt.xlabel("Temperature")
plt.ylabel("Depth")
plt.title("Temperature vs. Depth for 2025_05_07.db file system 001")
plt.grid(True)
plt.gca().invert_yaxis()

# Optional: Rotate date labels for clarity
plt.xticks(rotation=45)

plt.show()
