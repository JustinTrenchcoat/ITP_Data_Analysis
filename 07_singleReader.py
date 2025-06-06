import matplotlib.pyplot as plt
import numpy as np
import h5py
import traceback
from scipy.interpolate import interp1d
from helper import height

# this script looks through each profile extract depth, temperature and every other measurement, interpolate the values by depth grid of 0.25m
# should it look like a dataframe or?

# set up file path
# full_path = r'D:\EOAS\ITP_Data_Analysis\datasets\itp62cormat\cor0008.mat'
full_path = r'D:\EOAS\ITP_Data_Analysis\datasets\itp1cormat\cor0002.mat'

def read_var(f, varname):
        data = np.array(f[varname])
        if data.dtype == "uint16":
            return data.tobytes().decode('utf-16-le')
        return data.reshape(-1)
        

try:
    with h5py.File(full_path, 'r') as f:
        # Print all root level object names (aka keys) 
        # these can be group or dataset names 
        print("Keys: %s" % f.keys()) 

        pr_filt = read_var(f, 'pr_filt')
        sa_adj = read_var(f, "sa_adj")
        te_adj = read_var(f, 'te_adj')
        lat = read_var(f, "latitude")
        lon = read_var(f, "longitude")

        # Filter out NaNs
        valid_mask = ~np.isnan(sa_adj) & ~np.isnan(pr_filt) & ~np.isnan(te_adj)
        test_mask =  ~np.isnan(pr_filt)
        print(f"valid_mask length:{len(valid_mask)}")
        # sa_cor = sa_cor[valid_mask]
        pr_filt = pr_filt[valid_mask]
        te_adj  = te_adj[valid_mask]

        

        print(f"test_mask length: {len(test_mask)}")



        print(np.max(te_adj))

        # calculate depth
        depth = height(pr_filt, lat)
        dep_max = np.max(depth)
        print(dep_max)

        # depth_index has all indeces of depth array who's value is above 200m
        depth_index = np.where(depth >= 200)[0]

        # temp_max_idx is the index of max value of all te_cor values within the range of depth_index
        temp_idx_max = np.argmax(te_adj[depth_index])
        # temp_max_depth is the value of depth at the max value of te_cor values that are under 200m
        temp_max_depth_idx = depth_index[temp_idx_max]
        print(f"Max temperature below 5 m is at {depth[temp_max_depth_idx]}m, is {te_adj[temp_max_depth_idx]}")

        temp_max_depth = depth[temp_max_depth_idx]

        # if the depth of temp_max beyond 200m, then it is good profile:

        if ((dep_max >= (temp_max_depth+2)) and (73 <= lat <= 81) and (-160 <= lon <= -130)):
            print("test1")
        else:
            print("test2")


        print(f"End Time: {read_var(f, "pedate")}, {read_var(f, "pstop")}")
        print(f"Start Time: {read_var(f, "psdate")}, {read_var(f, "pstart")}")


except Exception as e:
    print(f"Error processing file: {full_path}")
    traceback.print_exc()


# from minisom import MiniSom
# import numpy as np
# import matplotlib.pyplot as plt
# import h5py
# import traceback
# from sklearn.preprocessing import MinMaxScaler
# from helper import height, read_var

# # Path to .mat file
# full_path = r'D:\EOAS\ITP_Data_Analysis\datasets\itp1cormat\cor0002.mat'

# try:
#     with h5py.File(full_path, 'r') as f:
#         print("Keys: %s" % f.keys())

#         # Read variables
#         pr_filt = read_var(f, 'pr_filt')
#         sa_adj = read_var(f, "sa_adj")
#         te_adj = read_var(f, 'te_adj')
#         lat = read_var(f, "latitude")

#         # Filter out NaNs
#         valid_mask = ~np.isnan(sa_adj) & ~np.isnan(pr_filt) & ~np.isnan(te_adj)
#         pr_filt = pr_filt[valid_mask]
#         te_adj = te_adj[valid_mask]

#         # Compute depth and apply range filter
#         depth = height(pr_filt, lat)
#         depth_mask = (depth >= 200) & (depth <= 400)
#         depth = depth[depth_mask]
#         te_adj = te_adj[depth_mask]

#         # Combine and scale features
#         combined = np.column_stack((depth, te_adj))
#         scaler = MinMaxScaler()
#         combined_scaled = scaler.fit_transform(combined)

#         # Initial scatter plot of filtered data
#         plt.scatter(combined[:, 1], combined[:, 0])
#         plt.gca().invert_yaxis()
#         plt.title("Filtered Depth-Temperature Data (200-400 m)")
#         plt.xlabel("Temperature (°C)")
#         plt.ylabel("Depth (m)")
#         plt.show()

#         # Train SOM
#         som = MiniSom(10, 10, combined_scaled.shape[1], sigma=1, learning_rate=0.5,
#                       neighborhood_function='triangle', random_seed=10)
#         som.train(combined_scaled, 100, random_order=False, verbose=True)

#         # # Optional: error tracking
#         # max_iter = 1000
#         # q_error = []
#         # t_error = []
#         # for i in range(max_iter):
#         #     rand_i = np.random.randint(len(combined_scaled))
#         #     som.update(combined_scaled[rand_i], som.winner(combined_scaled[rand_i]), i, max_iter)
#         #     q_error.append(som.quantization_error(combined_scaled))
#         #     t_error.append(som.topographic_error(combined_scaled))

#         # plt.plot(np.arange(max_iter), q_error, label='quantization error')
#         # plt.plot(np.arange(max_iter), t_error, label='topographic error')
#         # plt.xlabel('Iteration')
#         # plt.ylabel('Error')
#         # plt.legend()
#         # plt.title("SOM Training Errors")
#         # plt.show()

#         # Get cluster assignments
#         som_shape = som.get_weights().shape[:2]
#         winner_coordinates = np.array([som.winner(x) for x in combined_scaled]).T
#         cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)

#         # Plot clusters
#         plt.figure(figsize=(10, 8))
#         for c in np.unique(cluster_index):
#             points = combined[cluster_index == c]
#             plt.scatter(points[:, 1], points[:, 0], label=f'cluster={c}', alpha=0.7)

#         # Plot centroids that actually have data points
#         weights = som.get_weights().reshape(-1, combined_scaled.shape[1])
#         weights_clipped = np.clip(weights, 0, 1)
#         centroids_unscaled = scaler.inverse_transform(weights_clipped)

#         unique_clusters = np.unique(cluster_index)
#         for i, centroid in enumerate(centroids_unscaled):
#             if i in unique_clusters:
#                 plt.scatter(centroid[1], centroid[0], marker='x', s=80, color='k')

#         plt.gca().invert_yaxis()
#         plt.xlabel("Temperature (°C)")
#         plt.ylabel("Depth (m)")
#         plt.title("SOM Clusters with Centroids (200-400 m)")
#         plt.legend()
#         plt.show()

# except Exception as e:
#     print(f"Error processing file: {full_path}")
#     traceback.print_exc()
