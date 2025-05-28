import pickle
import matplotlib.pyplot as plt
import numpy as np

# Load the depth differences
with open("depth_differences.pkl", "rb") as f:
    all_depth_differences = pickle.load(f)

# Convert to NumPy array in case it's a list
all_depth_differences = np.array(all_depth_differences)

# Basic statistics
print("Statistical Summary of Depth Differences:")
print(f"Count         : {len(all_depth_differences)}")
print(f"Min           : {np.min(all_depth_differences)}")
print(f"Max           : {np.max(all_depth_differences)}")
print(f"Mean          : {np.mean(all_depth_differences)}")
print(f"Median        : {np.median(all_depth_differences)}")
print(f"Std Dev       : {np.std(all_depth_differences)}")
print(f"Variance      : {np.var(all_depth_differences)}")
print(f"25th Percentile (Q1): {np.percentile(all_depth_differences, 25)}")
print(f"75th Percentile (Q3): {np.percentile(all_depth_differences, 75)}")
print(f"IQR           : {np.percentile(all_depth_differences, 75) - np.percentile(all_depth_differences, 25)}")



# Plot the histogram
plt.figure(figsize=(10, 6))
plt.hist(all_depth_differences, bins=500, range=(0,6), edgecolor='black')  # adjust bin count or bin edges here
plt.title("Histogram of Depth Differences Between Consecutive Data Points")
plt.xlabel("Depth Difference")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()
