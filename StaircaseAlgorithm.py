import numpy as np
import xarray as xr
import gsw
import matplotlib.pyplot as plt
import pandas as pd

# Load dataset
dataset_path = "/Users/kat/Documents/UBC/Courses/PRODIGY-24/ResearchPlotting/datasets/updated_dataset_with_salinity.nc"
ds = xr.open_dataset(dataset_path)

# Extract the time variable
time = ds["time"].values.astype("datetime64[s]").astype(object)

# Create a DataFrame with profile numbers and times
df = pd.DataFrame({
    "profile_number": np.arange(len(time)),
    "time": time
})

# Calculate duration between profiles
df["duration_since_last_profile_seconds"] = df["time"].diff().dt.total_seconds()

# Save to CSV
csv_path = "/Users/kat/Documents/UBC/Courses/PRODIGY-24/ResearchPlotting/Summer Students/profile_times.csv"
df.to_csv(csv_path, index=False)

print(f"Saved profile times and durations to {csv_path}")


# import xarray as xr

# # Load the dataset
# file_path = "/Users/kat/Documents/UBC/Courses/PRODIGY-24/ResearchPlotting/datasets/updated_dataset_with_corrected_salinity.nc"
# ds = xr.open_dataset(file_path)

# # Check the variables
# print(ds)

# # Inspect the time variable
# print(ds['time'])

# # Optionally, convert it to datetime values
# time_values = ds['time'].values
# print(time_values)

# Extract variables
pressure = ds["pressure"].values  
temperature = ds["temperature"].values
salinity = ds["salinity"].values
latitude = ds["latitude"].values  

# Convert pressure to depth
depth = -gsw.z_from_p(pressure, latitude)

# Number of profiles
time_dim = ds.sizes["time"]

# Total number of profiles in the dataset
# total_profiles = ds.sizes["time"]
# print(f"Total number of profiles in the dataset: {total_profiles}")

# Store selected depth ranges for each profile
selected_depth_ranges = {}

# for every profile:
for t in range(time_dim):
    # extract temp and salinity
    temp_profile = temperature[:, t]
    depth_profile = depth[:, t]

    try:
        # grab index for max temp
        max_temp_index = np.nanargmax(temp_profile)
        # grab depth value at max temp
        max_temp_depth = depth_profile[max_temp_index]
    except:
        max_temp_depth = 400  # fallback if entire profile is NaN

    # this is the upper limit for depth
    upper_depth_limit = max_temp_depth - 200
    if upper_depth_limit < 0:
        upper_depth_limit = 0

    #t-th depth range would be from upper to max temp depth
    selected_depth_ranges[t] = (upper_depth_limit, max_temp_depth)

# print how many profiles have been selected
num_selected_profiles = len(selected_depth_ranges)
print(f"Number of profiles with selected depth ranges: {num_selected_profiles}")

# print("Selected depth ranges for each profile:")
# for profile, (upper, lower) in selected_depth_ranges.items():
    # print(f"Profile {profile}: {upper:.2f}m to {lower:.2f}m")

# Function to detect mixed layers and interfaces within the selected depth range
def detect_mixed_layers(depth, temperature, salinity, mixed_layer_threshold=0.0002, interface_threshold=0.005): # 0.0005, 0.005
    # list of temp vs depth gradients
    temp_gradient = np.gradient(temperature, depth)

    mixed_layers = []
    interfaces = []
    in_mixed_layer = False
    in_interface = False
    start_depth = None

    for i in range(1, len(temp_gradient) - 1):
        slope_prev = (temperature[i] - temperature[i - 1]) / (depth[i] - depth[i - 1])

        # Mixed layer detection
        # if the gradient at point i is below the threshold:
        # if in-mixed_layer is true, skip i-th point, if in_mixed_layer is false:
        # set start_depth at i-th point
        # check slope with prev point: if slope below threshold, start_depth reset to i-1 th point.
        # set in_mixed_layer to true.
        if abs(temp_gradient[i]) < mixed_layer_threshold:
            if not in_mixed_layer:
                start_depth = depth[i]
                if slope_prev is not None and abs(slope_prev) < mixed_layer_threshold:
                    start_depth = depth[i - 1]
                in_mixed_layer = True
        else:
            # if gradient >= to threshold;
            # set new variable end_depth to ith point
            # check if prev slope is below threshold. if still above, then directly append the set (start, end) to set of mixedlayers, reset mixed_layer to false
            if in_mixed_layer:
                end_depth = depth[i]
                if slope_prev is not None and abs(slope_prev) < mixed_layer_threshold:
                    end_depth = depth[i + 1]
                mixed_layers.append((start_depth, end_depth))
                in_mixed_layer = False

        # Interface detection
        if abs(temp_gradient[i]) > interface_threshold:
            if not in_interface:
                start_depth = depth[i]
                if slope_prev is not None and abs(slope_prev) > interface_threshold:
                    start_depth = depth[i - 1]
                in_interface = True
        else:
            if in_interface:
                end_depth = depth[i]
                if slope_prev is not None and abs(slope_prev) > interface_threshold:
                    end_depth = depth[i + 1]
                interfaces.append((start_depth, end_depth))
                in_interface = False

    return mixed_layers, interfaces

# Storage for detected layers
all_mixed_layers = {}
all_interfaces = {}

# Loop over profiles but only within selected depth ranges
for t in range(time_dim):
    if t not in selected_depth_ranges:
        continue  # Skip profiles without a selected depth range
    
    upper_depth_limit, max_temp_depth = selected_depth_ranges[t]

    # Select only data within the range
    valid_indices = np.where((depth[:, t] >= upper_depth_limit) & (depth[:, t] <= max_temp_depth))[0]
    
    if len(valid_indices) > 10:  # Ensure there's enough data
        depth_subset = depth[valid_indices, t]
        temp_subset = temperature[valid_indices, t]
        salinity_subset = salinity[valid_indices, t]

        mixed_layers, interfaces = detect_mixed_layers(depth_subset, temp_subset, salinity_subset)

        all_mixed_layers[t] = mixed_layers
        all_interfaces[t] = interfaces
        
# Number of profiles used for analysis
num_profiles_used = len(all_mixed_layers)
# print(f"Number of profiles used in the analysis: {num_profiles_used}")

# STAIRCASE DETECTION METHOD
staircase_profiles = {}

for profile_idx in all_mixed_layers:
    flagmap = np.zeros(len(depth[:, profile_idx]))

    for (start, end) in all_mixed_layers[profile_idx]:
        start_idx = np.argmax(depth[:, profile_idx] >= start)
        end_idx = np.argmax(depth[:, profile_idx] >= end) - 1
        flagmap[start_idx:end_idx] = 1

    for (start, end) in all_interfaces[profile_idx]:
        start_idx = np.argmax(depth[:, profile_idx] >= start)
        end_idx = np.argmax(depth[:, profile_idx] >= end) - 1
        flagmap[start_idx:end_idx] = 2

    sharp_transition_indices = []
    mushy_transition_indices = []
    super_mushy_transition_indices = []
    
    prev_flag_index = 0
    for i in range(1, len(flagmap)):
        curr = flagmap[i]
        prev = flagmap[prev_flag_index]
        
        if curr == 0:
            continue
        
        if prev != curr:
            num_zeros_in_between = i - prev_flag_index - 1
            if num_zeros_in_between == 0:
                sharp_transition_indices.append(i)
            if num_zeros_in_between == 1:
                mushy_transition_indices.append(i)
            if 2 <= num_zeros_in_between <= 4:
                super_mushy_transition_indices.append(i)
                
        prev_flag_index = i

    staircase_profiles[profile_idx] = {
        "sharp": sharp_transition_indices,
        "mushy": mushy_transition_indices,
        "super_mushy": super_mushy_transition_indices,
    }

# Example: Get sharp transition depths for a profile
profile_idx = 62
sharp_transitions = staircase_profiles.get(profile_idx, {}).get("sharp", [])
sharp_depths = [float(depth[i, profile_idx]) for i in sharp_transitions if i < len(depth[:, profile_idx])]
# print(f"Profile {profile_idx} sharp transitions at depths: {sharp_depths}")

# Plot desired profile
fig, ax = plt.subplots(figsize=(5, 7))
ax.invert_yaxis()
ax.set_xlabel("Temperature (°C)")
ax.set_ylabel("Depth (m)")
ax.set_title(f"Profile {profile_idx}: Mixed Layers and Interfaces")
ax.plot(temperature[:, profile_idx], depth[:, profile_idx], 'k', alpha=0.5, label="Temperature Profile")

# Flags to ensure only one legend entry per category
mixed_layer_plotted = False
interface_plotted = False

for (start, end) in all_mixed_layers.get(profile_idx, []):
    start_idx = np.argmax(depth[:, profile_idx] >= start)
    end_idx = np.argmax(depth[:, profile_idx] >= end)
    
    # Add label only once
    if not mixed_layer_plotted:
        ax.plot(temperature[start_idx:end_idx, profile_idx], depth[start_idx:end_idx, profile_idx], 
                'b', lw=2, label="Mixed Layer")
        mixed_layer_plotted = True
    else:
        ax.plot(temperature[start_idx:end_idx, profile_idx], depth[start_idx:end_idx, profile_idx], 'b', lw=2)

for (start, end) in all_interfaces.get(profile_idx, []):
    start_idx = np.argmax(depth[:, profile_idx] >= start)
    end_idx = np.argmax(depth[:, profile_idx] >= end)
    
    # Add label only once
    if not interface_plotted:
        ax.plot(temperature[start_idx:end_idx, profile_idx], depth[start_idx:end_idx, profile_idx], 
                'r', lw=2, label="Interface")
        interface_plotted = True
    else:
        ax.plot(temperature[start_idx:end_idx, profile_idx], depth[start_idx:end_idx, profile_idx], 'r', lw=2)

ax.legend()
plt.show()