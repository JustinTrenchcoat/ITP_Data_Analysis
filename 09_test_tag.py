import numpy as np
import gsw
import matplotlib.pyplot as plt
import pandas as pd
from helper import *
from scipy.io import loadmat


# load dataset
# just load one profile and see what happens
full_path  = r'gridDataMat\itp1cormat\cor0001.mat'
global temp
global salinity 
global depth

# ds = xr.open_dataset(full_path)
data = loadmat(full_path)  # For pre-v7.3 files
print(data.keys())
# each variable is stored as 1*N array in mat files
temp = data['Temperature'].T.flatten()
salinity = data['Salinity'].T.flatten()
depth = data['Depth'].T.flatten()
max_temp_index = np.nanargmax(temp)
max_temp_depth = depth[max_temp_index]

# Create DataFrame for tagging
df = pd.DataFrame({
    'Depth': depth,
    'Temperature': temp,
    'Salinity': salinity,
    'mixed': None,
    'interface': None,
    'staircase_type': None
    })

def detect_mixed_layers(depth, temperature, salinity, mixed_layer_threshold=0.0002,
                        interface_threshold=0.005):
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


# Select only data within the range
    

mixed_layers, interfaces = detect_mixed_layers(depth, temp, salinity)

all_mixed_layers[0] = mixed_layers
all_interfaces[0] = interfaces

# Function to check overlap between two ranges
def ranges_overlap(a, b):
    return a[0] < b[1] and b[0] < a[1]  # True if ranges intersect

# Loop to find overlaps
overlaps = []
for i, d1 in enumerate(mixed_layers):
    for j, d2 in enumerate(interfaces):
        if ranges_overlap(d1, d2):
            overlaps.append((i, j, d1, d2))

# Display
for i, j, d1, d2 in overlaps:
    print(f"depth[{i}] {d1} overlaps with depth_2[{j}] {d2}")

        
# Number of profiles used for analysis
num_profiles_used = len(all_mixed_layers)
# print(f"Number of profiles used in the analysis: {num_profiles_used}")

# STAIRCASE DETECTION METHOD
staircase_profiles = {}

for profile_idx in all_mixed_layers:
    flagmap = np.zeros(len(depth))

    for (start, end) in all_mixed_layers[profile_idx]:
        start_idx = np.argmax(depth >= start)
        end_idx = np.argmax(depth >= end) - 1
        flagmap[start_idx:end_idx] = 1

    for (start, end) in all_interfaces[profile_idx]:
        start_idx = np.argmax(depth >= start)
        end_idx = np.argmax(depth >= end) - 1
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
profile_idx = 0
sharp_transitions = staircase_profiles.get(profile_idx, {}).get("sharp", [])
mushy_transitions = staircase_profiles.get(profile_idx, {}).get("mushy", [])
smushy_transitions = staircase_profiles.get(profile_idx, {}).get("super_mushy", [])

sharp_depths = [float(depth[i]) for i in sharp_transitions if i < len(depth)]
sharp_temp = [float(temp[i]) for i in sharp_transitions if i < len(temp)]
mushy_depths = [float(depth[i]) for i in mushy_transitions if i < len(depth)]
mushy_temp = [float(temp[i]) for i in mushy_transitions if i < len(temp)]

smushy_depths = [float(depth[i]) for i in smushy_transitions if i < len(depth)]
smushy_temp = [float(temp[i]) for i in smushy_transitions if i < len(temp)]

print(f"Profile {profile_idx} sharp transitions at depths: {sharp_depths}")
print(f"Profile {profile_idx} mushy transitions at depths: {mushy_depths}")
print(f"Profile {profile_idx} super mushy transitions at depths: {smushy_depths}")


# Plot desired profile
fig, ax = plt.subplots(figsize=(5, 7))
ax.invert_yaxis()
ax.set_xlabel("Temperature (°C)")
ax.set_ylabel("Depth (m)")
ax.set_title(f"Profile {profile_idx}: Mixed Layers and Interfaces")
ax.plot(temp, depth, 'k', alpha=0.5, label="Temperature Profile", marker=".")

# Flags to ensure only one legend entry per category
mixed_layer_plotted = False
interface_plotted = False

for (start, end) in all_mixed_layers.get(profile_idx, []):
    start_idx = np.argmax(depth >= start)
    end_idx = np.argmax(depth >= end)
    
    # Add label only once
    if not mixed_layer_plotted:
        ax.plot(temp[start_idx:end_idx], depth[start_idx:end_idx], 
                'b', lw=2, label="Mixed Layer")
        mixed_layer_plotted = True
    else:
        ax.plot(temp[start_idx:end_idx], depth[start_idx:end_idx], 'b', lw=2)

for (start, end) in all_interfaces.get(profile_idx, []):
    start_idx = np.argmax(depth >= start)
    end_idx = np.argmax(depth >= end)
    
    # Add label only once
    if not interface_plotted:
        ax.plot(temp[start_idx:end_idx], depth[start_idx:end_idx], 
                'r', lw=2, label="Interface")
        interface_plotted = True
    else:
        ax.plot(temp[start_idx:end_idx], depth[start_idx:end_idx], 'r', lw=2)

ax.legend()
for i in range(len(mushy_depths)):
    plt.annotate("Mushy Ones", (mushy_temp[i], mushy_depths[i]), 
                 textcoords="data", color='blue',xytext=(mushy_temp[i]+0.002, mushy_depths[i]+0.002), 
                 arrowprops=dict(
                        facecolor='blue',         
                        edgecolor='black',       
                        arrowstyle='->',        
                        lw=2))
for i in range(len(smushy_depths)):
    plt.annotate("Super Mushy Ones", (smushy_temp[i], smushy_depths[i]), 
                 textcoords="data", color='red',xytext=(smushy_temp[i]+0.002, smushy_depths[i]+0.002), 
                 arrowprops=dict(
                        facecolor='red',         
                        edgecolor='black',       
                        arrowstyle='->',        
                        lw=2))

plt.show()