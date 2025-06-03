import os
import shutil

# Path to the datasets directory
datasets_dir = "goldData"

# Initialize counter
total_profiles = 0

# Loop over all itp*cormat folders
for folder_name in os.listdir(datasets_dir):
    folder_path = os.path.join(datasets_dir, folder_name)

    if os.path.isdir(folder_path) and folder_name.startswith("itp") and folder_name.endswith("cormat"):
        # Get all .mat files
        mat_files = [f for f in os.listdir(folder_path) if f.endswith(".mat")]
        profile_count = len(mat_files)

        if profile_count == 0:
            # Delete empty folder
            shutil.rmtree(folder_path)
            print(f"Deleted empty folder: {folder_name}")
        else:
            total_profiles += profile_count
            print(f"{folder_name}: {profile_count} profiles")

# Print total
print(f"\nTotal number of remaining profiles: {total_profiles}")
