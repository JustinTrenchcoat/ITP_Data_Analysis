# this script downloads all level 3 high-resolution data
# in the .mat format into a new folder called datasets
import os
import requests
import zipfile
from tqdm import tqdm

# Base URL pattern
base_url = "https://scienceweb.whoi.edu/itp/data/itpsys{0}/itp{0}cormat.zip"

failList = []
# Base directory and datasets directory
base_dir = os.getcwd()
datasets_dir = os.path.join(base_dir, "rawData")
os.makedirs(datasets_dir, exist_ok=True)

# Loop over ITP numbers
for itp_num in tqdm(range(1, 144), desc="Downloading"):
    url = base_url.format(itp_num)
    zip_filename = f"itp{itp_num}cormat.zip"
    zip_path = os.path.join(base_dir, zip_filename)

    try:
        # Download the zip file
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(f"Downloaded: {zip_filename}")

            # Extract into rawData/itp*cormat/
            extract_folder = os.path.join(datasets_dir, f"itp{itp_num}cormat")
            os.makedirs(extract_folder, exist_ok=True)

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_folder)
            print(f"Extracted to: {extract_folder}")

            # Optionally delete the zip file after extraction
            os.remove(zip_path)

        else:
            print(f"Not found or failed to download: {url}")
            failList.append(f"Not found or failed to download: {url}")


    except Exception as e:
        print(f"Error processing ITP {itp_num}: {e}")


print(failList)