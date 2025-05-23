# very sus ChatGPT code: do not use until fully figure out what it does and wait until fram is back and running. 
# might be better to be done on a machine that doesnt get turned off that often.


import os
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from tqdm import tqdm
from zipfile import ZipFile

# Base WHOI URL and local download folder
base_url = 'https://scienceweb.whoi.edu/itp/data/'
download_root = 'organized_cormat_data'

os.makedirs(download_root, exist_ok=True)

# Step 1: Get the main data page
main_page = requests.get(base_url)
soup = BeautifulSoup(main_page.text, 'html.parser')

# Step 2: Find all itpsys* directories
subdirs = [a['href'] for a in soup.find_all('a', href=True) if a['href'].startswith('itpsys')]

# Step 3: Process each itpsys folder
for subdir in tqdm(subdirs, desc='Scanning itpsys folders'):
    subdir_url = urljoin(base_url, subdir)
    local_subdir = os.path.join(download_root, subdir.strip('/'))
    os.makedirs(local_subdir, exist_ok=True)

    sub_resp = requests.get(subdir_url)
    sub_soup = BeautifulSoup(sub_resp.text, 'html.parser')

    for link in sub_soup.find_all('a', href=True):
        href = link['href']
        if 'cormat.zip' in href:
            zip_url = urljoin(subdir_url, href)
            local_zip_path = os.path.join(local_subdir, href)

            # Skip download if file already exists
            if not os.path.exists(local_zip_path):
                print(f'Downloading {zip_url}')
                r = requests.get(zip_url, stream=True)
                with open(local_zip_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

            # Extract only .mat files
            print(f'Extracting {local_zip_path}')
            with ZipFile(local_zip_path, 'r') as zip_ref:
                for file in zip_ref.namelist():
                    if file.endswith('.mat'):
                        zip_ref.extract(file, local_subdir)

            # Optional: remove the .zip file to save space
            # os.remove(local_zip_path)
