import os
import time
import requests
import pandas as pd
from PIL import Image, ImageFile
from concurrent.futures import ThreadPoolExecutor, as_completed

ImageFile.LOAD_TRUNCATED_IMAGES = True

MANIFEST_FILE = r'D:\PINN\data\missing_5classes_manifest.csv'
FAILED_FILE = r'D:\PINN\data\failed_missing_downloads.csv'
NODE_ID = 'syht9'
PROVIDER = 'googledrive'

folder_cache = {}

def get_folder_contents(folder_url):
    if folder_url in folder_cache:
        return folder_cache[folder_url]
        
    contents = []
    url = folder_url
    while url:
        try:
            resp = requests.get(url, params={'page[size]': 100}, timeout=15)
            if resp.status_code != 200:
                return []
            data = resp.json()
            contents.extend(data.get('data', []))
            url = data.get('links', {}).get('next')
        except Exception:
            time.sleep(1)
            continue
            
    folder_cache[folder_url] = contents
    return contents

def get_node_by_name(contents, name):
    for item in contents:
        if item['attributes']['name'] == name:
            return item
    return None

def fetch_z_folder_mapping(sample_name, z_folder):
    z_name = z_folder['attributes']['name']
    z_url = z_folder['relationships']['files']['links']['related']['href']
    z_contents = get_folder_contents(z_url)
    
    sub_map = {}
    for file_item in z_contents:
        if file_item['attributes']['kind'] == 'file':
            file_name = file_item['attributes']['name']
            rel_path = f"SW-Dataset/{sample_name}/{z_name}/{file_name}"
            sub_map[rel_path] = file_item
    return sub_map

def build_full_mapping_fast(samples):
    print("Building OSF file mapping concurrently (Fast Threaded Mapping)...")
    mapping = {}
    
    root_url = f"https://api.osf.io/v2/nodes/{NODE_ID}/files/{PROVIDER}/"
    root_contents = get_folder_contents(root_url)
    
    sw_folder = get_node_by_name(root_contents, 'SW-Dataset')
    if not sw_folder:
        print("Could not find SW-Dataset in OSF root")
        return mapping
        
    sw_url = sw_folder['relationships']['files']['links']['related']['href']
    sw_contents = get_folder_contents(sw_url)
    
    z_tasks = []
    for sample_name in samples:
        sample_folder = get_node_by_name(sw_contents, sample_name)
        if not sample_folder:
            print(f"  Could not find sample {sample_name} in OSF SW-Dataset")
            continue
            
        sample_url = sample_folder['relationships']['files']['links']['related']['href']
        sample_contents = get_folder_contents(sample_url)
        
        for z_folder in sample_contents:
            if z_folder['attributes']['kind'] == 'folder':
                z_tasks.append((sample_name, z_folder))
                
    print(f"  Found {len(z_tasks)} z-folders across {len(samples)} samples. Mapping in parallel...")
    with ThreadPoolExecutor(max_workers=30) as executor:
        futures = [executor.submit(fetch_z_folder_mapping, s, zf) for s, zf in z_tasks]
        for future in as_completed(futures):
            res_map = future.result()
            mapping.update(res_map)
            
    return mapping

def verify_image(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            if img.size != (4640, 3506) or img.mode != 'L':
                return False
        return True
    except Exception:
        if os.path.getsize(path) > 1000000:
            return True
        return False

def download_file_task(row, mapping):
    image_path = row['image_path']
    rel_path = image_path.replace('\\', '/').split('data/')[-1]
    
    if verify_image(image_path):
        return True, row, True
        
    if rel_path not in mapping:
        return False, row, False
        
    download_url = mapping[rel_path]['links']['download']
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = requests.get(download_url, stream=True, timeout=20)
            if resp.status_code == 200:
                with open(image_path, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)
                if verify_image(image_path):
                    return True, row, False
        except Exception:
            time.sleep(2)
            
    return False, row, False

def main():
    if not os.path.exists(MANIFEST_FILE):
        print("Missing manifest file:", MANIFEST_FILE)
        return
        
    df = pd.read_csv(MANIFEST_FILE)
    samples = df['sample'].unique()
    
    mapping = build_full_mapping_fast(samples)
    print(f"Total mapped files for missing 5 classes: {len(mapping)}")
    
    print("\nStarting concurrent image download (25 parallel workers)...")
    
    success_count = 0
    skipped_count = 0
    failed_rows = []
    total_processed = 0
    total_files = len(df)
    
    with ThreadPoolExecutor(max_workers=25) as executor:
        futures = {executor.submit(download_file_task, row, mapping): row for _, row in df.iterrows()}
        
        for idx, future in enumerate(as_completed(futures)):
            success, row, skipped = future.result()
            total_processed += 1
            if success:
                if skipped:
                    skipped_count += 1
                else:
                    success_count += 1
            else:
                failed_rows.append(row)
                
            if total_processed % 200 == 0 or total_processed == total_files:
                print(f"  Downloaded {total_processed}/{total_files} images...")

    failed_df = pd.DataFrame(failed_rows)
    if not failed_df.empty:
        failed_df.to_csv(FAILED_FILE, index=False)
        
    print("\n--- Download Summary ---")
    print(f"Total images requested : {total_files}")
    print(f"Newly downloaded      : {success_count}")
    print(f"Already existing      : {skipped_count}")
    print(f"Failed                : {len(failed_rows)}")

if __name__ == '__main__':
    main()
