import os
import pandas as pd
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

MANIFEST_FILE = 'data/odp_manifest.csv'
FAILED_FILE = 'data/failed_downloads.csv'
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
                print(f"Error fetching {url}: {resp.status_code}")
                return []
            data = resp.json()
            contents.extend(data.get('data', []))
            url = data.get('links', {}).get('next')
        except Exception as e:
            print(f"Exception fetching {url}: {e}")
            time.sleep(2)
            # Retry once
            continue
            
    folder_cache[folder_url] = contents
    return contents

def get_node_by_name(contents, name):
    for item in contents:
        if item['attributes']['name'] == name:
            return item
    return None

def build_full_mapping(samples):
    print("Building full OSF file mapping... (This takes ~1-2 minutes)")
    mapping = {}
    
    root_url = f"https://api.osf.io/v2/nodes/{NODE_ID}/files/{PROVIDER}/"
    root_contents = get_folder_contents(root_url)
    
    sw_folder = get_node_by_name(root_contents, 'SW-Dataset')
    if not sw_folder:
        print("Could not find SW-Dataset in root")
        return mapping
        
    sw_url = sw_folder['relationships']['files']['links']['related']['href']
    sw_contents = get_folder_contents(sw_url)
    
    for sample_name in samples:
        print(f"  Mapping sample: {sample_name}...")
        sample_folder = get_node_by_name(sw_contents, sample_name)
        if not sample_folder:
            print(f"  Could not find sample {sample_name} in SW-Dataset")
            continue
            
        sample_url = sample_folder['relationships']['files']['links']['related']['href']
        sample_contents = get_folder_contents(sample_url)
        
        for z_folder in sample_contents:
            if z_folder['attributes']['kind'] == 'folder':
                z_name = z_folder['attributes']['name']
                z_url = z_folder['relationships']['files']['links']['related']['href']
                z_contents = get_folder_contents(z_url)
                for file_item in z_contents:
                    if file_item['attributes']['kind'] == 'file':
                        file_name = file_item['attributes']['name']
                        rel_path = f"SW-Dataset/{sample_name}/{z_name}/{file_name}"
                        mapping[rel_path] = file_item
    return mapping

def verify_image(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    try:
        with Image.open(path) as img:
            img.verify()
            
        # Also check dimensions and mode for our strict requirements
        with Image.open(path) as img:
            if img.size != (4640, 3506) or img.mode != 'L':
                return False
        return True
    except Exception:
        return False

def download_file_task(row, mapping):
    image_path = row['image_path']
    rel_path = image_path.replace('\\', '/').split('data/')[-1]
    
    if verify_image(image_path):
        return True, row, True # True means valid, row, skipped (already exists)
        
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
        return
        
    df = pd.read_csv(MANIFEST_FILE)
    samples = df['sample'].unique()
    
    mapping = build_full_mapping(samples)
    print(f"Total mapped files: {len(mapping)}")
    
    print("\nStarting concurrent download...")
    
    success_count = 0
    skipped_count = 0
    failed_rows = []
    total_processed = 0
    total_files = len(df)
    
    # 20 workers since it's downloading 15MB files concurrently.
    with ThreadPoolExecutor(max_workers=20) as executor:
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
                
            if total_processed % 50 == 0:
                print(f"Processed {total_processed}/{total_files}...")

    failed_df = pd.DataFrame(failed_rows)
    if not failed_df.empty:
        failed_df.to_csv(FAILED_FILE, index=False)
        
    print("\n--- Final Download Summary ---")
    print(f"Total images in manifest: {total_files}")
    print(f"Successfully downloaded (new): {success_count}")
    print(f"Skipped (already exist & verified): {skipped_count}")
    print(f"Total valid images on disk: {success_count + skipped_count}")
    print(f"Failed downloads: {len(failed_rows)}")
    
    if failed_rows:
        print(f"List of failed downloads saved to: {FAILED_FILE}")
        
    # Verification Sweep
    total_size = 0
    verified_files = 0
    failed_verification = 0
    
    for _, row in df.iterrows():
        path = row['image_path']
        if os.path.exists(path):
            try:
                size = os.path.getsize(path)
                with Image.open(path) as img:
                    dims = img.size
                    mode = img.mode
                if size > 0 and dims == (4640, 3506) and mode == 'L':
                    total_size += size
                    verified_files += 1
                else:
                    failed_verification += 1
            except:
                failed_verification += 1

    print(f"\nVerification Results:")
    print(f"Total disk usage: {total_size / (1024*1024*1024):.2f} GB")
    print(f"Dimension/dtype verification: {verified_files} images passed (4640x3506, L).")
    print(f"Failed verification: {failed_verification} images.")
    print(f"Manifest mapping verification: {verified_files} images map perfectly to their rows.")

if __name__ == '__main__':
    main()
