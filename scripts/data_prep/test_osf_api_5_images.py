import os
import pandas as pd
import requests
from PIL import Image

MANIFEST_FILE = 'data/odp_manifest.csv'
NODE_ID = 'syht9'
PROVIDER = 'googledrive'

folder_cache = {}

def get_folder_contents(folder_url):
    if folder_url in folder_cache:
        return folder_cache[folder_url]
        
    contents = []
    url = folder_url
    while url:
        resp = requests.get(url, params={'page[size]': 100}, timeout=10)
        if resp.status_code != 200:
            print(f"Error fetching {url}: {resp.status_code}")
            return []
        data = resp.json()
        contents.extend(data.get('data', []))
        url = data.get('links', {}).get('next')
        
    folder_cache[folder_url] = contents
    return contents

def find_item_by_name(contents, name):
    for item in contents:
        if item['attributes']['name'] == name:
            return item
    return None

def resolve_path_to_id(rel_path, root_url):
    parts = rel_path.split('/')
    url = root_url
    
    for i, part in enumerate(parts):
        contents = get_folder_contents(url)
        item = find_item_by_name(contents, part)
        
        if not item:
            print(f"Could not find {part} in {parts[:i]}")
            return None
            
        if i == len(parts) - 1:
            return item
        else:
            url = item['relationships']['files']['links']['related']['href']
            
    return None

def main():
    if not os.path.exists(MANIFEST_FILE):
        return
        
    df = pd.read_csv(MANIFEST_FILE)
    test_df = df.head(5)
    
    root_url = f"https://api.osf.io/v2/nodes/{NODE_ID}/files/{PROVIDER}/"
    success_count = 0
    
    print("\nStarting 5-image test download with cached traversal...")
    for idx, row in test_df.iterrows():
        image_path = row['image_path']
        rel_path = image_path.replace('\\', '/').split('data/')[-1]
        
        print(f"\nResolving: {rel_path}")
        file_item = resolve_path_to_id(rel_path, root_url)
        
        if not file_item:
            print(f"-> Failed to resolve path.")
            continue
            
        file_id = file_item['id']
        download_url = file_item['links']['download']
        
        print(f"-> OSF File ID: {file_id}")
        print(f"-> Download URL: {download_url}")
        
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        resp = requests.get(download_url, timeout=15)
        
        if resp.status_code == 200:
            with open(image_path, 'wb') as f:
                f.write(resp.content)
            
            size_mb = os.path.getsize(image_path) / (1024 * 1024)
            with Image.open(image_path) as img:
                dims = img.size
                mode = img.mode
                
            print(f"-> Verified! Size: {size_mb:.2f} MB, Dims: {dims}, Mode: {mode}")
            success_count += 1
        else:
            print(f"-> Download failed with status {resp.status_code}")
            
    print(f"\nFinished 5-image test. Success: {success_count}/5")

if __name__ == '__main__':
    main()
