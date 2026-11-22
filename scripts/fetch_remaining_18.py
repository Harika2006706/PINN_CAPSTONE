import os
import time
import requests
import pandas as pd
from PIL import Image

MANIFEST_FILE = r'D:\PINN\data\missing_5classes_manifest.csv'
NODE_ID = 'syht9'
PROVIDER = 'googledrive'

def verify_image(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        if os.path.getsize(path) > 1000000:
            return True
        return False

def get_folder_contents(folder_url):
    contents = []
    url = folder_url
    while url:
        try:
            resp = requests.get(url, params={'page[size]': 100}, timeout=20)
            if resp.status_code != 200:
                return []
            data = resp.json()
            contents.extend(data.get('data', []))
            url = data.get('links', {}).get('next')
        except Exception:
            time.sleep(2)
            continue
    return contents

def get_node_by_name(contents, name):
    for item in contents:
        if item['attributes']['name'] == name:
            return item
    return None

def main():
    df = pd.read_csv(MANIFEST_FILE)
    missing_rows = [r for _, r in df.iterrows() if not verify_image(r['image_path'])]
    print(f"Total remaining missing images to download: {len(missing_rows)}")
    if not missing_rows:
        print("All images already present and verified!")
        return

    print("Fetching OSF folder tree for remaining images...")
    root_url = f"https://api.osf.io/v2/nodes/{NODE_ID}/files/{PROVIDER}/"
    root_contents = get_folder_contents(root_url)
    sw_folder = get_node_by_name(root_contents, 'SW-Dataset')
    sw_url = sw_folder['relationships']['files']['links']['related']['href']
    sw_contents = get_folder_contents(sw_url)

    sample_contents_cache = {}

    success_count = 0
    for idx, row in enumerate(missing_rows):
        img_path = row['image_path']
        sample_name = row['sample']
        rel_path = img_path.replace('\\', '/').split('data/')[-1]
        parts = rel_path.split('/') # ['SW-Dataset', sample_name, z_folder, file_name]
        z_name = parts[2]
        file_name = parts[3]

        if sample_name not in sample_contents_cache:
            sf = get_node_by_name(sw_contents, sample_name)
            s_url = sf['relationships']['files']['links']['related']['href']
            sample_contents_cache[sample_name] = get_folder_contents(s_url)

        z_folder = get_node_by_name(sample_contents_cache[sample_name], z_name)
        if not z_folder:
            print(f"Could not find z folder {z_name} for {img_path}")
            continue

        z_url = z_folder['relationships']['files']['links']['related']['href']
        z_contents = get_folder_contents(z_url)
        file_node = get_node_by_name(z_contents, file_name)

        if not file_node:
            print(f"Could not find file node for {img_path}")
            continue

        dl_url = file_node['links']['download']
        os.makedirs(os.path.dirname(img_path), exist_ok=True)

        print(f"[{idx+1}/{len(missing_rows)}] Downloading {rel_path}...")
        for attempt in range(5):
            try:
                resp = requests.get(dl_url, stream=True, timeout=30)
                if resp.status_code == 200:
                    with open(img_path, 'wb') as f:
                        for chunk in resp.iter_content(chunk_size=65536):
                            if chunk:
                                f.write(chunk)
                    if verify_image(img_path):
                        print(f"  Successfully downloaded: {rel_path}")
                        success_count += 1
                        break
            except Exception as e:
                print(f"  Attempt {attempt+1} failed: {e}")
                time.sleep(3)

    print(f"\nDone. Successfully downloaded {success_count}/{len(missing_rows)} remaining images.")

if __name__ == '__main__':
    main()
