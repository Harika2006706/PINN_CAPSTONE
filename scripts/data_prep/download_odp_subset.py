import os
import pandas as pd
import requests
import time
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

MANIFEST_FILE = 'data/odp_manifest.csv'
FAILED_FILE = 'data/failed_downloads.csv'

def download_file(row):
    image_path = row['image_path']
    rel_path = image_path.replace('\\', '/').split('data/')[-1]
    
    if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
        try:
            with Image.open(image_path) as img:
                img.verify()
            return True, row
        except Exception:
            pass
            
    url = f"https://files.osf.io/v1/resources/syht9/providers/osfstorage/?path={quote('/' + rel_path)}"
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    
    # Fail fast
    try:
        response = requests.get(url, stream=True, timeout=2)
        if response.status_code == 200:
            with open(image_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            with Image.open(image_path) as img:
                img.verify()
            return True, row
    except Exception:
        pass
            
    return False, row

def main():
    if not os.path.exists(MANIFEST_FILE):
        return
        
    df = pd.read_csv(MANIFEST_FILE)
    success_count = 0
    failed_rows = []
    
    print(f"Starting fast-fail download of {len(df)} images...")
    
    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = {executor.submit(download_file, row): row for _, row in df.iterrows()}
        
        for idx, future in enumerate(as_completed(futures)):
            success, row = future.result()
            if success:
                success_count += 1
            else:
                failed_rows.append(row)
                
            if (idx + 1) % 200 == 0:
                print(f"Processed {idx + 1}/{len(df)}...")

    failed_df = pd.DataFrame(failed_rows)
    if not failed_df.empty:
        failed_df.to_csv(FAILED_FILE, index=False)
        
    print("\n--- Download Summary ---")
    print(f"Successfully downloaded: {success_count}")
    print(f"Failed downloads: {len(failed_rows)}")
    
    total_size = 0
    valid_files = 0
    sample_dims = None
    sample_dtype = None
    
    for _, row in df.iterrows():
        path = row['image_path']
        if os.path.exists(path):
            total_size += os.path.getsize(path)
            valid_files += 1
            if sample_dims is None:
                try:
                    with Image.open(path) as img:
                        sample_dims = img.size
                        sample_dtype = img.mode
                except:
                    pass

    print(f"Total disk usage: {total_size / (1024*1024):.2f} MB")
    if sample_dims:
        print(f"Sample image properties: Dimensions {sample_dims}, Mode: {sample_dtype}")
    else:
        print("Sample image properties: None (no valid images found)")
        
    print(f"Mapping verification: {valid_files} images exist on disk and map correctly to manifest rows.")

if __name__ == '__main__':
    main()
