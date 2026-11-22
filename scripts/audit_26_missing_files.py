import os
import requests
import json
import time

NODE_ID = 'syht9'
PROVIDER = 'googledrive'
OUT_FILE = r'D:\PINN\data\audit_26_results.txt'

def get_contents(url):
    contents = []
    curr = url
    while curr:
        try:
            r = requests.get(curr, params={'page[size]': 100}, timeout=15)
            if r.status_code != 200:
                break
            data = r.json()
            contents.extend(data.get('data', []))
            curr = data.get('links', {}).get('next')
        except Exception:
            time.sleep(1)
            break
    return contents

def find_item(contents, name):
    for item in contents:
        if item['attributes']['name'] == name:
            return item
    return None

def main():
    lines = []
    lines.append("=== Auditing OSF Repository for exact 26 missing paths ===")
    root_url = f"https://api.osf.io/v2/nodes/{NODE_ID}/files/{PROVIDER}/"
    root_contents = get_contents(root_url)
    
    sw_folder = find_item(root_contents, 'SW-Dataset')
    sw_url = sw_folder['relationships']['files']['links']['related']['href']
    sw_contents = get_contents(sw_url)
    
    # Check 1: 2_Circle
    lines.append("\n--- Auditing 2_Circle missing files in OSF ---")
    circle_folder = find_item(sw_contents, '2_Circle')
    circle_url = circle_folder['relationships']['files']['links']['related']['href']
    circle_z_folders = get_contents(circle_url)
    
    target_circle_files = [
        ('z13', '6_Int.tif'), ('z2', '06_Int.tif'), ('z22', '7_Int.tif'),
        ('z25', '6_Int.tif'), ('z27', '6_Int.tif'), ('z30', '7_Int.tif'),
        ('z7', '6_Int.tif'), ('z9', '6_Int.tif')
    ]
    
    for z_name, fname in target_circle_files:
        zf = find_item(circle_z_folders, z_name)
        if not zf:
            lines.append(f"  [2_Circle] Folder {z_name} NOT found in OSF!")
            continue
        z_url = zf['relationships']['files']['links']['related']['href']
        files_in_z = get_contents(z_url)
        file_item = find_item(files_in_z, fname)
        if file_item:
            lines.append(f"  [2_Circle] {z_name}/{fname} EXISTS in OSF! Download URL: {file_item['links']['download']}")
        else:
            all_names = [f['attributes']['name'] for f in files_in_z]
            lines.append(f"  [2_Circle] {z_name}/{fname} NOT in OSF. Actual files in {z_name}: {all_names}")

    # Check 2: 7_Leaf
    lines.append("\n--- Auditing 7_Leaf missing files in OSF ---")
    leaf_folder = find_item(sw_contents, '7_Leaf')
    leaf_url = leaf_folder['relationships']['files']['links']['related']['href']
    leaf_z_folders = get_contents(leaf_url)
    
    for z_name in ['z8', 'z9']:
        zf = find_item(leaf_z_folders, z_name)
        if not zf:
            lines.append(f"  [7_Leaf] Folder {z_name} NOT found in OSF!")
            continue
        z_url = zf['relationships']['files']['links']['related']['href']
        files_in_z = get_contents(z_url)
        all_names = [f['attributes']['name'] for f in files_in_z]
        lines.append(f"  [7_Leaf] {z_name} in OSF contains {len(all_names)} files: {all_names}")

    # Check 3: 9_Volvox_Algae
    lines.append("\n--- Auditing 9_Volvox_Algae missing files in OSF ---")
    volvox_folder = find_item(sw_contents, '9_Volvox_Algae')
    volvox_url = volvox_folder['relationships']['files']['links']['related']['href']
    volvox_z_folders = get_contents(volvox_url)
    
    for z_name in ['z1', 'z2', 'z3', 'z4']:
        zf = find_item(volvox_z_folders, z_name)
        if not zf:
            lines.append(f"  [9_Volvox_Algae] Folder {z_name} NOT found in OSF!")
            continue
        z_url = zf['relationships']['files']['links']['related']['href']
        files_in_z = get_contents(z_url)
        all_names = [f['attributes']['name'] for f in files_in_z]
        lines.append(f"  [9_Volvox_Algae] {z_name} in OSF contains {len(all_names)} files: {all_names}")

    res_str = "\n".join(lines)
    print(res_str)
    with open(OUT_FILE, 'w') as f:
        f.write(res_str)

if __name__ == '__main__':
    main()
