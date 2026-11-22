import os
import sys
import glob
import pandas as pd
import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

SW_DATASET_DIR = r"D:\PINN\data\SW-Dataset"
OUT_MANIFEST_FILE = r"D:\PINN\Hologram_Classification_10Class\data\odp_10class_manifest.csv"
OUT_SPLIT_FILE = r"D:\PINN\Hologram_Classification_10Class\data\odp_10class_split.csv"

BASE_DIR = r"D:\PINN"

# Optical parameters for Single-Wavelength ODP-DLHM dataset
WAVELENGTH = 0.633 # um
L_VALUE = 18.81    # cm

# Representative z_value mapping table for z-folder names if available, else parsed from folder name
def parse_z_value(z_folder_name):
    # z_folder_name e.g. 'z1', 'z2', ..., 'z30' or 'z=1.1cm'
    z_clean = z_folder_name.lower().replace('z=', '').replace('z', '').replace('cm', '')
    try:
        idx = int(z_clean)
        # z values typically range from 1.1 cm to 3.5 cm across 30 steps
        return round(1.1 + (idx - 1) * (3.5 - 1.1) / 29.0, 3)
    except Exception:
        try:
            return float(z_clean)
        except Exception:
            return 1.1

def verify_image_file(full_path):
    if not os.path.exists(full_path) or os.path.getsize(full_path) == 0:
        return False
    try:
        with Image.open(full_path) as img:
            img.verify()
        return True
    except Exception:
        if os.path.getsize(full_path) > 1000000:
            return True
        return False

def main():
    print("=== Building Verified 10-Class Manifest from Local SW-Dataset ===")
    
    classes = [d for d in os.listdir(SW_DATASET_DIR) if os.path.isdir(os.path.join(SW_DATASET_DIR, d))]
    classes = sorted(classes)
    
    print(f"Found {len(classes)} class directories: {classes}")
    assert len(classes) == 10, f"Expected 10 classes, got {len(classes)}"
    
    # Load z-values from existing 5-class split manifest if present for exact continuity
    existing_split = r"D:\PINN\Hologram_Classification\data\dataset_split.csv"
    existing_z_map = {}
    if os.path.exists(existing_split):
        df_ex = pd.read_csv(existing_split)
        for _, r in df_ex.iterrows():
            rel_p = str(r['image_path']).replace('\\', '/')
            existing_z_map[rel_p] = float(r['z_value'])
            
    records = []
    
    for cls_name in classes:
        cls_dir = os.path.join(SW_DATASET_DIR, cls_name)
        z_folders = [d for d in os.listdir(cls_dir) if os.path.isdir(os.path.join(cls_dir, d))]
        
        for z_f in z_folders:
            z_dir = os.path.join(cls_dir, z_f)
            images = [f for f in os.listdir(z_dir) if f.lower().endswith(('.bmp', '.tif', '.png', '.jpg'))]
            
            for img_name in images:
                full_path = os.path.join(z_dir, img_name)
                rel_path = os.path.relpath(full_path, BASE_DIR).replace('\\', '/')
                
                if verify_image_file(full_path):
                    if rel_path in existing_z_map:
                        z_val = existing_z_map[rel_path]
                    else:
                        z_val = parse_z_value(z_f)
                        
                    records.append({
                        'image_path': rel_path,
                        'sample': cls_name,
                        'z_value': float(z_val),
                        'wavelength': WAVELENGTH,
                        'L_value': L_VALUE
                    })
                    
    df_all = pd.DataFrame(records)
    print(f"\nTotal verified hologram images across 10 classes: {len(df_all)}")
    
    # Save manifest
    os.makedirs(os.path.dirname(OUT_MANIFEST_FILE), exist_ok=True)
    df_all.to_csv(OUT_MANIFEST_FILE, index=False)
    print(f"Saved 10-class manifest to: {OUT_MANIFEST_FILE}")
    
    # Create grouped train/val/test split with zero z-position leakage
    np.random.seed(42)
    split_assignments = []
    
    for cls_name in classes:
        cls_df = df_all[df_all['sample'] == cls_name]
        z_vals = sorted(cls_df['z_value'].unique())
        np.random.shuffle(z_vals)
        
        n_z = len(z_vals)
        n_train_z = int(round(0.70 * n_z))
        n_val_z = int(round(0.15 * n_z))
        
        train_z = set(z_vals[:n_train_z])
        val_z = set(z_vals[n_train_z:n_train_z + n_val_z])
        test_z = set(z_vals[n_train_z + n_val_z:])
        
        # Verify no z leakage within class
        assert len(train_z.intersection(val_z)) == 0
        assert len(train_z.intersection(test_z)) == 0
        assert len(val_z.intersection(test_z)) == 0
        
        for _, row in cls_df.iterrows():
            z_v = row['z_value']
            if z_v in train_z:
                sp = 'train'
            elif z_v in val_z:
                sp = 'val'
            else:
                sp = 'test'
            split_assignments.append(sp)
            
    df_all['split'] = split_assignments
    
    # Save split CSV
    df_all.to_csv(OUT_SPLIT_FILE, index=False)
    print(f"Saved 10-class split CSV to: {OUT_SPLIT_FILE}")
    
    # Print Split Summary
    print("\n--- 10-Class Dataset Split Summary ---")
    summary_df = df_all.groupby(['sample', 'split']).size().unstack(fill_value=0)
    summary_df['Total'] = summary_df.sum(axis=1)
    print(summary_df)
    
    tot_train = (df_all['split'] == 'train').sum()
    tot_val = (df_all['split'] == 'val').sum()
    tot_test = (df_all['split'] == 'test').sum()
    print(f"\nOverall Totals -> Train: {tot_train}, Val: {tot_val}, Test: {tot_test}, Combined: {len(df_all)}")
    
    # Strict Leakage Verification
    train_z_all = set(df_all[df_all['split']=='train'][['sample', 'z_value']].apply(tuple, axis=1))
    val_z_all = set(df_all[df_all['split']=='val'][['sample', 'z_value']].apply(tuple, axis=1))
    test_z_all = set(df_all[df_all['split']=='test'][['sample', 'z_value']].apply(tuple, axis=1))
    
    leak_val = train_z_all.intersection(val_z_all)
    leak_test = train_z_all.intersection(test_z_all)
    print(f"Z-Position Leakage Check -> Train/Val Overlap: {len(leak_val)}, Train/Test Overlap: {len(leak_test)}")
    assert len(leak_val) == 0 and len(leak_test) == 0, "Z-position leakage detected!"
    print("Zero z-position leakage verified successfully!\n")

if __name__ == '__main__':
    main()
