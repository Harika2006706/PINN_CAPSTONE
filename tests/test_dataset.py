import os
import sys
import torch
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset import get_dataloaders

def main():
    print("--- DATASET MODULE TESTS ---")
    
    manifest_path = 'data/development_manifest.csv'
    if not os.path.exists(manifest_path):
        print(f"Test Failed: Manifest not found at {manifest_path}")
        return
        
    print("1. Testing DataLoader Initialization and Splitting...")
    train_loader, val_loader, train_df, val_df = get_dataloaders(manifest_path, batch_size=2)
    
    print(f"Total training samples: {len(train_df)}")
    print(f"Total validation samples: {len(val_df)}")
    
    print("\n2. Testing Data Leakage Prevention (Z-Value Group Split)...")
    train_z = set(train_df['z_value'].unique())
    val_z = set(val_df['z_value'].unique())
    
    intersection = train_z.intersection(val_z)
    print(f"Train unique Z-values: {len(train_z)}")
    print(f"Val unique Z-values: {len(val_z)}")
    print(f"Intersection of Z-values between train and val: {len(intersection)}")
    if len(intersection) == 0:
        print("-> SUCCESS: No leakage between acquisition groups.")
    else:
        print("-> FAILED: Data leakage detected!")
        
    print("\n3. Testing Data Loading and Batch Shapes...")
    batch = next(iter(train_loader))
    
    images = batch['image']
    z_vals = batch['z_value']
    w_vals = batch['wavelength']
    l_vals = batch['L_value']
    
    print(f"Image Batch Shape: {images.shape} (Expected: B, 1, 3506, 4640)")
    print(f"Image dtype: {images.dtype}")
    print(f"Image intensity range: min = {images.min().item():.4f}, max = {images.max().item():.4f}")
    
    print(f"Z-Value Batch Shape: {z_vals.shape}")
    print(f"Wavelength Batch Shape: {w_vals.shape}")
    print(f"L_Value Batch Shape: {l_vals.shape}")
    
    # Assertions
    assert images.dim() == 4, "Images must be 4D tensors"
    assert images.shape[1] == 1, "Images must be single-channel grayscale"
    assert images.shape[2:] == (3506, 4640), "Images must preserve original dimensions"
    assert not torch.isnan(images).any(), "Images contain NaNs"
    assert len(intersection) == 0, "Data leakage between splits"
    
    print("\n4. One Batch Example Content:")
    for i in range(len(images)):
        print(f"  Sample [{i}]:")
        print(f"    Path: {batch['image_path'][i]}")
        print(f"    Sample: {batch['sample'][i]}")
        print(f"    Z-value: {z_vals[i].item()}")
        print(f"    Wavelength: {w_vals[i].item()}")
        print(f"    L-value: {l_vals[i].item()}")
        
    print("\n--- ALL TESTS PASSED ---")

if __name__ == '__main__':
    main()
