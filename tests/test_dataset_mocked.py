import os
import sys
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit

# We mock torch because the environment's network throttles the 122MB wheel download.
# We will directly test the logic, splits, and PIL/Numpy shapes.
import sys
from unittest.mock import MagicMock

class MockDataset:
    def __init__(self, *args, **kwargs):
        pass

class MockDataLoader:
    def __init__(self, dataset, batch_size=1, *args, **kwargs):
        self.dataset = dataset
        self.batch_size = batch_size
        
    def __iter__(self):
        # Yield one batch
        batch_images = []
        batch_z = []
        batch_w = []
        batch_l = []
        batch_s = []
        batch_paths = []
        
        for i in range(min(self.batch_size, len(self.dataset))):
            item = self.dataset[i]
            batch_images.append(item['image'])
            batch_z.append(item['z_value'])
            batch_w.append(item['wavelength'])
            batch_l.append(item['L_value'])
            batch_s.append(item['sample'])
            batch_paths.append(item['image_path'])
            
        yield {
            'image': np.stack([img.numpy() if hasattr(img, 'numpy') else img for img in batch_images]),
            'z_value': np.array([z.item() if hasattr(z, 'item') else z for z in batch_z]),
            'wavelength': np.array([w.item() if hasattr(w, 'item') else w for w in batch_w]),
            'L_value': np.array([l.item() if hasattr(l, 'item') else l for l in batch_l]),
            'sample': batch_s,
            'image_path': batch_paths
        }

# Inject mocks
torch_mock = MagicMock()
torch_mock.tensor = lambda x, dtype=None: x
torch_mock.from_numpy = lambda x: MagicMock(unsqueeze=lambda dim: np.expand_dims(x, axis=dim))
sys.modules['torch'] = torch_mock
sys.modules['torch.utils'] = MagicMock()
sys.modules['torch.utils.data'] = MagicMock()
sys.modules['torch.utils.data'].Dataset = MockDataset
sys.modules['torch.utils.data'].DataLoader = MockDataLoader

# Now import the actual implementation
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dataset import get_dataloaders

def main():
    print("--- DATASET MODULE TESTS ---")
    
    manifest_path = 'data/development_manifest.csv'
    if not os.path.exists(manifest_path):
        print(f"Test Failed: Manifest not found at {manifest_path}")
        return
        
    print("1. Testing DataLoader Initialization and Splitting...")
    train_loader, val_loader, train_df, val_df = get_dataloaders(manifest_path, batch_size=4)
    
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
        
    print("\n3. Testing Z-Value Distribution...")
    print("Train Z-value distribution (first 5):")
    print(train_df['z_value'].value_counts().head(5).to_string())
    print("Val Z-value distribution (first 5):")
    print(val_df['z_value'].value_counts().head(5).to_string())
        
    print("\n4. Testing Data Loading and Batch Shapes...")
    batch = next(iter(train_loader))
    
    images = batch['image']
    z_vals = batch['z_value']
    w_vals = batch['wavelength']
    l_vals = batch['L_value']
    
    print(f"Image Batch Shape: {images.shape} (Expected: B, 1, 3506, 4640)")
    print(f"Image dtype: {images.dtype}")
    print(f"Image intensity range: min = {images.min():.4f}, max = {images.max():.4f}")
    
    print(f"Z-Value Batch Shape: {z_vals.shape}")
    print(f"Wavelength Batch Shape: {w_vals.shape}")
    print(f"L_Value Batch Shape: {l_vals.shape}")
    
    # Assertions
    assert len(images.shape) == 4, "Images must be 4D arrays (B, C, H, W)"
    assert images.shape[1] == 1, "Images must be single-channel grayscale"
    assert images.shape[2:] == (3506, 4640), "Images must preserve original dimensions"
    assert not np.isnan(images).any(), "Images contain NaNs"
    assert len(intersection) == 0, "Data leakage between splits"
    
    print("\n5. One Batch Example Content:")
    for i in range(len(images)):
        print(f"  Sample [{i}]:")
        print(f"    Path: {batch['image_path'][i]}")
        print(f"    Sample: {batch['sample'][i]}")
        print(f"    Z-value: {z_vals[i]}")
        print(f"    Wavelength: {w_vals[i]}")
        print(f"    L-value: {l_vals[i]}")
        
    print("\n--- ALL TESTS PASSED ---")

if __name__ == '__main__':
    main()
