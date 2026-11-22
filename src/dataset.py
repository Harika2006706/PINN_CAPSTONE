import os
import pandas as pd
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = False
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit

class ODPDataset(Dataset):
    """
    Dataset for Optical Diffraction Tomography / Holography.
    Loads uncompressed 8-bit grayscale BMPs and associated physical parameters.
    """
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row['image_path'].replace('\\', '/')
        
        # Load image as 8-bit grayscale
        img = Image.open(image_path).convert('L')
        
        # Ensure all images have the exact standard dimensions to prevent batch collate crashes
        if img.size != (4640, 3506):
            img = img.resize((4640, 3506), Image.Resampling.BILINEAR)
            
        # Convert to numpy array and normalize to [0, 1]
        img_np = np.array(img, dtype=np.float32) / 255.0
        
        # Convert to tensor (C, H, W)
        img_tensor = torch.from_numpy(img_np).unsqueeze(0)
        
        if self.transform:
            img_tensor = self.transform(img_tensor)
            
        return {
            'image': img_tensor,
            'z_value': torch.tensor(row['z_value'], dtype=torch.float32),
            'wavelength': torch.tensor(row['wavelength'], dtype=torch.float32),
            'L_value': torch.tensor(row['L_value'], dtype=torch.float32),
            'sample': row['sample'],
            'image_path': image_path
        }

def get_dataloaders(manifest_path, batch_size=4, val_split=0.2, random_state=42):
    """
    Creates train and validation dataloaders ensuring no data leakage between
    identical or near-identical acquisitions (i.e. same z_value FOV).
    """
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        
    df = pd.read_csv(manifest_path)
    
    print("Validating images for corruption/truncation... (This may take a minute)")
    valid_indices = []
    for i, row in df.iterrows():
        path = row['image_path'].replace('\\', '/')
        if not os.path.exists(path):
            continue
        try:
            with Image.open(path) as img:
                img.load()  # Forces PIL to actually read the pixel data
            valid_indices.append(i)
        except Exception:
            continue
            
    df = df.loc[valid_indices].reset_index(drop=True)
    print(f"Final valid training images: {len(df)}")
    
    # We must split by z_value to prevent leakage of identical acquisitions
    # (e.g. time-lapse sequences of the same FOV/z_value)
    groups = df['z_value']
    
    # GroupShuffleSplit ensures that all rows with the same z_value go to the same split
    gss = GroupShuffleSplit(n_splits=1, test_size=val_split, random_state=random_state)
    train_idx, val_idx = next(gss.split(df, groups=groups))
    
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    
    train_dataset = ODPDataset(train_df)
    val_dataset = ODPDataset(val_df)
    
    # Dataloaders (num_workers=0 for tests on windows to avoid multiprocessing issues)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)
    
    return train_loader, val_loader, train_df, val_df
