import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.pinn import DLHMPINN
from src.dataset import get_dataloaders

def main():
    device = 'cpu'
    model = DLHMPINN(use_fft_channel=True, input_size=(256, 256), base_channels=16).to(device)
    model.load_state_dict(torch.load('results/pinn_training/best_pinn_dubois.pth', map_location=device))
    model.eval()

    # 1. Dataset preprocessing
    train_loader, val_loader, _, _ = get_dataloaders(
        manifest_path='data/development_manifest.csv',
        batch_size=1,
        val_split=0.2,
        random_state=42
    )
    batch = next(iter(val_loader))
    dataset_img = batch['image']
    dataset_z = batch['z_value']
    
    print(f"Dataset image shape: {dataset_img.shape}")
    
    # Run through model directly as done in trainer
    with torch.no_grad():
        raw_z_dataset = model.backbone(dataset_img)
        pred_z_dataset = model(dataset_img)
    print(f"Validation Dataloader -> Raw CNN Out: {raw_z_dataset.item():.4f} | Final Z Pred: {pred_z_dataset.item():.4f} | Target: {dataset_z.item():.4f}")

    # 2. Reconstruct_image.py preprocessing
    img_path = batch['image_path'][0]
    img = Image.open(img_path).convert('L')
    img_np = np.array(img, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)
    
    W = img_tensor.shape[3]
    H = img_tensor.shape[2]
    crop_w_start = (W - H) // 2
    crop_w_end = crop_w_start + H
    
    img_cropped = img_tensor[:, :, :, crop_w_start:crop_w_end]
    img_256 = F.interpolate(img_cropped, size=(256, 256), mode='bilinear', align_corners=False)
    
    print(f"\nReconstruct script image shape passed to model: {img_256.shape}")
    with torch.no_grad():
        raw_z_recon = model.backbone(img_256)
        pred_z_recon = model(img_256)
    print(f"Reconstruct Script -> Raw CNN Out: {raw_z_recon.item():.4f} | Final Z Pred: {pred_z_recon.item():.4f}")

if __name__ == '__main__':
    main()
