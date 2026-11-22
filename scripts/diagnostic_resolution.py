import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dataset import get_dataloaders
from src.physics.asm import propagate_dlhm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def main():
    save_dir = 'results/diagnostic'
    os.makedirs(save_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    _, val_loader, _, _ = get_dataloaders(
        manifest_path='data/development_manifest.csv',
        batch_size=1,
        val_split=0.2,
        random_state=42
    )
    
    # Grab first valid batch
    batch = next(iter(val_loader))
    img_full = batch['image'].to(device) # Original 3506 x 4640
    
    gt_z = batch['z_value'][0].item()
    wave = batch['wavelength'][0].item()
    L = batch['L_value'][0].item()
    sample_name = batch['sample'][0]
    
    print(f"--- DIAGNOSTIC: {sample_name} ---")
    print(f"Ground Truth z: {gt_z:.4f} mm")
    print(f"Wavelength: {wave:.6f} mm")
    print(f"L: {L:.4f} mm")
    
    z_min = max(1.1, gt_z - 0.5)
    z_max = min(3.0, gt_z + 0.5)
    z_vals = np.linspace(z_min, z_max, 50)
    
    # --- Condition A: Original Resolution ---
    print("\n[Condition A] Original Resolution (3506x4640)")
    dx_orig = 0.0038
    print(f"Pixel pitch: {dx_orig} mm")
    metrics_A = []
    
    with torch.no_grad():
        for z in z_vals:
            z_t = torch.tensor([z], device=device, dtype=torch.float32)
            # Memory might be tight for 4640x3506 FFT, we'll try it
            try:
                u_samp, _, _ = propagate_dlhm(img_full, dx_orig, wave, z_t, L, backward=True)
                metrics_A.append(torch.var(torch.abs(u_samp)).item())
            except RuntimeError as e:
                print(f"OOM on Condition A: {e}")
                metrics_A.append(0)
    
    if np.any(metrics_A):
        min_z_A = z_vals[np.argmin(metrics_A)]
        print(f"Var Minimum z: {min_z_A:.4f} mm | Error: {abs(min_z_A - gt_z):.4f} mm")
        
    # --- Condition B: 256x256 Resolution ---
    print("\n[Condition B] Resized Resolution (256x256)")
    img_256 = F.interpolate(img_full, size=(256, 256), mode='bilinear', align_corners=False)
    dx_scaled = dx_orig * (4640.0 / 256.0)
    print(f"Scaled pixel pitch used: {dx_scaled:.6f} mm")
    metrics_B = []
    
    with torch.no_grad():
        for z in z_vals:
            z_t = torch.tensor([z], device=device, dtype=torch.float32)
            u_samp, _, _ = propagate_dlhm(img_256, dx_scaled, wave, z_t, L, backward=True)
            metrics_B.append(torch.var(torch.abs(u_samp)).item())
            
    min_z_B = z_vals[np.argmin(metrics_B)]
    print(f"Var Minimum z: {min_z_B:.4f} mm | Error: {abs(min_z_B - gt_z):.4f} mm")

    # --- Condition C: Scaling Inspection ---
    print("\n[Condition C] Scaling Inspection")
    print(f"Original image shape: {img_full.shape}")
    print(f"Resized image shape: {img_256.shape}")
    print(f"Scale X: 4640 / 256 = {4640.0/256.0:.4f}")
    print(f"Scale Y: 3506 / 256 = {3506.0/256.0:.4f}")
    
    if np.any(metrics_A):
        plt.figure(figsize=(10, 5))
        plt.plot(z_vals, metrics_A, 'g-', label='A: 4640x3506')
        plt.plot(z_vals, metrics_B, 'b-', label='B: 256x256')
        plt.axvline(gt_z, color='k', linestyle='--', label='Ground Truth')
        plt.title('Variance Metric Comparison')
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'resolution_diagnostic.png'))

if __name__ == '__main__':
    main()
