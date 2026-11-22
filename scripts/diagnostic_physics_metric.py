import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dataset import get_dataloaders
from src.physics.asm import propagate_dlhm

def main():
    save_dir = 'results/diagnostic'
    os.makedirs(save_dir, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    train_loader, val_loader, _, _ = get_dataloaders(
        manifest_path='data/development_manifest.csv',
        batch_size=1,
        val_split=0.2,
        random_state=42
    )
    
    # Get 5 unique samples
    selected_samples = []
    seen_sample_names = set()
    
    for batch in train_loader:
        selected_samples.append(batch)
        if len(selected_samples) == 5:
            break
            
    dx_base = 0.0038
    dx_scaled = dx_base * (3506.0 / 256.0)
    
    results = []
    
    for idx, batch in enumerate(selected_samples):
        img_full = batch['image'].to(device)
        # Center-crop the width to 3506 to preserve the physical aspect ratio
        img_cropped = img_full[:, :, :, 567:4073]
        holo = F.interpolate(img_cropped, size=(256, 256), mode='bilinear', align_corners=False)
        
        gt_z = batch['z_value'][0].item()
        wave = batch['wavelength'][0].item()
        L = batch['L_value'][0].item()
        sample_name = batch['sample'][0]
        
        # Sweep range: +/- 0.5 mm from GT, or [1.1, 3.0] whichever is tighter
        z_min_sweep = max(1.1, gt_z - 0.5)
        z_max_sweep = min(3.0, gt_z + 0.5)
        z_vals = np.linspace(z_min_sweep, z_max_sweep, 50)
        
        var_metrics = []
        dubois_metrics = []
        
        with torch.no_grad():
            for z in z_vals:
                z_tensor = torch.tensor([z], device=device, dtype=torch.float32)
                u_sample, _, _ = propagate_dlhm(holo, dx_scaled, wave, z_tensor, L, backward=True)
                amp_module = torch.abs(u_sample)
                var_metrics.append(torch.var(amp_module).item())
                dubois_metrics.append(torch.mean(amp_module).item())
                
        var_metrics = np.array(var_metrics)
        dubois_metrics = np.array(dubois_metrics)
        
        var_min_z = z_vals[np.argmin(var_metrics)]
        var_max_z = z_vals[np.argmax(var_metrics)]
        dubois_min_z = z_vals[np.argmin(dubois_metrics)]
        dubois_max_z = z_vals[np.argmax(dubois_metrics)]
        
        results.append({
            'sample': sample_name,
            'gt_z': gt_z,
            'var_min_z': var_min_z,
            'var_max_z': var_max_z,
            'dubois_min_z': dubois_min_z,
            'dubois_max_z': dubois_max_z,
        })
        
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(z_vals, var_metrics, 'b-', label='Variance')
        plt.axvline(gt_z, color='g', linestyle='--', label='GT')
        plt.title('Variance Metric')
        
        plt.subplot(1, 2, 2)
        plt.plot(z_vals, dubois_metrics, 'r-', label='Dubois (Sum Amp)')
        plt.axvline(gt_z, color='g', linestyle='--', label='GT')
        plt.title('Dubois Metric')
        
        plot_path = os.path.join(save_dir, f"diagnostic_{idx}.png")
        plt.savefig(plot_path)
        plt.close()
        
    print("--- DIAGNOSTIC RESULTS ---")
    for res in results:
        print(f"Sample: {res['sample']} | GT: {res['gt_z']:.3f} | VarMin: {res['var_min_z']:.3f} | VarMax: {res['var_max_z']:.3f} | DubMin: {res['dubois_min_z']:.3f} | DubMax: {res['dubois_max_z']:.3f}")
        
if __name__ == '__main__':
    main()
