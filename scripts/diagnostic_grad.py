import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dataset import get_dataloaders
from src.models.pinn import DLHMPINN
from src.physics.asm import propagate_dlhm

def physics_loss_fn(holograms_256, z_preds, wavelengths, L_values, dx_base):
    dx_scaled = dx_base * (3506.0 / 256.0)
    B = holograms_256.shape[0]
    phys_loss = 0.0
    for i in range(B):
        holo = holograms_256[i:i+1]
        z = z_preds[i] 
        wave = wavelengths[i].item() if isinstance(wavelengths[i], torch.Tensor) else wavelengths[i]
        L = L_values[i].item() if isinstance(L_values[i], torch.Tensor) else L_values[i]
        dx = dx_scaled[i].item() if isinstance(dx_scaled, torch.Tensor) else dx_scaled
        
        u_sample, _, _ = propagate_dlhm(holo, dx, wave, z, L, backward=True)
        amp_module = torch.abs(u_sample)
        metric_val = torch.mean(amp_module)
        phys_loss = phys_loss + metric_val
    return phys_loss / B

def run_diagnostic():
    device = 'cpu' # Use CPU for quick diagnostic
    print("Loading data...")
    train_loader, _, _, _ = get_dataloaders('data/development_manifest.csv', batch_size=2)
    batch = next(iter(train_loader))
    
    model = DLHMPINN(use_fft_channel=True, input_size=(256, 256), base_channels=16).to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    mse_loss = nn.MSELoss()
    lambda_phys = 0.1
    
    images = batch['image'].to(device)
    targets = batch['z_value'].to(device)
    wavelengths = batch['wavelength']
    L_values = batch['L_value']
    dx_base = 0.0038
    
    print("\n--- FORWARD PASS ---")
    optimizer.zero_grad()
    
    # Store initial params
    initial_params = [p.clone() for p in model.parameters()]
    
    z_preds = model(images)
    z_preds.retain_grad()
    
    print(f"z_pred before step: {z_preds.detach().numpy()}")
    print(f"z_pred requires_grad: {z_preds.requires_grad}")
    print(f"z_pred grad_fn: {z_preds.grad_fn}")
    
    loss_data = mse_loss(z_preds, targets)
    print(f"L_data: {loss_data.item():.6f}")
    
    images_cropped = images[:, :, :, 567:4073] 
    images_phys_256 = F.interpolate(images_cropped, size=(256, 256), mode='bilinear', align_corners=False)
    
    loss_phys = physics_loss_fn(images_phys_256, z_preds, wavelengths, L_values, dx_base)
    print(f"L_physics: {loss_phys.item():.6f}")
    
    loss = loss_data + lambda_phys * loss_phys
    print(f"Total Loss: {loss.item():.6f}")
    
    print("\n--- BACKWARD PASS ---")
    loss.backward()
    
    if z_preds.grad is not None:
        print(f"z_pred gradient from loss: {z_preds.grad.numpy()}")
    else:
        print("WARNING: z_pred received NO gradients!")
        
    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.norm().item() ** 2
    grad_norm = grad_norm ** 0.5
    print(f"Gradient norm of CNN backbone: {grad_norm:.6f}")
    
    print("\n--- OPTIMIZER STEP ---")
    optimizer.step()
    
    changed = False
    for p_init, p_new in zip(initial_params, model.parameters()):
        if not torch.equal(p_init, p_new):
            changed = True
            break
    print(f"Did model parameters change? {changed}")
    
    with torch.no_grad():
        z_preds_after = model(images)
    print(f"z_pred after step:  {z_preds_after.detach().numpy()}")

if __name__ == '__main__':
    run_diagnostic()
