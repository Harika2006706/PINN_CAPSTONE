import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

from src.physics.asm import propagate_dlhm

class PINNTrainer:
    def __init__(self, model, train_loader, val_loader, device='cpu', lr=1e-3, save_dir='checkpoints/', lambda_phys=0.1):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir
        self.lambda_phys = lambda_phys
        
        self.mse_loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        
        os.makedirs(self.save_dir, exist_ok=True)
        
    def physics_loss_fn(self, holograms_256, z_preds, wavelengths, L_values, dx_base):
        """
        Physics-Informed Loss Formulation:
        
        - Source-supported: FocusNET uses variance/focus metrics to identify the reconstruction distance (Pages 1 & 7).
        # - Our derived modelling choice: We use the Dubois metric (sum of amplitude module) evaluated at z_pred as a physics-informed loss.
        #   By minimizing Sum(|ASM_bwd(I, -z_pred)|), we encourage the network's prediction to land on the focal plane.
        # - Modelling assumption (ASM): Equivalent Plane-Wave ASM is explicitly used to connect the DLHM spherical-wave 
        #   geometry with the ASM requirement mandated by pptEE74.pptx.
        """
        # The crop preserves the Y-axis (3506), so scaling ratio is 3506/256
        dx_scaled = dx_base * (3506.0 / 256.0)
        
        B = holograms_256.shape[0]
        phys_loss = 0.0
        
        for i in range(B):
            holo = holograms_256[i:i+1] # (1, 1, 256, 256)
            z = z_preds[i] 
            wave = wavelengths[i].item() if isinstance(wavelengths[i], torch.Tensor) else wavelengths[i]
            L = L_values[i].item() if isinstance(L_values[i], torch.Tensor) else L_values[i]
            dx = dx_scaled[i].item() if isinstance(dx_scaled, torch.Tensor) else dx_scaled
            
            # 1. Back-propagate to sample plane at predicted z
            u_sample, _, _ = propagate_dlhm(holo, dx, wave, z, L, backward=True)
            
            # 2. Extract amplitude module
            amp_module = torch.abs(u_sample)
            
            # 3. Source-supported physics constraint (Dubois Metric)
            # We minimize the Dubois metric (mean of amplitude) as stated in FocusNET.pdf Fig 12 / Page 7
            metric_val = torch.mean(amp_module)
            
            phys_loss = phys_loss + metric_val
            
        return phys_loss / B
        
    def train_epoch(self):
        self.model.train()
        total_data_loss = 0.0
        total_phys_loss = 0.0
        
        for batch in self.train_loader:
            images = batch['image'].to(self.device)
            targets = batch['z_value'].to(self.device)
            
            wavelengths = batch['wavelength']
            L_values = batch['L_value']
            dx_base = 0.0038
            
            self.optimizer.zero_grad()
            z_preds = self.model(images)
            
            loss_data = self.mse_loss(z_preds, targets)
            
            # --- PHYSICS PATHWAY (Differentiable ASM) ---
            # 1. 'images' is still (B, 1, 3506, 4640). Center-crop the width to 3506 to preserve the physical aspect ratio.
            # Width difference: 4640 - 3506 = 1134. Crop 567 pixels from each side (567:4073).
            images_cropped = images[:, :, :, 567:4073] 
            
            # 2. Resize the true square down to 256x256 so the ASM runs fast without astigmatism.
            images_phys_256 = F.interpolate(images_cropped, size=(256, 256), mode='bilinear', align_corners=False)
            
            loss_phys = self.physics_loss_fn(images_phys_256, z_preds, wavelengths, L_values, dx_base)
            
            loss = loss_data + self.lambda_phys * loss_phys
            loss.backward()
            self.optimizer.step()
            
            total_data_loss += loss_data.item() * images.size(0)
            total_phys_loss += loss_phys.item() * images.size(0)
            
        N = len(self.train_loader.dataset)
        return total_data_loss / N, total_phys_loss / N
        
    def validate(self):
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device)
                targets = batch['z_value'].to(self.device)
                preds = self.model(images)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        mae = mean_absolute_error(all_targets, all_preds)
        rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
        r2 = r2_score(all_targets, all_preds)
        
        return {'mae': mae, 'rmse': rmse, 'r2': r2}
        
    def train(self, num_epochs=50):
        best_mae = float('inf')
        metrics_history = []
        
        for epoch in range(num_epochs):
            train_data_loss, train_phys_loss = self.train_epoch()
            val_metrics = self.validate()
            self.scheduler.step()
            
            print(f"Epoch {epoch+1}/{num_epochs} - Data MSE: {train_data_loss:.4f} - Phys MSE: {train_phys_loss:.4f} - Val MAE: {val_metrics['mae']:.4f} - Val RMSE: {val_metrics['rmse']:.4f} - Val R2: {val_metrics['r2']:.4f}")
            
            metrics_history.append({
                'epoch': epoch + 1,
                'train_data_loss': train_data_loss,
                'train_phys_loss': train_phys_loss,
                'val_mae': val_metrics['mae'],
                'val_rmse': val_metrics['rmse'],
                'val_r2': val_metrics['r2']
            })
            
            if val_metrics['mae'] < best_mae:
                best_mae = val_metrics['mae']
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'best_pinn_dubois.pth'))
                
        import json
        with open(os.path.join(self.save_dir, 'metrics_history_dubois.json'), 'w') as f:
            json.dump(metrics_history, f, indent=4)
            
        return best_mae
