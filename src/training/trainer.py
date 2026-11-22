import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

class BaselineTrainer:
    def __init__(self, model, train_loader, val_loader, device='cpu', lr=1e-3, save_dir='checkpoints/'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        
        os.makedirs(self.save_dir, exist_ok=True)
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        
        for batch in self.train_loader:
            images = batch['image'].to(self.device)
            targets = batch['z_value'].to(self.device)
            
            self.optimizer.zero_grad()
            preds = self.model(images)
            loss = self.criterion(preds, targets)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * images.size(0)
            
        return total_loss / len(self.train_loader.dataset)
        
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
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
        
    def train(self, num_epochs=50):
        best_mae = float('inf')
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_metrics = self.validate()
            self.scheduler.step()
            
            print(f"Epoch {epoch+1}/{num_epochs} - Train MSE: {train_loss:.4f} - Val MAE: {val_metrics['mae']:.4f} - Val RMSE: {val_metrics['rmse']:.4f} - Val R2: {val_metrics['r2']:.4f}")
            
            if val_metrics['mae'] < best_mae:
                best_mae = val_metrics['mae']
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'best_baseline.pth'))
                print("  --> Saved best model")
