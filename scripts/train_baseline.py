import os
import sys
import torch
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dataset import get_dataloaders
from src.models.baseline import DLHMBaselineCNN
from src.training.trainer import BaselineTrainer

def main():
    print("--- STARTING BASELINE DLHM TRAINING ---")
    manifest_path = 'data/development_manifest.csv'
    save_dir = 'results/baseline_training/'
    
    os.makedirs(save_dir, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print("Loading datasets...")
    train_loader, val_loader, _, _ = get_dataloaders(
        manifest_path=manifest_path,
        batch_size=8,
        val_split=0.2
    )
    
    print("Initializing model...")
    # Using 16 base channels for stronger feature extraction
    model = DLHMBaselineCNN(use_fft_channel=True, input_size=(256, 256), base_channels=16)
    
    print("Setting up trainer...")
    trainer = BaselineTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=1e-3,
        save_dir=save_dir
    )
    
    print("Starting training...")
    epochs = 50
    metrics_history = []
    
    best_mae = float('inf')
    
    for epoch in range(epochs):
        train_loss = trainer.train_epoch()
        val_metrics = trainer.validate()
        trainer.scheduler.step()
        
        log_str = f"Epoch {epoch+1}/{epochs} - Train MSE: {train_loss:.4f} - Val MAE: {val_metrics['mae']:.4f} - Val RMSE: {val_metrics['rmse']:.4f} - Val R2: {val_metrics['r2']:.4f}"
        print(log_str)
        
        metrics_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_mae': val_metrics['mae'],
            'val_rmse': val_metrics['rmse'],
            'val_r2': val_metrics['r2']
        })
        
        if val_metrics['mae'] < best_mae:
            best_mae = val_metrics['mae']
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_baseline.pth'))
            print("  --> Saved best model")
            
    # Save training history
    with open(os.path.join(save_dir, 'metrics_history.json'), 'w') as f:
        json.dump(metrics_history, f, indent=4)
        
    print(f"Training complete! Best validation MAE: {best_mae:.4f}")

if __name__ == '__main__':
    main()
