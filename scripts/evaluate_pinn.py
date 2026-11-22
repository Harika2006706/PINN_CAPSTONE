import os
import sys
import torch
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dataset import get_dataloaders
from src.models.pinn import DLHMPINN

def main(checkpoint_path='results/pinn_training/best_pinn.pth'):
    print("--- STARTING PINN EVALUATION ---")
    manifest_path = 'data/development_manifest.csv'
    
    # Save directly to the pinn_training folder as requested
    save_dir = 'results/pinn_training/'
    os.makedirs(save_dir, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print("Loading datasets...")
    # Exact same 66-image validation split logic used during baseline and PINN training
    _, val_loader, _, _ = get_dataloaders(
        manifest_path=manifest_path,
        batch_size=8,
        val_split=0.2,
        random_state=42
    )
    
    print("Initializing model...")
    model = DLHMPINN(use_fft_channel=True, input_size=(256, 256), base_channels=16)
    
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return
            
    print(f"Loading checkpoint from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    
    all_preds = []
    all_targets = []
    all_paths = []
    all_samples = []
    
    print("Running validation inference...")
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            targets = batch['z_value'].to(device)
            
            preds = model(images)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_paths.extend(batch['image_path'])
            all_samples.extend(batch['sample'])
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    abs_errors = np.abs(all_preds - all_targets)
    
    mae = mean_absolute_error(all_targets, all_preds)
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    r2 = r2_score(all_targets, all_preds)
    
    print(f"\nPINN Validation MAE: {mae:.4f}")
    print(f"PINN Validation RMSE: {rmse:.4f}")
    print(f"PINN Validation R2: {r2:.4f}")
    
    # 1. Save evaluation metrics
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'val_samples': len(all_targets)
    }
    with open(os.path.join(save_dir, 'evaluation_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # 2. Save predictions CSV
    df_preds = pd.DataFrame({
        'image_path': all_paths,
        'sample': all_samples,
        'actual_z': all_targets,
        'predicted_z': all_preds,
        'absolute_error': abs_errors
    })
    df_preds.to_csv(os.path.join(save_dir, 'predictions.csv'), index=False)
    
    # 3. Plot and save predicted vs actual scatter plot
    plt.figure(figsize=(8, 8))
    plt.scatter(all_targets, all_preds, alpha=0.6, edgecolors='k', label='PINN Predictions')
    
    min_val = min(np.min(all_targets), np.min(all_preds))
    max_val = max(np.max(all_targets), np.max(all_preds))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal (y=x)')
    
    plt.title('Physics-Informed CNN: Actual vs Predicted z-value')
    plt.xlabel('Actual z (mm)')
    plt.ylabel('Predicted z (mm)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plot_path = os.path.join(save_dir, 'predicted_vs_actual.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved artifacts to {save_dir}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='results/pinn_training/best_pinn.pth')
    args = parser.parse_args()
    main(checkpoint_path=args.checkpoint)
