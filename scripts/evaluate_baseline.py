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
from src.models.baseline import DLHMBaselineCNN

def main(checkpoint_path='results/best_baseline.pth'):
    print("--- STARTING BASELINE EVALUATION ---")
    manifest_path = 'data/development_manifest.csv'
    save_dir = 'results/baseline_evaluation/'
    
    os.makedirs(save_dir, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print("Loading datasets...")
    # Exact same logic as in training: Random state 42, group shuffle split
    _, val_loader, _, _ = get_dataloaders(
        manifest_path=manifest_path,
        batch_size=8,
        val_split=0.2,
        random_state=42
    )
    
    print("Initializing model...")
    # Exact same architecture and base_channels as used in train_baseline.py
    model = DLHMBaselineCNN(use_fft_channel=True, input_size=(256, 256), base_channels=16)
    
    if not os.path.exists(checkpoint_path):
        # Fallback to the nested directory if the user ran the training script as-is
        fallback = 'results/baseline_training/best_baseline.pth'
        if os.path.exists(fallback):
            checkpoint_path = fallback
        else:
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
    
    print(f"\nValidation MAE: {mae:.4f}")
    print(f"Validation RMSE: {rmse:.4f}")
    print(f"Validation R2: {r2:.4f}")
    
    # Save metrics
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'val_samples': len(all_targets)
    }
    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # Save predictions.csv
    df_preds = pd.DataFrame({
        'image_path': all_paths,
        'sample': all_samples,
        'actual_z': all_targets,
        'predicted_z': all_preds,
        'absolute_error': abs_errors
    })
    df_preds.to_csv(os.path.join(save_dir, 'predictions.csv'), index=False)
    
    # Plot predicted vs actual
    plt.figure(figsize=(8, 8))
    plt.scatter(all_targets, all_preds, alpha=0.6, edgecolors='k', label='Predictions')
    
    # y = x reference line
    min_val = min(np.min(all_targets), np.min(all_preds))
    max_val = max(np.max(all_targets), np.max(all_preds))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal (y=x)')
    
    plt.title('Baseline CNN: Actual vs Predicted z-value')
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
    parser.add_argument('--checkpoint', type=str, default='results/best_baseline.pth')
    args = parser.parse_args()
    main(checkpoint_path=args.checkpoint)
