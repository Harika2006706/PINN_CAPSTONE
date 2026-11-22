import os
import sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dataset import get_dataloaders
from src.models.pinn import DLHMPINN
from src.training.pinn_trainer import PINNTrainer

def main():
    print("--- STARTING PINN DLHM TRAINING ---")
    manifest_path = 'data/odp_manifest.csv'
    save_dir = 'results/pinn_training/'
    
    os.makedirs(save_dir, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print("Loading datasets...")
    # Use exact same group-aware random state as baseline
    train_loader, val_loader, _, _ = get_dataloaders(
        manifest_path=manifest_path,
        batch_size=8,
        val_split=0.2,
        random_state=42
    )
    
    print("Initializing PINN...")
    model = DLHMPINN(use_fft_channel=True, input_size=(256, 256), base_channels=16)
    
    print("Setting up PINN trainer...")
    trainer = PINNTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=1e-3,
        save_dir=save_dir,
        lambda_phys=0.1 # Configurable physics penalty
    )
    
    print("Starting PINN training...")
    trainer.train(num_epochs=50)
    print("Training complete!")

if __name__ == '__main__':
    main()
