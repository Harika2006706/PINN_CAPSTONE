import os
import sys
import torch
import torch.nn as nn
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.training.pinn_trainer import PINNTrainer

class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.tensor(1.0))
    def forward(self, x):
        return torch.tensor([1.5, 1.5], requires_grad=True)

class MockDataset:
    def __len__(self):
        return 2

class MockLoader:
    def __init__(self):
        self.dataset = MockDataset()
        
    def __iter__(self):
        yield {
            'image': torch.rand(2, 1, 3506, 4640),
            'z_value': torch.tensor([1.5, 1.5]),
            'wavelength': torch.tensor([0.000633, 0.000633]),
            'L_value': torch.tensor([18.8, 18.8])
        }

def main():
    print("Testing Syntax and Import...")
    model = MockModel()
    train_loader = MockLoader()
    val_loader = MockLoader()
    
    trainer = PINNTrainer(model, train_loader, val_loader, lambda_phys=0.1)
    
    print("Testing Minimal Forward Pass & Cropping...")
    try:
        data_loss, phys_loss = trainer.train_epoch()
        print("-> PASS: Forward/Backward completed successfully.")
        print(f"Data Loss: {data_loss}")
        print(f"Phys Loss: {phys_loss}")
        print("-> PASS: Loss is finite.")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == '__main__':
    main()
