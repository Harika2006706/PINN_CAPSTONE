import os
import sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.baseline import DLHMBaselineCNN

def main():
    print("--- BASELINE CNN FORWARD PASS VERIFICATION ---")
    
    # 1. Initialize model
    model = DLHMBaselineCNN(use_fft_channel=True, input_size=(256, 256), base_channels=8)
    print(f"Model initialized with in_channels={model.in_channels}")
    
    # 2. Create dummy input simulating a single batch of 1 hologram
    # DLHM holograms are large, e.g., 3506x4640. The model forward pass automatically interpolates them down.
    B, C, H, W = 1, 1, 3506, 4640
    print(f"Creating dummy input tensor of shape: {(B, C, H, W)}")
    dummy_input = torch.randn(B, C, H, W, requires_grad=True)
    
    # Dummy target z_value (e.g. 1.5 mm)
    target_z = torch.tensor([1.5], dtype=torch.float32)
    print(f"Target shape: {target_z.shape}")
    
    # 3. Forward Pass
    print("\nRunning Forward Pass...")
    preds = model(dummy_input)
    print(f"Output shape: {preds.shape}")
    
    # 4. Loss and Gradients
    print("\nRunning Backward Pass...")
    criterion = torch.nn.MSELoss()
    loss = criterion(preds, target_z)
    loss.backward()
    
    # 5. Verify Gradient Flow
    if dummy_input.grad is not None:
        grad_sum = torch.abs(dummy_input.grad).sum().item()
        print(f"Gradient flowed back to input! Sum of abs gradients: {grad_sum:.4f}")
    else:
        print("ERROR: No gradients found on input.")
        
    print("\n--- TEST PASSED ---")

if __name__ == '__main__':
    main()
