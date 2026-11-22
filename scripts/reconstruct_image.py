import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.pinn import DLHMPINN
from src.physics.asm import propagate_dlhm

def main():
    parser = argparse.ArgumentParser(description="Reconstruct a single DLHM hologram using PINN.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input BMP hologram.")
    parser.add_argument("--weights", type=str, default="results/pinn_training/best_pinn.pth", help="Path to model weights.")
    parser.add_argument("--wavelength", type=float, default=0.0006328, help="Wavelength in mm (default: He-Ne laser 632.8 nm).")
    parser.add_argument("--L", type=float, default=14.0, help="Source-to-camera distance L in mm (default: 14.0 mm).")
    parser.add_argument("--dx", type=float, default=0.0038, help="Camera pixel pitch in mm (default: 0.0038 mm).")
    parser.add_argument("--out", type=str, default="reconstruction.png", help="Output image path.")
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 1. Load the trained PINN
    print(f"Loading model weights from {args.weights}...")
    model = DLHMPINN(use_fft_channel=True, input_size=(256, 256), base_channels=16).to(device)
    
    try:
        model.load_state_dict(torch.load(args.weights, map_location=device))
    except FileNotFoundError:
        print(f"Warning: {args.weights} not found. Running with untrained weights for demonstration.")
        
    model.eval()
    
    # 2. Load and preprocess the image
    print(f"Loading image {args.image}...")
    img = Image.open(args.image).convert('L')
    img_np = np.array(img, dtype=np.float32) / 255.0
    
    # Shape: (1, 1, H, W)
    img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device)
    H, W = img_tensor.shape[2], img_tensor.shape[3]
    print(f"Original image shape: {H}x{W}")
    
    # 3. Format for physics reconstruction (Crop)
    # The dataset uses a center crop to preserve the 1:1 aspect ratio.
    # Assuming W=4640, H=3506, we crop width from 567 to 4073
    crop_w_start = (W - H) // 2
    crop_w_end = crop_w_start + H
    
    img_cropped = img_tensor[:, :, :, crop_w_start:crop_w_end]
    print(f"Cropped shape for physics consistency: {img_cropped.shape[2]}x{img_cropped.shape[3]}")
    
    # 4. Predict z-distance using the FULL raw image to match trainer behavior
    with torch.no_grad():
        z_pred = model(img_tensor)[0].item()
    print(f"Predicted Reconstruction Distance (z): {z_pred:.4f} mm")
    
    # 5. Physics Reconstruction (Full Resolution)
    # We use the high-res cropped image for reconstruction so the output is sharp!
    print("Running Angular Spectrum Method (ASM) reconstruction on full resolution...")
    
    z_tensor = torch.tensor([z_pred], device=device)
    wave_tensor = torch.tensor([args.wavelength], device=device)
    L_tensor = torch.tensor([args.L], device=device)
    
    # Backward=True to backpropagate the wave to the focal plane
    u_out, _, _ = propagate_dlhm(img_cropped, args.dx, wave_tensor, z_tensor, L_tensor, backward=True)
    
    # Extract amplitude
    amplitude = torch.abs(u_out)[0, 0].cpu().numpy()
    
    # 6. Normalize and save
    print(f"Saving reconstructed image to {args.out}...")
    
    # Basic contrast stretching to make biological details visible (1st to 99th percentile)
    p1, p99 = np.percentile(amplitude, (1, 99))
    amp_normalized = np.clip((amplitude - p1) / (p99 - p1), 0, 1)
    
    plt.imsave(args.out, amp_normalized, cmap='gray')
    print("Done!")

if __name__ == '__main__':
    main()
