import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFile
import matplotlib.pyplot as plt
import glob

ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.pinn import DLHMPINN
from src.physics.asm import propagate_dlhm

def create_contact_sheet(images, labels, cols=5):
    if not images:
        return None
    
    rows = (len(images) + cols - 1) // cols
    w, h = images[0].size
    
    # Increase height to add text label space
    label_h = 60
    
    sheet_w = cols * w
    sheet_h = rows * (h + label_h)
    
    sheet = Image.new('RGB', (sheet_w, sheet_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    
    for i, (img, label) in enumerate(zip(images, labels)):
        col = i % cols
        row = i // cols
        
        x = col * w
        y = row * (h + label_h)
        
        sheet.paste(img, (x, y))
        
        # Draw label below the image
        text_x = x + 10
        text_y = y + h + 10
        draw.text((text_x, text_y), label, fill=(0, 0, 0))
        
    return sheet

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = DLHMPINN(use_fft_channel=True, input_size=(256, 256), base_channels=16).to(device)
    weights_path = 'results/pinn_training/best_pinn_dubois.pth'
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        print(f"Warning: weights not found at {weights_path}")
    model.eval()

    # Find 20 valid BMP files
    all_bmps = glob.glob('data/SW-Dataset/**/*.bmp', recursive=True)
    if len(all_bmps) > 20:
        all_bmps = all_bmps[:20]
        
    print(f"Found {len(all_bmps)} BMP images to process.")
    
    success_count = 0
    fail_count = 0
    
    reconstructed_images = []
    labels = []
    
    out_dir = 'results/reconstructions'
    os.makedirs(out_dir, exist_ok=True)
    
    # We'll downscale them for the contact sheet to save memory
    sheet_img_size = (512, 512) 

    for idx, img_path in enumerate(all_bmps):
        try:
            filename = os.path.basename(img_path)
            sample_name = os.path.basename(os.path.dirname(os.path.dirname(img_path)))
            
            # Use default wavelengths from our testing
            wavelength = 0.000633
            L = 18.81 if 'Cheek' in sample_name else 13.92
            dx = 0.0038
            
            img = Image.open(img_path).convert('L')
            img_np = np.array(img, dtype=np.float32) / 255.0
            
            img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device)
            H, W = img_tensor.shape[2], img_tensor.shape[3]
            
            crop_w_start = (W - H) // 2
            crop_w_end = crop_w_start + H
            img_cropped = img_tensor[:, :, :, crop_w_start:crop_w_end]
            
            with torch.no_grad():
                z_pred = model(img_tensor)[0].item()
                
            z_tensor = torch.tensor([z_pred], device=device)
            wave_tensor = torch.tensor([wavelength], device=device)
            L_tensor = torch.tensor([L], device=device)
            
            u_out, _, _ = propagate_dlhm(img_cropped, dx, wave_tensor, z_tensor, L_tensor, backward=True)
            amplitude = torch.abs(u_out)[0, 0].cpu().numpy()
            
            p1, p99 = np.percentile(amplitude, (1, 99))
            amp_normalized = np.clip((amplitude - p1) / (p99 - p1), 0, 1)
            
            out_path = os.path.join(out_dir, f"{sample_name}_{filename}_recon.png")
            plt.imsave(out_path, amp_normalized, cmap='gray')
            
            # Load for contact sheet
            recon_img = Image.open(out_path).resize(sheet_img_size)
            reconstructed_images.append(recon_img)
            
            label = f"{sample_name}/{filename}\nz: {z_pred:.4f} mm"
            labels.append(label)
            
            success_count += 1
            print(f"[{idx+1}/{len(all_bmps)}] Success: {sample_name}/{filename} -> z={z_pred:.4f}")
            
        except Exception as e:
            fail_count += 1
            print(f"[{idx+1}/{len(all_bmps)}] Failed: {img_path} | Error: {e}")
            
    print(f"\nFinished! Successful: {success_count}, Failed: {fail_count}")
    
    if success_count > 0:
        print("Generating contact sheet...")
        sheet = create_contact_sheet(reconstructed_images, labels, cols=5)
        sheet_path = 'results/reconstructions_contact_sheet.png'
        sheet.save(sheet_path)
        print(f"Contact sheet saved to {sheet_path}")

if __name__ == '__main__':
    main()
