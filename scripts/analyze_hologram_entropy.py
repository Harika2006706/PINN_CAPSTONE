import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
import torch.nn.functional as F

ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.insert(0, r"D:\PINN")
from Hologram_Classification_10Class.physics.asm_module import propagate_dlhm
from scripts.train_10class_pinn import MLP10Class

MANIFEST_PATH = r"D:\PINN\Hologram_Classification_10Class\data\odp_10class_manifest.csv"
SPLIT_PATH = r"D:\PINN\Hologram_Classification_10Class\data\odp_10class_split.csv"
CLASSIFIER_PATH = r"D:\PINN\Hologram_Classification_10Class\classifiers\pinn_mlp_vgg19.pth"
FEATURE_PATH = r"D:\PINN\Hologram_Classification_10Class\features\vgg19_pca_test.pt"

OUT_DIR = r"D:\PINN\Hologram_Classification_10Class\results\plots"
OUT_CSV_PATH = os.path.join(OUT_DIR, "entropy_summary_metrics.csv")
OUT_CLASS_PLOT = os.path.join(OUT_DIR, "entropy_class_comparison.png")
OUT_FOCUS_PLOT = os.path.join(OUT_DIR, "entropy_z_focus_curve.png")

def calculate_shannon_entropy(image_np):
    """
    Computes 8-bit Shannon Spatial Image Entropy:
    H(I) = - sum(p(k) * log2(p(k))) for k in [0, 255]
    """
    # Scale image array to [0, 255] uint8 histogram bins
    if image_np.max() <= 1.0:
        img_uint8 = (image_np * 255.0).astype(np.uint8)
    else:
        img_uint8 = image_np.astype(np.uint8)
        
    counts = np.bincount(img_uint8.ravel(), minlength=256)
    probs = counts / float(counts.sum())
    probs = probs[probs > 0] # Exclude zero probabilities to avoid log2(0)
    entropy = -np.sum(probs * np.log2(probs))
    return entropy

def calculate_prediction_entropy(probs):
    """
    Computes Softmax Classifier Prediction Entropy:
    H(P) = - sum(P_i * log2(P_i + 1e-12))
    """
    probs_clamped = np.clip(probs, 1e-12, 1.0)
    return -np.sum(probs_clamped * np.log2(probs_clamped))

def main():
    print("==================================================")
    print("   Running Hologram & Classification Entropy Analysis")
    print("==================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device}")
    
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # ----------------------------------------------------
    # Part 1: Class-Wise Spatial Entropy Analysis (Raw vs ASM Reconstructed)
    # ----------------------------------------------------
    print("\n1. Calculating Spatial Entropy across 10 Classes (Raw vs ASM Reconstructed)...")
    df_manifest = pd.read_csv(MANIFEST_PATH)
    
    classes = sorted(df_manifest['sample'].unique())
    entropy_records = []
    
    for cls in classes:
        cls_df = df_manifest[df_manifest['sample'] == cls].reset_index(drop=True)
        # Pick 5 representative samples per class
        sample_rows = cls_df.iloc[:5]
        
        raw_entropies = []
        recon_entropies = []
        
        for _, r in sample_rows.iterrows():
            rel_p = r['image_path'].replace('\\', '/')
            full_p = os.path.join(r"D:\PINN", rel_p) if not os.path.isabs(rel_p) else rel_p
            z_gt_cm = float(r['z_value'])
            
            try:
                img_raw = Image.open(full_p).convert('L')
                raw_np = np.array(img_raw, dtype=np.float32) / 255.0
                h_raw = calculate_shannon_entropy(raw_np)
                raw_entropies.append(h_raw)
                
                # ASM Reconstruction at GT z
                t_256 = F.interpolate(torch.from_numpy(raw_np).unsqueeze(0).unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False).to(device)
                wavelength_mm = torch.tensor(0.633 * 1e-3, device=device)
                L_mm = torch.tensor(188.1, device=device)
                z_mm = torch.tensor(z_gt_cm * 10.0, device=device)
                dx_mm = torch.tensor(0.0038, device=device)
                
                with torch.no_grad():
                    u_out, _, _ = propagate_dlhm(t_256, dx_mm, wavelength_mm, z_mm, L_mm, backward=True)
                    amp = torch.abs(u_out).squeeze().cpu().numpy()
                    
                amp_norm = (amp - amp.min()) / (amp.max() - amp.min() + 1e-8)
                h_recon = calculate_shannon_entropy(amp_norm)
                recon_entropies.append(h_recon)
            except Exception:
                pass
                
        entropy_records.append({
            'class_name': cls,
            'mean_raw_entropy': np.mean(raw_entropies),
            'mean_recon_entropy': np.mean(recon_entropies),
            'entropy_reduction': np.mean(raw_entropies) - np.mean(recon_entropies)
        })
        
    df_entropy_class = pd.DataFrame(entropy_records)
    print(df_entropy_class)
    
    # ----------------------------------------------------
    # Part 2: Reconstruction Entropy vs z Focus Distance Curve
    # ----------------------------------------------------
    print("\n2. Computing Minimum Entropy Focus Curve vs Propagation Distance z...")
    sample_img_path = os.path.join(r"D:\PINN", "data/SW-Dataset/1_Cheek_cells/z1/1.bmp")
    img_sample = Image.open(sample_img_path).convert('L')
    sample_np = np.array(img_sample, dtype=np.float32) / 255.0
    t_sample = F.interpolate(torch.from_numpy(sample_np).unsqueeze(0).unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False).to(device)
    
    z_eval_range_mm = np.linspace(5.0, 25.0, 41) # Sweep z from 5 mm to 25 mm
    focus_entropy_list = []
    
    wavelength_mm = torch.tensor(0.633 * 1e-3, device=device)
    L_mm = torch.tensor(188.1, device=device)
    dx_mm = torch.tensor(0.0038, device=device)
    
    with torch.no_grad():
        for z_val in z_eval_range_mm:
            z_tensor = torch.tensor(z_val, device=device)
            u_out, _, _ = propagate_dlhm(t_sample, dx_mm, wavelength_mm, z_tensor, L_mm, backward=True)
            amp = torch.abs(u_out).squeeze().cpu().numpy()
            amp_norm = (amp - amp.min()) / (amp.max() - amp.min() + 1e-8)
            h_val = calculate_shannon_entropy(amp_norm)
            focus_entropy_list.append(h_val)
            
    min_entropy_idx = np.argmin(focus_entropy_list)
    optimal_focus_z = z_eval_range_mm[min_entropy_idx]
    print(f"   --> Minimum Entropy Focal Distance: z = {optimal_focus_z:.2f} mm (Min Entropy = {focus_entropy_list[min_entropy_idx]:.4f} bits)")
    
    # ----------------------------------------------------
    # Part 3: Classifier Prediction Output Uncertainty Entropy
    # ----------------------------------------------------
    print("\n3. Evaluating Test Set Classifier Prediction Entropy...")
    ckpt = torch.load(CLASSIFIER_PATH, map_location=device, weights_only=False)
    pca_test = torch.load(FEATURE_PATH, map_location=device, weights_only=False)
    
    feats = pca_test['features'].to(device)
    labels = pca_test['class_labels']
    
    classifier = MLP10Class(input_dim=feats.shape[1], num_classes=10).to(device)
    classifier.load_state_dict(ckpt['model_state_dict'])
    classifier.eval()
    
    with torch.no_grad():
        logits = classifier(feats)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        
    pred_entropies = [calculate_prediction_entropy(p) for p in probs]
    avg_pred_entropy = np.mean(pred_entropies)
    max_possible_entropy = np.log2(10.0) # 3.3219 bits for 10 uniform classes
    print(f"   --> Mean Softmax Prediction Entropy: {avg_pred_entropy:.6f} bits (Ideal Maximum Uncertainty = {max_possible_entropy:.4f} bits)")
    
    # Save CSV Summary
    df_entropy_class.to_csv(OUT_CSV_PATH, index=False)
    print(f"\nSaved Entropy Summary Metrics to: {OUT_CSV_PATH}")
    
    # ----------------------------------------------------
    # Part 4: Generate Publication Quality Visualizations
    # ----------------------------------------------------
    print("\n4. Generating Entropy Visualization Figures...")
    
    # Plot 1: Class-Wise Spatial Entropy (Raw vs ASM Reconstructed)
    plt.figure(figsize=(12, 5), dpi=300)
    x_indices = np.arange(len(classes))
    width = 0.35
    
    class_labels_clean = [c.replace('_', ' ') for c in classes]
    
    plt.bar(x_indices - width/2, df_entropy_class['mean_raw_entropy'], width, label='Raw Out-of-Focus Hologram', color='#1f77b4', edgecolor='black', alpha=0.85)
    plt.bar(x_indices + width/2, df_entropy_class['mean_recon_entropy'], width, label='ASM In-Focus Reconstructed Field', color='#2ca02c', edgecolor='black', alpha=0.85)
    
    plt.xticks(x_indices, class_labels_clean, rotation=25, ha='right', fontsize=9, fontweight='bold')
    plt.ylabel("Shannon Spatial Entropy (bits)", fontsize=11, fontweight='bold')
    plt.title("Spatial Shannon Entropy Comparison Across 10 Biological Classes (Raw vs In-Focus ASM)", fontsize=11, fontweight='bold', pad=12)
    plt.legend(fontsize=9, loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.5, axis='y')
    plt.tight_layout()
    plt.savefig(OUT_CLASS_PLOT, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Class Entropy Figure to: {OUT_CLASS_PLOT}")
    
    # Plot 2: Minimum Entropy Focus Criterion Curve vs z
    plt.figure(figsize=(8, 5), dpi=300)
    plt.plot(z_eval_range_mm, focus_entropy_list, 'b-o', linewidth=2, markersize=5, label='Reconstruction Field Entropy H(z)')
    plt.axvline(x=11.0, color='r', linestyle='--', linewidth=1.5, label='Ground-Truth Focal Distance z = 11.0 mm')
    plt.axvline(x=optimal_focus_z, color='g', linestyle=':', linewidth=2, label=f'Min Entropy Focal z = {optimal_focus_z:.2f} mm')
    
    plt.title("Minimum Entropy Criterion: Reconstruction Field Entropy H(z) vs Distance z", fontsize=11, fontweight='bold', pad=12)
    plt.xlabel("Propagation Distance z (mm)", fontsize=10, fontweight='bold')
    plt.ylabel("Shannon Spatial Entropy (bits)", fontsize=10, fontweight='bold')
    plt.legend(fontsize=9, loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUT_FOCUS_PLOT, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Focus Entropy Curve to: {OUT_FOCUS_PLOT}")
    
    print("\n==================================================")
    print("   Entropy Analysis & Visualization Complete!")
    print("==================================================")

if __name__ == '__main__':
    main()
