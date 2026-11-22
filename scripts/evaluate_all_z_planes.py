import os
import sys
import torch
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from collections import defaultdict

ImageFile.LOAD_TRUNCATED_IMAGES = True

BASE_DIR = r"D:\PINN"
sys.path.insert(0, BASE_DIR)

from src.models.pinn import DLHMPINN
from scripts.train_10class_pinn import MLP10Class

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PINN_Z_PATH = os.path.join(BASE_DIR, r"results\pinn_training\best_pinn.pth")
SPLIT_CSV = os.path.join(BASE_DIR, r"Hologram_Classification_10Class\data\odp_10class_split.csv")
CLASSIFIER_DIR = os.path.join(BASE_DIR, r"Hologram_Classification_10Class\classifiers")
FEATURES_DIR = os.path.join(BASE_DIR, r"Hologram_Classification_10Class\features")

def evaluate_cheek_cells_z_stack(extractor="vgg19"):
    print("=" * 70)
    print(f"  EVALUATING CHEEK CELLS z1 to z30: BEFORE PINN vs AFTER PINN")
    print(f"  Feature Backbone: {extractor.upper()} | Device: {DEVICE}")
    print("=" * 70)

    # 1. Load Split CSV
    df = pd.read_csv(SPLIT_CSV)
    df_cheek = df[df['sample'] == '1_Cheek_cells'].copy()
    print(f"Total Cheek cells images found: {len(df_cheek)}")

    # 2. Load Models
    # A. Depth Regressor (PINN Z)
    pinn_z_model = None
    if os.path.exists(PINN_Z_PATH):
        try:
            pinn_z_model = DLHMPINN(use_fft_channel=True, input_size=(256, 256), base_channels=16).to(DEVICE)
            pinn_z_model.load_state_dict(torch.load(PINN_Z_PATH, map_location=DEVICE, weights_only=False))
            pinn_z_model.eval()
            print("Successfully loaded PINN depth regressor (best_pinn.pth)")
        except Exception as e:
            print(f"Warning loading PINN z model: {e}")

    # B. Classifiers: Baseline vs PINN
    base_ckpt_path = os.path.join(CLASSIFIER_DIR, f"baseline_mlp_{extractor.lower()}.pth")
    pinn_ckpt_path = os.path.join(CLASSIFIER_DIR, f"pinn_mlp_{extractor.lower()}.pth")

    base_ckpt = torch.load(base_ckpt_path, map_location=DEVICE, weights_only=False)
    pinn_ckpt = torch.load(pinn_ckpt_path, map_location=DEVICE, weights_only=False)

    class_to_idx = pinn_ckpt['class_to_idx']
    target_class_idx = class_to_idx['1_Cheek_cells']

    # Load pre-extracted PCA features to match image paths
    pca_feature_map = {}
    for split_name in ['train', 'val', 'test']:
        pt_path = os.path.join(FEATURES_DIR, f"{extractor.lower()}_pca_{split_name}.pt")
        if os.path.exists(pt_path):
            data = torch.load(pt_path, weights_only=False)
            for p, feat in zip(data['image_paths'], data['features']):
                norm_p = os.path.normpath(p).replace('\\', '/').lower()
                pca_feature_map[norm_p] = feat

    input_dim = next(iter(pca_feature_map.values())).shape[0]

    baseline_clf = MLP10Class(input_dim=input_dim, num_classes=len(class_to_idx)).to(DEVICE)
    baseline_clf.load_state_dict(base_ckpt['model_state_dict'])
    baseline_clf.eval()

    pinn_clf = MLP10Class(input_dim=input_dim, num_classes=len(class_to_idx)).to(DEVICE)
    pinn_clf.load_state_dict(pinn_ckpt['model_state_dict'])
    pinn_clf.eval()

    # 3. Group by z folder (z1, z2, ..., z30)
    df_cheek['z_folder'] = df_cheek['image_path'].apply(lambda p: os.path.basename(os.path.dirname(p)))
    z_groups = sorted(df_cheek['z_folder'].unique(), key=lambda x: int(x.replace('z', '')) if x.replace('z', '').isdigit() else 999)

    table_records = []

    print(f"\nProcessing {len(z_groups)} z-planes (z1 to z30)...")

    for z_f in z_groups:
        sub_df = df_cheek[df_cheek['z_folder'] == z_f]
        gt_z_cm = float(sub_df['z_value'].iloc[0])
        gt_z_mm = gt_z_cm * 10.0

        depth_errors_mm = []
        depth_acc_pcts = []
        base_correct = 0
        pinn_correct = 0
        total_imgs = len(sub_df)

        for _, row in sub_df.iterrows():
            img_rel = row['image_path']
            img_full = os.path.join(BASE_DIR, img_rel) if not os.path.isabs(img_rel) else img_rel

            # Depth Prediction
            if pinn_z_model is not None and os.path.exists(img_full):
                try:
                    with Image.open(img_full) as raw_img:
                        img_256 = raw_img.convert('L').resize((256, 256), Image.Resampling.BILINEAR)
                        arr_256 = np.array(img_256, dtype=np.float32) / 255.0
                        x_t = torch.from_numpy(arr_256).unsqueeze(0).unsqueeze(0).to(DEVICE)
                        with torch.no_grad():
                            z_pred_cm = abs(pinn_z_model(x_t).item())
                            err_mm = abs(z_pred_cm - gt_z_cm) * 10.0
                            acc_pct = max(0.0, 100.0 - (err_mm / gt_z_mm * 100.0))
                            depth_errors_mm.append(err_mm)
                            depth_acc_pcts.append(acc_pct)
                except Exception:
                    pass

            # Classification
            norm_key = os.path.normpath(img_rel).replace('\\', '/').lower()
            if norm_key in pca_feature_map:
                feat = pca_feature_map[norm_key].unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    base_pred = baseline_clf(feat).argmax(dim=-1).item()
                    pinn_pred = pinn_clf(feat).argmax(dim=-1).item()

                    if base_pred == target_class_idx:
                        base_correct += 1
                    if pinn_pred == target_class_idx:
                        pinn_correct += 1

        mean_err_mm = np.mean(depth_errors_mm) if depth_errors_mm else 0.0
        mean_depth_acc = np.mean(depth_acc_pcts) if depth_acc_pcts else 0.0
        base_acc = (base_correct / total_imgs) * 100.0 if total_imgs > 0 else 0.0
        pinn_acc = (pinn_correct / total_imgs) * 100.0 if total_imgs > 0 else 0.0

        table_records.append({
            'z_plane': z_f,
            'ground_truth_z_cm': gt_z_cm,
            'ground_truth_z_mm': gt_z_mm,
            'mean_depth_err_mm': mean_err_mm,
            'pinn_depth_accuracy_pct': mean_depth_acc,
            'baseline_class_acc_pct': base_acc,
            'pinn_class_acc_pct': pinn_acc,
            'class_gain_pct': pinn_acc - base_acc,
            'num_images': total_imgs
        })

    res_df = pd.DataFrame(table_records)

    # Print Summary Table
    print("\n" + "=" * 95)
    print(f"{'Plane':<7} | {'GT z (cm)':<9} | {'Mean Depth Err':<14} | {'Depth Acc %':<11} | {'Baseline Acc %':<14} | {'PINN Acc %':<10} | {'Gain %':<6}")
    print("-" * 95)
    for _, r in res_df.iterrows():
        print(f"{r['z_plane']:<7} | {r['ground_truth_z_cm']:<9.3f} | {r['mean_depth_err_mm']:<10.3f} mm | {r['pinn_depth_accuracy_pct']:<9.2f}% | {r['baseline_class_acc_pct']:<14.1f}% | {r['pinn_class_acc_pct']:<10.1f}% | {r['class_gain_pct']:+5.1f}%")
    print("=" * 95)

    # Compute Averages
    avg_depth_err = res_df['mean_depth_err_mm'].mean()
    avg_depth_acc = res_df['pinn_depth_accuracy_pct'].mean()
    avg_base_acc = res_df['baseline_class_acc_pct'].mean()
    avg_pinn_acc = res_df['pinn_class_acc_pct'].mean()
    avg_gain = avg_pinn_acc - avg_base_acc

    print("\n--- OVERALL AVERAGE ACROSS ALL 30 z-PLANES ---")
    print(f"Average Depth Error (|Delta z|) : {avg_depth_err:.4f} mm")
    print(f"Average PINN Depth Accuracy      : {avg_depth_acc:.2f}%")
    print(f"Average Baseline Class Accuracy  : {avg_base_acc:.2f}% (Without PINN)")
    print(f"Average PINN Class Accuracy      : {avg_pinn_acc:.2f}% (With PINN)")
    print(f"Average Classification Gain      : {avg_gain:+.2f}%")

    out_csv = os.path.join(BASE_DIR, r"results\cheek_cells_z1_to_z30_evaluation.csv")
    res_df.to_csv(out_csv, index=False)
    print(f"\nSaved full per-plane CSV report to: {out_csv}")

if __name__ == '__main__':
    ext = sys.argv[1] if len(sys.argv) > 1 else 'vgg19'
    evaluate_cheek_cells_z_stack(ext)

