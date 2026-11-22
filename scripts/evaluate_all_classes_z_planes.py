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
RESULTS_DIR = os.path.join(BASE_DIR, r"results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def evaluate_all_classes(extractor="vgg19"):
    print("=" * 90)
    print(f"  EVALUATING ALL 10 OBJECT CLASSES ACROSS Z-PLANES (Z1 to Z30)")
    print(f"  Feature Backbone: {extractor.upper()} | Device: {DEVICE}")
    print("=" * 90)

    # 1. Load Split CSV
    df = pd.read_csv(SPLIT_CSV)
    all_classes = sorted(df['sample'].unique())
    print(f"Found {len(all_classes)} classes in dataset ({len(df)} total images):")
    for c in all_classes:
        print(f"  - {c}")

    # 2. Load Models
    pinn_z_model = None
    if os.path.exists(PINN_Z_PATH):
        try:
            pinn_z_model = DLHMPINN(use_fft_channel=True, input_size=(256, 256), base_channels=16).to(DEVICE)
            pinn_z_model.load_state_dict(torch.load(PINN_Z_PATH, map_location=DEVICE, weights_only=False))
            pinn_z_model.eval()
            print("\nSuccessfully loaded PINN depth regressor (best_pinn.pth)")
        except Exception as e:
            print(f"Warning loading PINN z model: {e}")

    base_ckpt_path = os.path.join(CLASSIFIER_DIR, f"baseline_mlp_{extractor.lower()}.pth")
    pinn_ckpt_path = os.path.join(CLASSIFIER_DIR, f"pinn_mlp_{extractor.lower()}.pth")

    base_ckpt = torch.load(base_ckpt_path, map_location=DEVICE, weights_only=False)
    pinn_ckpt = torch.load(pinn_ckpt_path, map_location=DEVICE, weights_only=False)

    class_to_idx = pinn_ckpt['class_to_idx']

    # Load pre-extracted PCA features
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

    master_records = []
    summary_records = []

    for cls_name in all_classes:
        sub_df = df[df['sample'] == cls_name].copy()
        target_class_idx = class_to_idx[cls_name]

        # Extract z_folder
        sub_df['z_folder'] = sub_df['image_path'].apply(lambda p: os.path.basename(os.path.dirname(p)))
        z_groups = sorted(sub_df['z_folder'].unique(), key=lambda x: int(x.replace('z', '')) if x.replace('z', '').isdigit() else 999)

        print(f"\nProcessing {cls_name} ({len(sub_df)} images across {len(z_groups)} z-groups)...")

        cls_depth_accs = []
        cls_depth_errs = []
        cls_base_accs = []
        cls_pinn_accs = []
        total_cls_imgs = 0

        for z_f in z_groups:
            z_sub = sub_df[sub_df['z_folder'] == z_f]
            gt_z_cm = float(z_sub['z_value'].iloc[0])
            gt_z_mm = gt_z_cm * 10.0

            depth_errors_mm = []
            depth_acc_pcts = []
            base_correct = 0
            pinn_correct = 0
            num_imgs = len(z_sub)
            total_cls_imgs += num_imgs

            for _, row in z_sub.iterrows():
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

            mean_err_mm = float(np.mean(depth_errors_mm)) if depth_errors_mm else 0.0
            mean_depth_acc = float(np.mean(depth_acc_pcts)) if depth_acc_pcts else 0.0
            base_acc = (base_correct / num_imgs) * 100.0 if num_imgs > 0 else 0.0
            pinn_acc = (pinn_correct / num_imgs) * 100.0 if num_imgs > 0 else 0.0

            cls_depth_errs.append(mean_err_mm)
            cls_depth_accs.append(mean_depth_acc)
            cls_base_accs.append(base_acc)
            cls_pinn_accs.append(pinn_acc)

            master_records.append({
                'class_name': cls_name,
                'z_plane': z_f,
                'ground_truth_z_cm': gt_z_cm,
                'ground_truth_z_mm': gt_z_mm,
                'num_images': num_imgs,
                'mean_depth_err_mm': mean_err_mm,
                'pinn_depth_accuracy_pct': mean_depth_acc,
                'baseline_class_acc_pct': base_acc,
                'pinn_class_acc_pct': pinn_acc,
                'class_gain_pct': pinn_acc - base_acc
            })

        # Summary for this class
        avg_depth_err = float(np.mean(cls_depth_errs))
        avg_depth_acc = float(np.mean(cls_depth_accs))
        avg_base_acc = float(np.mean(cls_base_accs))
        avg_pinn_acc = float(np.mean(cls_pinn_accs))

        summary_records.append({
            'class_name': cls_name,
            'total_images': total_cls_imgs,
            'num_z_planes': len(z_groups),
            'avg_baseline_class_acc_pct': avg_base_acc,
            'avg_pinn_class_acc_pct': avg_pinn_acc,
            'avg_class_gain_pct': avg_pinn_acc - avg_base_acc,
            'avg_pinn_depth_acc_pct': avg_depth_acc,
            'avg_depth_error_mm': avg_depth_err
        })

    master_df = pd.DataFrame(master_records)
    summary_df = pd.DataFrame(summary_records)

    master_csv = os.path.join(RESULTS_DIR, "all_10_classes_z1_to_z30_evaluation.csv")
    summary_csv = os.path.join(RESULTS_DIR, "all_10_classes_summary_across_z.csv")

    master_df.to_csv(master_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    print("\n" + "=" * 90)
    print("  OVERALL SUMMARY ACROSS ALL 10 OBJECT CLASSES (AVERAGED OVER Z-PLANES)")
    print("=" * 90)
    print(f"{'Class Name':<26} | {'Total Img':<10} | {'Base Acc %':<11} | {'PINN Acc %':<11} | {'Depth Acc %':<11} | {'Depth Err (mm)':<14}")
    print("-" * 90)
    for _, r in summary_df.iterrows():
        print(f"{r['class_name']:<26} | {int(r['total_images']):<10} | {r['avg_baseline_class_acc_pct']:<11.2f}% | {r['avg_pinn_class_acc_pct']:<11.2f}% | {r['avg_pinn_depth_acc_pct']:<11.2f}% | {r['avg_depth_error_mm']:<14.3f}")
    print("=" * 90)
    print(f"Master per-plane CSV saved to: {master_csv}")
    print(f"Summary CSV saved to: {summary_csv}")

if __name__ == '__main__':
    evaluate_all_classes()

