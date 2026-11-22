import os
import sys
import torch
import torch.nn.functional as F
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

def evaluate_all_objects(extractor="vgg19"):
    print("=" * 80)
    print(f"  PER-OBJECT EVALUATION ACROSS ALL z-PLANES: BEFORE PINN vs AFTER PINN")
    print(f"  Feature Extractor: {extractor.upper()} | Compute Device: {DEVICE}")
    print("=" * 80)

    # 1. Load Split CSV
    df = pd.read_csv(SPLIT_CSV)
    print(f"Total dataset images: {len(df)}")

    # 2. Load Classifiers: Baseline vs PINN
    base_ckpt_path = os.path.join(CLASSIFIER_DIR, f"baseline_mlp_{extractor.lower()}.pth")
    pinn_ckpt_path = os.path.join(CLASSIFIER_DIR, f"pinn_mlp_{extractor.lower()}.pth")

    base_ckpt = torch.load(base_ckpt_path, map_location=DEVICE, weights_only=False)
    pinn_ckpt = torch.load(pinn_ckpt_path, map_location=DEVICE, weights_only=False)

    class_to_idx = pinn_ckpt['class_to_idx']
    idx_to_class = {v: k for k, v in class_to_idx.items()}

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

    # 3. Group by Object Class
    classes_list = sorted(df['sample'].unique())
    table_records = []

    print(f"\nProcessing {len(classes_list)} biological object categories...\n")

    for cls_name in classes_list:
        sub_df = df[df['sample'] == cls_name]
        target_idx = class_to_idx[cls_name]
        total_imgs = len(sub_df)

        base_correct = 0
        pinn_correct = 0
        base_confs = []
        pinn_confs = []

        for _, row in sub_df.iterrows():
            img_rel = row['image_path']
            norm_key = os.path.normpath(img_rel).replace('\\', '/').lower()

            if norm_key in pca_feature_map:
                feat = pca_feature_map[norm_key].unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    base_logits = baseline_clf(feat)
                    pinn_logits = pinn_clf(feat)

                    base_prob = F.softmax(base_logits, dim=-1).squeeze().cpu().numpy()
                    pinn_prob = F.softmax(pinn_logits, dim=-1).squeeze().cpu().numpy()

                    base_pred = int(base_logits.argmax(dim=-1).item())
                    pinn_pred = int(pinn_logits.argmax(dim=-1).item())

                    if base_pred == target_idx:
                        base_correct += 1
                    if pinn_pred == target_idx:
                        pinn_correct += 1

                    base_confs.append(float(base_prob[target_idx]) * 100.0)
                    pinn_confs.append(float(pinn_prob[target_idx]) * 100.0)

        base_acc = (base_correct / total_imgs) * 100.0 if total_imgs > 0 else 0.0
        pinn_acc = (pinn_correct / total_imgs) * 100.0 if total_imgs > 0 else 0.0
        mean_base_conf = float(np.mean(base_confs)) if base_confs else 0.0
        mean_pinn_conf = float(np.mean(pinn_confs)) if pinn_confs else 0.0
        conf_gain = mean_pinn_conf - mean_base_conf

        clean_name = cls_name.split('_', 1)[-1].replace('_', ' ') if '_' in cls_name else cls_name

        table_records.append({
            'Class_ID': cls_name,
            'Object_Name': clean_name,
            'Total_Images': total_imgs,
            'Baseline_Acc_Pct': base_acc,
            'PINN_Acc_Pct': pinn_acc,
            'Acc_Gain_Pct': pinn_acc - base_acc,
            'Baseline_Confidence_Pct': mean_base_conf,
            'PINN_Confidence_Pct': mean_pinn_conf,
            'Confidence_Gain_Pct': conf_gain
        })

    res_df = pd.DataFrame(table_records)

    # Print Summary Table
    print("=" * 105)
    print(f"{'Object Name':<22} | {'Images':<6} | {'Before PINN Acc':<15} | {'After PINN Acc':<14} | {'Before PINN Conf':<16} | {'After PINN Conf':<15} | {'Gain':<6}")
    print("-" * 105)
    for _, r in res_df.iterrows():
        print(f"{r['Object_Name']:<22} | {r['Total_Images']:<6} | {r['Baseline_Acc_Pct']:<14.1f}% | {r['PINN_Acc_Pct']:<13.1f}% | {r['Baseline_Confidence_Pct']:<15.2f}% | {r['PINN_Confidence_Pct']:<14.2f}% | {r['Confidence_Gain_Pct']:+5.2f}%")
    print("=" * 105)

    # Compute Overall Averages
    avg_base_acc = res_df['Baseline_Acc_Pct'].mean()
    avg_pinn_acc = res_df['PINN_Acc_Pct'].mean()
    avg_base_conf = res_df['Baseline_Confidence_Pct'].mean()
    avg_pinn_conf = res_df['PINN_Confidence_Pct'].mean()
    avg_conf_gain = avg_pinn_conf - avg_base_conf

    print("\n--- OVERALL AVERAGE ACROSS ALL 10 OBJECTS ---")
    print(f"Average Accuracy Before PINN (Baseline)   : {avg_base_acc:.2f}%")
    print(f"Average Accuracy After PINN (Proposed)    : {avg_pinn_acc:.2f}%")
    print(f"Average Confidence Before PINN            : {avg_base_conf:.2f}%")
    print(f"Average Confidence After PINN             : {avg_pinn_conf:.2f}%")
    print(f"Average Prediction Certainty Gain         : {avg_conf_gain:+.2f}%")

    out_csv = os.path.join(BASE_DIR, r"results\all_10_objects_before_vs_after_pinn.csv")
    res_df.to_csv(out_csv, index=False)
    print(f"\nSaved full per-object CSV report to: {out_csv}")

if __name__ == '__main__':
    ext = sys.argv[1] if len(sys.argv) > 1 else 'vgg19'
    evaluate_all_objects(ext)

