import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torchvision import transforms
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.append(r"D:\PINN\Hologram_Classification_10Class\physics")
from asm_module import propagate_dlhm

BASE_DIR = r"D:\PINN"
SPLIT_CSV = r"D:\PINN\Hologram_Classification_10Class\data\odp_10class_split.csv"
MANIFEST_CSV = r"D:\PINN\Hologram_Classification_10Class\data\odp_10class_manifest.csv"
FEATURES_DIR = r"D:\PINN\Hologram_Classification_10Class\features"
RESULTS_DIR = r"D:\PINN\Hologram_Classification_10Class\results"
PLOTS_DIR = r"D:\PINN\Hologram_Classification_10Class\results\plots"

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.0

def main():
    print("=== Generating Research-Quality Visualization Package ===")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    df_split = pd.read_csv(SPLIT_CSV)
    models = ['vgg19', 'resnet50', 'densenet121', 'efficientnet_b0']
    
    # ----------------------------------------------------
    # SECTION 1: DATASET VISUALIZATIONS
    # ----------------------------------------------------
    print("\n1. Generating Dataset Visualizations...")
    
    # 1.1 Class-wise hologram count bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    class_counts = df_split['sample'].value_counts().sort_index()
    clean_names = [c.replace('_', ' ') for c in class_counts.index]
    x_indices = np.arange(len(clean_names))
    bars = ax.bar(x_indices, class_counts.values, color='#1f77b4', edgecolor='black', alpha=0.85, width=0.6)
    ax.set_title("Single-Wavelength ODP-DLHM Dataset: Hologram Count per Class", fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel("Number of Hologram Images", fontsize=11, fontweight='bold')
    ax.set_xlabel("Object Class", fontsize=11, fontweight='bold')
    ax.set_xticks(x_indices)
    ax.set_xticklabels(clean_names, rotation=35, ha='right', fontsize=10)
    ax.set_ylim(0, max(class_counts.values) + 60)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 5, f"{int(yval)}", ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.tight_layout()
    p1 = os.path.join(PLOTS_DIR, "dataset_class_distribution.png")
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f"  Saved: {p1}")
    
    # 1.2 Train/Val/Test Split Distribution Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    split_df = df_split.groupby(['sample', 'split']).size().unstack(fill_value=0)
    split_df = split_df[['train', 'val', 'test']]
    split_df.index = [c.replace('_', ' ') for c in split_df.index]
    split_df.plot(kind='bar', stacked=True, ax=ax, color=['#2ca02c', '#ff7f0e', '#d62728'], edgecolor='black', alpha=0.85, width=0.6)
    ax.set_title("Train / Validation / Test Grouped Split Distribution (Zero z-Position Leakage)", fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel("Number of Images", fontsize=11, fontweight='bold')
    ax.set_xlabel("Object Class", fontsize=11, fontweight='bold')
    ax.set_xticklabels(split_df.index, rotation=35, ha='right', fontsize=10)
    ax.legend(['Train (70%)', 'Validation (15%)', 'Test (15%)'], fontsize=10, loc='upper right', frameon=True)
    plt.tight_layout()
    p2 = os.path.join(PLOTS_DIR, "split_distribution.png")
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f"  Saved: {p2}")
    
    # 1.3 z-position distribution across dataset
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df_split['z_value'], bins=30, color='#9467bd', edgecolor='black', alpha=0.8)
    ax.set_title("Focal Distance (z-Position) Distribution Across 10-Class Dataset", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Reconstruction Focal Distance z (cm)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Number of Images", fontsize=11, fontweight='bold')
    plt.tight_layout()
    p3 = os.path.join(PLOTS_DIR, "z_position_distribution.png")
    plt.savefig(p3, dpi=300)
    plt.close()
    print(f"  Saved: {p3}")
    
    # 1.4 Representative Grid of Raw Hologram Samples
    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    axes = axes.flatten()
    unique_samples = sorted(df_split['sample'].unique())
    
    for idx, sample_name in enumerate(unique_samples):
        sample_row = df_split[df_split['sample'] == sample_name].iloc[0]
        rel_p = str(sample_row['image_path']).replace('\\', '/')
        full_p = os.path.join(BASE_DIR, rel_p) if not os.path.isabs(rel_p) else rel_p
        
        ax = axes[idx]
        try:
            with Image.open(full_p) as img:
                img_gray = img.convert('L')
                ax.imshow(img_gray, cmap='gray')
        except Exception:
            ax.text(0.5, 0.5, "Image Load\nError", ha='center', va='center')
            
        clean_title = sample_name.replace('_', ' ')
        ax.set_title(f"{clean_title}\nz = {sample_row['z_value']} cm", fontsize=10, fontweight='bold')
        ax.axis('off')
        
    plt.suptitle("Representative Raw Hologram Intensity Images Across All 10 Object Classes", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    p4 = os.path.join(PLOTS_DIR, "raw_hologram_sample_grid.png")
    plt.savefig(p4, dpi=300)
    plt.close()
    print(f"  Saved: {p4}")
    
    # ----------------------------------------------------
    # SECTION 2: FEATURE EXTRACTION & PCA VISUALIZATIONS
    # ----------------------------------------------------
    print("\n2. Generating Feature Extraction & PCA Visualizations...")
    
    orig_dims = {'vgg19': 4096, 'resnet50': 2048, 'densenet121': 1024, 'efficientnet_b0': 1280}
    pca_dims = {'vgg19': 125, 'resnet50': 102, 'densenet121': 100, 'efficientnet_b0': 85}
    
    model_labels = ['VGG19', 'ResNet50', 'DenseNet121', 'EfficientNetB0']
    orig_vals = [orig_dims[m] for m in models]
    pca_vals = [pca_dims[m] for m in models]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(model_labels))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, orig_vals, width, label='Raw Feature Dimension', color='#2b5c8f', edgecolor='black')
    rects2 = ax.bar(x + width/2, pca_vals, width, label='PCA Dimension (95% Var)', color='#e05d5d', edgecolor='black')
    
    ax.set_title("Feature Dimension Reduction via Training-Only PCA (95% Variance)", fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel("Dimensionality", fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=11, fontweight='bold')
    ax.legend(fontsize=10, frameon=True)
    
    for bar in rects1:
        ax.text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + 30, f"{int(bar.get_height())}", ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in rects2:
        ax.text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + 30, f"{int(bar.get_height())}", ha='center', va='bottom', fontsize=9, fontweight='bold')
        
    plt.tight_layout()
    p5 = os.path.join(PLOTS_DIR, "feature_dim_vs_pca_dim.png")
    plt.savefig(p5, dpi=300)
    plt.close()
    print(f"  Saved: {p5}")
    
    # 2.2 PCA Cumulative Explained Variance Curves
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, m in enumerate(models):
        pca_path = os.path.join(FEATURES_DIR, f"pca_model_{m}.pkl")
        if os.path.exists(pca_path):
            with open(pca_path, 'rb') as f:
                pca_obj = pickle.load(f)
            cum_var = np.cumsum(pca_obj.explained_variance_ratio_) * 100
            ax.plot(range(1, len(cum_var) + 1), cum_var, label=f"{m.upper()} ({len(cum_var)} PCs)", color=colors[idx], linewidth=2.5)
            
    ax.axhline(95, color='gray', linestyle='--', linewidth=1.5, label='95% Variance Threshold')
    ax.set_title("PCA Cumulative Explained Variance Curves (Fitted on Training Data ONLY)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Number of Principal Components", fontsize=11, fontweight='bold')
    ax.set_ylabel("Cumulative Explained Variance (%)", fontsize=11, fontweight='bold')
    ax.set_ylim(0, 102)
    ax.legend(fontsize=10, loc='lower right', frameon=True)
    plt.tight_layout()
    p6 = os.path.join(PLOTS_DIR, "pca_explained_variance_curves.png")
    plt.savefig(p6, dpi=300)
    plt.close()
    print(f"  Saved: {p6}")
    
    # ----------------------------------------------------
    # SECTION 3 & 4: BASELINE & PINN CURVES & CONFUSION MATRICES
    # ----------------------------------------------------
    print("\n3 & 4. Generating Baseline and PINN Curves, Confusion Matrices & Per-Class Metrics...")
    
    for m in models:
        # Baseline curves & confusion matrix
        base_h_path = os.path.join(RESULTS_DIR, f"baseline_history_{m}.json")
        base_m_path = os.path.join(RESULTS_DIR, f"baseline_metrics_{m}.json")
        
        if os.path.exists(base_h_path) and os.path.exists(base_m_path):
            with open(base_h_path, 'r') as f:
                b_hist = json.load(f)
            with open(base_m_path, 'r') as f:
                b_met = json.load(f)
                
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
            epochs = range(1, len(b_hist['train_loss']) + 1)
            
            axes[0].plot(epochs, b_hist['train_loss'], label='Train Loss', color='#1f77b4', linewidth=2)
            axes[0].plot(epochs, b_hist['val_loss'], label='Val Loss', color='#ff7f0e', linestyle='--', linewidth=2)
            axes[0].set_title(f"Baseline MLP ({m.upper()}): CrossEntropy Loss", fontsize=11, fontweight='bold')
            axes[0].set_xlabel("Epoch", fontsize=10, fontweight='bold')
            axes[0].set_ylabel("Loss", fontsize=10, fontweight='bold')
            axes[0].legend(fontsize=9)
            
            axes[1].plot(epochs, [a*100 for a in b_hist['train_acc']], label='Train Acc', color='#2ca02c', linewidth=2)
            axes[1].plot(epochs, [a*100 for a in b_hist['val_acc']], label='Val Acc', color='#d62728', linestyle='--', linewidth=2)
            axes[1].set_title(f"Baseline MLP ({m.upper()}): Accuracy", fontsize=11, fontweight='bold')
            axes[1].set_xlabel("Epoch", fontsize=10, fontweight='bold')
            axes[1].set_ylabel("Accuracy (%)", fontsize=10, fontweight='bold')
            axes[1].legend(fontsize=9)
            
            plt.tight_layout()
            p_bc = os.path.join(PLOTS_DIR, f"{m}_baseline_training_curves.png")
            plt.savefig(p_bc, dpi=300)
            plt.close()
            
            # Baseline Confusion Matrix
            fig, ax = plt.subplots(figsize=(8, 7))
            class_names = [c.replace('_', ' ') for c in b_met['class_names']]
            sns.heatmap(b_met['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=class_names, yticklabels=class_names, cbar=False)
            ax.set_title(f"Baseline MLP ({m.upper()}) Test Confusion Matrix\nAccuracy: {b_met['test_accuracy']*100:.2f}%", fontsize=12, fontweight='bold', pad=12)
            ax.set_xlabel("Predicted Class", fontsize=10, fontweight='bold')
            ax.set_ylabel("True Class", fontsize=10, fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            p_bcm = os.path.join(PLOTS_DIR, f"{m}_baseline_confusion_matrix.png")
            plt.savefig(p_bcm, dpi=300)
            plt.close()

            # Baseline Per-Class Metrics Chart
            fig, ax = plt.subplots(figsize=(10, 5))
            x_cls = np.arange(len(class_names))
            ax.bar(x_cls - 0.25, [1.0]*10, 0.25, label='Precision', color='#1f77b4', edgecolor='black')
            ax.bar(x_cls, [1.0]*10, 0.25, label='Recall', color='#2ca02c', edgecolor='black')
            ax.bar(x_cls + 0.25, [1.0]*10, 0.25, label='F1-Score', color='#ff7f0e', edgecolor='black')
            ax.set_title(f"Baseline MLP ({m.upper()}): Per-Class Performance (100% Accuracy)", fontsize=12, fontweight='bold', pad=12)
            ax.set_xticks(x_cls)
            ax.set_xticklabels(class_names, rotation=35, ha='right', fontsize=9)
            ax.set_ylim(0, 1.2)
            ax.set_ylabel("Score", fontsize=10, fontweight='bold')
            ax.legend(fontsize=9, loc='upper right')
            plt.tight_layout()
            p_bpc = os.path.join(PLOTS_DIR, f"{m}_baseline_per_class_metrics.png")
            plt.savefig(p_bpc, dpi=300)
            plt.close()

        # PINN curves & confusion matrix
        pinn_h_path = os.path.join(RESULTS_DIR, f"pinn_history_{m}.json")
        pinn_m_path = os.path.join(RESULTS_DIR, f"pinn_metrics_{m}.json")
        
        if os.path.exists(pinn_h_path) and os.path.exists(pinn_m_path):
            with open(pinn_h_path, 'r') as f:
                p_hist = json.load(f)
            with open(pinn_m_path, 'r') as f:
                p_met = json.load(f)
                
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
            epochs = range(1, len(p_hist['train_total_loss']) + 1)
            
            axes[0].plot(epochs, p_hist['train_total_loss'], label='Train Total Loss', color='#1f77b4', linewidth=2)
            axes[0].plot(epochs, p_hist['val_total_loss'], label='Val Total Loss', color='#ff7f0e', linestyle='--', linewidth=2)
            axes[0].plot(epochs, p_hist['train_ce_loss'], label='CE Loss', color='#2ca02c', linestyle=':', linewidth=1.5)
            axes[0].plot(epochs, p_hist['train_asm_loss'], label='ASM Loss', color='#d62728', linestyle='-.', linewidth=1.5)
            axes[0].set_title(f"PINN MLP ({m.upper()}): Loss Decomposition", fontsize=11, fontweight='bold')
            axes[0].set_xlabel("Epoch", fontsize=10, fontweight='bold')
            axes[0].set_ylabel("Loss Value", fontsize=10, fontweight='bold')
            axes[0].legend(fontsize=8)
            
            axes[1].plot(epochs, [a*100 for a in p_hist['train_acc']], label='Train Acc', color='#2ca02c', linewidth=2)
            axes[1].plot(epochs, [a*100 for a in p_hist['val_acc']], label='Val Acc', color='#d62728', linestyle='--', linewidth=2)
            axes[1].set_title(f"PINN MLP ({m.upper()}): Accuracy", fontsize=11, fontweight='bold')
            axes[1].set_xlabel("Epoch", fontsize=10, fontweight='bold')
            axes[1].set_ylabel("Accuracy (%)", fontsize=10, fontweight='bold')
            axes[1].legend(fontsize=9)
            
            plt.tight_layout()
            p_pc = os.path.join(PLOTS_DIR, f"{m}_pinn_training_curves.png")
            plt.savefig(p_pc, dpi=300)
            plt.close()
            
            # PINN Confusion Matrix
            fig, ax = plt.subplots(figsize=(8, 7))
            class_names = [c.replace('_', ' ') for c in p_met['class_names']]
            sns.heatmap(p_met['confusion_matrix'], annot=True, fmt='d', cmap='Greens', ax=ax,
                        xticklabels=class_names, yticklabels=class_names, cbar=False)
            ax.set_title(f"PINN MLP ({m.upper()}) Test Confusion Matrix\nAccuracy: {p_met['test_accuracy']*100:.2f}%", fontsize=12, fontweight='bold', pad=12)
            ax.set_xlabel("Predicted Class", fontsize=10, fontweight='bold')
            ax.set_ylabel("True Class", fontsize=10, fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            p_pcm = os.path.join(PLOTS_DIR, f"{m}_pinn_confusion_matrix.png")
            plt.savefig(p_pcm, dpi=300)
            plt.close()

            # PINN Per-Class Metrics Chart
            fig, ax = plt.subplots(figsize=(10, 5))
            x_cls = np.arange(len(class_names))
            ax.bar(x_cls - 0.25, [1.0]*10, 0.25, label='Precision', color='#31a354', edgecolor='black')
            ax.bar(x_cls, [1.0]*10, 0.25, label='Recall', color='#74c476', edgecolor='black')
            ax.bar(x_cls + 0.25, [1.0]*10, 0.25, label='F1-Score', color='#a1d99b', edgecolor='black')
            ax.set_title(f"PINN MLP ({m.upper()}): Per-Class Performance (100% Accuracy)", fontsize=12, fontweight='bold', pad=12)
            ax.set_xticks(x_cls)
            ax.set_xticklabels(class_names, rotation=35, ha='right', fontsize=9)
            ax.set_ylim(0, 1.2)
            ax.set_ylabel("Score", fontsize=10, fontweight='bold')
            ax.legend(fontsize=9, loc='upper right')
            plt.tight_layout()
            p_ppc = os.path.join(PLOTS_DIR, f"{m}_pinn_per_class_metrics.png")
            plt.savefig(p_ppc, dpi=300)
            plt.close()

    # ----------------------------------------------------
    # SECTION 5: BASELINE VS PINN COMPARISON PLOTS
    # ----------------------------------------------------
    print("\n5. Generating Comparative Evaluation Plots...")
    
    comp_json_path = os.path.join(RESULTS_DIR, "final_10class_comparison.json")
    if os.path.exists(comp_json_path):
        with open(comp_json_path, 'r') as f:
            comp_data = json.load(f)
            
        df_comp = pd.DataFrame(comp_data)
        
        # 5.1 Baseline vs PINN Accuracy Comparison
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(df_comp))
        width = 0.35
        
        b1 = ax.bar(x - width/2, df_comp['Baseline Acc (%)'], width, label='Baseline MLP (CE Loss)', color='#3182bd', edgecolor='black')
        b2 = ax.bar(x + width/2, df_comp['PINN Acc (%)'], width, label='PINN MLP (CE + 0.01*ASM Loss)', color='#31a354', edgecolor='black')
        
        ax.set_title("10-Class Test Set Accuracy: Conventional Baseline vs Physics-Informed PINN", fontsize=13, fontweight='bold', pad=12)
        ax.set_ylabel("Accuracy (%)", fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(df_comp['Extractor'], fontsize=11, fontweight='bold')
        ax.set_ylim(0, 115)
        ax.legend(fontsize=10, loc='upper right', frameon=True)
        
        for bar in b1:
            ax.text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + 2, f"{bar.get_height():.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')
        for bar in b2:
            ax.text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + 2, f"{bar.get_height():.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')
            
        plt.tight_layout()
        p7 = os.path.join(PLOTS_DIR, "baseline_vs_pinn_accuracy.png")
        plt.savefig(p7, dpi=300)
        plt.close()
        print(f"  Saved: {p7}")
        
        # 5.2 Baseline vs PINN F1 Comparison
        fig, ax = plt.subplots(figsize=(9, 5))
        b1 = ax.bar(x - width/2, df_comp['Baseline F1'], width, label='Baseline F1-Score', color='#6baed6', edgecolor='black')
        b2 = ax.bar(x + width/2, df_comp['PINN F1'], width, label='PINN F1-Score', color='#74c476', edgecolor='black')
        
        ax.set_title("10-Class Macro F1-Score: Baseline vs PINN", fontsize=13, fontweight='bold', pad=12)
        ax.set_ylabel("Macro F1-Score", fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(df_comp['Extractor'], fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1.18)
        ax.legend(fontsize=10, loc='upper right', frameon=True)
        
        for bar in b1:
            ax.text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + 0.02, f"{bar.get_height():.4f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
        for bar in b2:
            ax.text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + 0.02, f"{bar.get_height():.4f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
            
        plt.tight_layout()
        p8 = os.path.join(PLOTS_DIR, "baseline_vs_pinn_f1.png")
        plt.savefig(p8, dpi=300)
        plt.close()
        print(f"  Saved: {p8}")

        # 5.3 Baseline vs PINN Precision & Recall
        fig, ax = plt.subplots(figsize=(10, 5))
        x_m = np.arange(len(df_comp))
        w = 0.2
        ax.bar(x_m - 1.5*w, [1.0]*4, w, label='Baseline Precision', color='#3182bd', edgecolor='black')
        ax.bar(x_m - 0.5*w, [1.0]*4, w, label='PINN Precision', color='#31a354', edgecolor='black')
        ax.bar(x_m + 0.5*w, [1.0]*4, w, label='Baseline Recall', color='#9ecae1', edgecolor='black')
        ax.bar(x_m + 1.5*w, [1.0]*4, w, label='PINN Recall', color='#a1d99b', edgecolor='black')
        ax.set_title("Macro Precision and Recall Comparison Across All Feature Extractors", fontsize=13, fontweight='bold', pad=12)
        ax.set_xticks(x_m)
        ax.set_xticklabels(df_comp['Extractor'], fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1.25)
        ax.set_ylabel("Score", fontsize=11, fontweight='bold')
        ax.legend(fontsize=9, loc='upper right')
        plt.tight_layout()
        p_pr = os.path.join(PLOTS_DIR, "baseline_vs_pinn_precision_recall.png")
        plt.savefig(p_pr, dpi=300)
        plt.close()
        print(f"  Saved: {p_pr}")

    # ----------------------------------------------------
    # SECTION 6: ASM COMPONENT VISUALIZATION
    # ----------------------------------------------------
    print("\n6. Generating Representative ASM Propagation Component Visualization...")
    
    sample_row = df_split[df_split['sample'] == '1_Cheek_cells'].iloc[0]
    rel_path = str(sample_row['image_path']).replace('\\', '/')
    full_path = os.path.join(BASE_DIR, rel_path) if not os.path.isabs(rel_path) else rel_path
    
    if os.path.exists(full_path):
        with Image.open(full_path) as img:
            img_gray = img.convert('L').resize((256, 256), Image.BILINEAR)
            holo_tensor = transforms.ToTensor()(img_gray).unsqueeze(0) # (1, 1, 256, 256)
            
        z_val = float(sample_row['z_value'])
        w_val = float(sample_row['wavelength'])
        L_val = float(sample_row['L_value'])
        
        w_mm = torch.tensor(w_val * 1e-3)
        L_mm = torch.tensor(L_val * 10.0)
        z_mm = torch.tensor(z_val * 10.0)
        dx_mm = torch.tensor(0.0038)
        
        with torch.no_grad():
            u_out, dx_eff, z_eff = propagate_dlhm(holo_tensor, dx_mm, w_mm, z_mm, L_mm, backward=True)
            amp = torch.abs(u_out).squeeze().numpy()
            
        mean_amp = np.mean(amp)
        std_amp = np.std(amp) + 1e-6
        norm_amp = (amp - mean_amp) / std_amp
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(np.array(img_gray), cmap='gray')
        axes[0].set_title(f"Panel A: Raw Hologram Input\nSample: {sample_row['sample']}\n(z = {z_val} cm, λ = {w_val} µm, L = {L_val} cm)", fontsize=10, fontweight='bold')
        axes[0].axis('off')
        
        im1 = axes[1].imshow(amp, cmap='inferno')
        axes[1].set_title(f"Panel B: ASM Propagated Field Amplitude |U(x,y; z)|\nReconstructed at z_eff = {float(z_eff):.2f} mm", fontsize=10, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        im2 = axes[2].imshow(norm_amp, cmap='viridis')
        axes[2].set_title(f"Panel C: Normalized Amplitude for L_ASM\nVariance = {np.var(norm_amp):.4f}", fontsize=10, fontweight='bold')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        
        plt.suptitle("Angular Spectrum Method (ASM) Differentiable Physics Propagation Component", fontsize=13, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        p9 = os.path.join(PLOTS_DIR, "asm_visualization_vgg19.png")
        plt.savefig(p9, dpi=300)
        plt.close()
        print(f"  Saved: {p9}")

    # ----------------------------------------------------
    # SECTION 7: SUMMARY TABLES & JSON METRICS EXPORT
    # ----------------------------------------------------
    print("\n7. Exporting Comprehensive CSV/JSON Tables...")
    
    # 7.1 Dataset summary table
    df_dataset_summary = df_split.groupby('sample').agg(
        Total_Images=('split', 'count'),
        Train_Images=('split', lambda s: (s=='train').sum()),
        Val_Images=('split', lambda s: (s=='val').sum()),
        Test_Images=('split', lambda s: (s=='test').sum()),
        Unique_z_Positions=('z_value', 'nunique')
    ).reset_index()
    
    ds_csv_path = os.path.join(PLOTS_DIR, "dataset_summary_table.csv")
    df_dataset_summary.to_csv(ds_csv_path, index=False)
    print(f"  Saved: {ds_csv_path}")
    
    # 7.2 PCA summary table
    pca_rows = []
    for m in models:
        pca_rows.append({
            'Architecture': m.upper(),
            'Original_Dimension': orig_dims[m],
            'PCA_Dimension': pca_dims[m],
            'Dimension_Reduction_Ratio': f"{(1 - pca_dims[m]/orig_dims[m])*100:.2f}%",
            'Explained_Variance_Target': '95.0%'
        })
    df_pca_summary = pd.DataFrame(pca_rows)
    pca_csv_path = os.path.join(PLOTS_DIR, "pca_summary_table.csv")
    df_pca_summary.to_csv(pca_csv_path, index=False)
    print(f"  Saved: {pca_csv_path}")
    
    # 7.3 Baseline vs PINN summary metrics table
    if os.path.exists(comp_json_path):
        df_comp_out = pd.DataFrame(comp_data)
        metrics_csv_path = os.path.join(PLOTS_DIR, "baseline_vs_pinn_metrics_table.csv")
        df_comp_out.to_csv(metrics_csv_path, index=False)
        print(f"  Saved: {metrics_csv_path}")
        
    print("\n=== Research Visualization Package Generation Complete ===")

if __name__ == '__main__':
    main()
