import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = r"D:\PINN\Hologram_Classification_10Class\results"

def main():
    print("=== Generating Comparative Plots & Final Metrics Summary ===")
    
    models = ['vgg19', 'resnet50', 'densenet121', 'efficientnet_b0']
    
    comparative_rows = []
    
    for m in models:
        base_metrics_path = os.path.join(RESULTS_DIR, f"baseline_metrics_{m}.json")
        pinn_metrics_path = os.path.join(RESULTS_DIR, f"pinn_metrics_{m}.json")
        
        if not (os.path.exists(base_metrics_path) and os.path.exists(pinn_metrics_path)):
            print(f"Skipping {m}: metrics files not found yet.")
            continue
            
        with open(base_metrics_path, 'r') as f:
            b_m = json.load(f)
        with open(pinn_metrics_path, 'r') as f:
            p_m = json.load(f)
            
        comparative_rows.append({
            'Extractor': m.upper(),
            'Baseline Acc (%)': round(b_m['test_accuracy'] * 100, 2),
            'Baseline F1': round(b_m['f1_macro'], 4),
            'PINN Acc (%)': round(p_m['test_accuracy'] * 100, 2),
            'PINN F1': round(p_m['f1_macro'], 4),
            'Acc Delta (%)': round((p_m['test_accuracy'] - b_m['test_accuracy']) * 100, 2)
        })
        
        # Plot confusion matrices side-by-side
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        class_names = [c.replace('_', ' ') for c in b_m['class_names']]
        
        sns.heatmap(b_m['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=axes[0],
                    xticklabels=class_names, yticklabels=class_names)
        axes[0].set_title(f"Baseline MLP ({m.upper()})\nTest Acc: {b_m['test_accuracy']*100:.2f}%")
        axes[0].set_xlabel("Predicted Class")
        axes[0].set_ylabel("True Class")
        axes[0].tick_params(axis='x', rotation=45)
        
        sns.heatmap(p_m['confusion_matrix'], annot=True, fmt='d', cmap='Greens', ax=axes[1],
                    xticklabels=class_names, yticklabels=class_names)
        axes[1].set_title(f"PINN MLP ({m.upper()})\nTest Acc: {p_m['test_accuracy']*100:.2f}%")
        axes[1].set_xlabel("Predicted Class")
        axes[1].set_ylabel("True Class")
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        cm_path = os.path.join(RESULTS_DIR, f"confusion_matrix_comparison_{m}.png")
        plt.savefig(cm_path, dpi=300)
        plt.close()
        print(f"Saved confusion matrix comparison plot to: {cm_path}")
        
    df_comp = pd.DataFrame(comparative_rows)
    print("\n--- 10-CLASS EXPERIMENT COMPARATIVE RESULTS TABLE ---")
    print(df_comp.to_string(index=False))
    
    comp_json_path = os.path.join(RESULTS_DIR, "final_10class_comparison.json")
    with open(comp_json_path, 'w') as f:
        json.dump(comparative_rows, f, indent=2)
    print(f"\nSaved final comparison JSON to: {comp_json_path}")

if __name__ == '__main__':
    main()
