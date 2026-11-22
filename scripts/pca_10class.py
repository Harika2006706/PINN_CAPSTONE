import os
import sys
import pickle
import argparse
import numpy as np
import torch
from sklearn.decomposition import PCA

FEATURES_DIR = r"D:\PINN\Hologram_Classification_10Class\features"

def run_pca_for_extractor(extractor_name: str, variance_threshold: float = 0.95):
    print(f"\n==================================================")
    print(f"   Fitting PCA for Extractor: [{extractor_name.upper()}]")
    print(f"==================================================")
    
    train_file = os.path.join(FEATURES_DIR, f"{extractor_name.lower()}_train.pt")
    val_file = os.path.join(FEATURES_DIR, f"{extractor_name.lower()}_val.pt")
    test_file = os.path.join(FEATURES_DIR, f"{extractor_name.lower()}_test.pt")
    
    if not (os.path.exists(train_file) and os.path.exists(val_file) and os.path.exists(test_file)):
        print(f"Error: Feature files for {extractor_name} not found in {FEATURES_DIR}")
        sys.exit(1)
        
    train_data = torch.load(train_file, weights_only=False)
    val_data = torch.load(val_file, weights_only=False)
    test_data = torch.load(test_file, weights_only=False)
    
    X_train = train_data['features'].numpy()
    X_val = val_data['features'].numpy()
    X_test = test_data['features'].numpy()
    
    orig_dim = X_train.shape[1]
    print(f"Original Feature Dimension: {orig_dim}")
    print(f"Sample counts -> Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    # Fit PCA ONLY on training data
    pca = PCA(n_components=variance_threshold, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    
    n_components = pca.n_components_
    explained_var = pca.explained_variance_ratio_
    cum_explained_var = np.sum(explained_var)
    
    print(f"\nPCA Fitted on Training Data ONLY:")
    print(f"  Selected Components: {n_components}")
    print(f"  Cumulative Explained Variance: {cum_explained_var * 100:.2f}%")
    print(f"  First 5 Variance Ratios: {[round(float(v), 4) for v in explained_var[:5]]}")
    print(f"  Last 5 Variance Ratios:  {[round(float(v), 4) for v in explained_var[-5:]]}")
    
    # Transform Val and Test data using training PCA
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)
    
    # Save PCA Model
    pca_model_path = os.path.join(FEATURES_DIR, f"pca_model_{extractor_name.lower()}.pkl")
    with open(pca_model_path, 'wb') as f:
        pickle.dump(pca, f)
    print(f"\nSaved fitted PCA model to: {pca_model_path}")
    
    # Package transformed datasets
    splits = [('train', train_data, X_train_pca), ('val', val_data, X_val_pca), ('test', test_data, X_test_pca)]
    
    for split_name, orig_dict, X_pca in splits:
        pca_dict = {
            'features': torch.tensor(X_pca, dtype=torch.float32),
            'image_paths': orig_dict['image_paths'],
            'class_labels': orig_dict['class_labels'],
            'z_values': orig_dict['z_values'],
            'wavelengths': orig_dict['wavelengths'],
            'L_values': orig_dict['L_values'],
            'split': split_name,
            'feature_dim': n_components,
            'orig_feature_dim': orig_dim,
            'cum_explained_variance': cum_explained_var
        }
        out_pt_path = os.path.join(FEATURES_DIR, f"{extractor_name.lower()}_pca_{split_name}.pt")
        torch.save(pca_dict, out_pt_path)
        print(f"Saved PCA features [{split_name}]: shape {pca_dict['features'].shape} -> {out_pt_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['vgg19', 'resnet50', 'densenet121', 'efficientnet_b0'])
    parser.add_argument('--variance', type=float, default=0.95)
    args = parser.parse_args()
    
    for m in args.models:
        run_pca_for_extractor(m, variance_threshold=args.variance)

if __name__ == '__main__':
    main()
