import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

FEATURES_DIR = r"D:\PINN\Hologram_Classification_10Class\features"
MODEL_SAVE_DIR = r"D:\PINN\Hologram_Classification_10Class\classifiers"
RESULTS_DIR = r"D:\PINN\Hologram_Classification_10Class\results"

class HologramFeatureDataset(Dataset):
    def __init__(self, pt_path, class_to_idx=None):
        data = torch.load(pt_path, weights_only=False)
        self.features = data['features'] # (N, D)
        self.class_labels = data['class_labels'] # List[str]
        self.image_paths = data['image_paths']
        self.z_values = data['z_values']
        self.wavelengths = data['wavelengths']
        self.L_values = data['L_values']
        
        if class_to_idx is None:
            unique_classes = sorted(list(set(self.class_labels)))
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(unique_classes)}
        else:
            self.class_to_idx = class_to_idx
            
        self.labels = torch.tensor([self.class_to_idx[lbl] for lbl in self.class_labels], dtype=torch.long)
        self.idx_to_class = {i: cls_name for cls_name, i in self.class_to_idx.items()}
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        return {
            'feature': self.features[idx],
            'label': self.labels[idx],
            'image_path': self.image_paths[idx],
            'z_value': self.z_values[idx],
            'wavelength': self.wavelengths[idx],
            'L_value': self.L_values[idx]
        }

class MLP10Class(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 10, hidden_dims: list = [256, 128], dropout_rate: float = 0.3):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

def train_baseline_model(extractor_name: str, max_epochs: int = 150, lr: float = 1e-3):
    print(f"\n==================================================")
    print(f"   Training 10-Class Baseline MLP: [{extractor_name.upper()}]")
    print(f"==================================================")
    
    train_pt = os.path.join(FEATURES_DIR, f"{extractor_name.lower()}_pca_train.pt")
    val_pt = os.path.join(FEATURES_DIR, f"{extractor_name.lower()}_pca_val.pt")
    test_pt = os.path.join(FEATURES_DIR, f"{extractor_name.lower()}_pca_test.pt")
    
    train_dataset = HologramFeatureDataset(train_pt)
    class_to_idx = train_dataset.class_to_idx
    val_dataset = HologramFeatureDataset(val_pt, class_to_idx=class_to_idx)
    test_dataset = HologramFeatureDataset(test_pt, class_to_idx=class_to_idx)
    
    input_dim = train_dataset.features.shape[1]
    num_classes = len(class_to_idx)
    print(f"Input Feature Dimension: {input_dim}, Num Classes: {num_classes}")
    print(f"Dataset Counts -> Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLP10Class(input_dim=input_dim, num_classes=num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    best_val_loss = float('inf')
    best_model_state = None
    patience = 25
    patience_counter = 0
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }
    
    for epoch in range(1, max_epochs + 1):
        # Training
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for batch in train_loader:
            x = batch['feature'].to(device)
            y = batch['label'].to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            t_loss += loss.item() * len(y)
            preds = logits.argmax(dim=-1)
            t_correct += (preds == y).sum().item()
            t_total += len(y)
            
        epoch_train_loss = t_loss / t_total
        epoch_train_acc = t_correct / t_total
        
        # Validation
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch['feature'].to(device)
                y = batch['label'].to(device)
                logits = model(x)
                loss = criterion(logits, y)
                
                v_loss += loss.item() * len(y)
                preds = logits.argmax(dim=-1)
                v_correct += (preds == y).sum().item()
                v_total += len(y)
                
        epoch_val_loss = v_loss / v_total
        epoch_val_acc = v_correct / v_total
        
        scheduler.step(epoch_val_loss)
        
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch:03d}/{max_epochs}] | Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc*100:.2f}% | Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc*100:.2f}%")
            
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch}. Best Val Loss: {best_val_loss:.4f}")
                break
                
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Save Model & History
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    model_save_path = os.path.join(MODEL_SAVE_DIR, f"baseline_mlp_{extractor_name.lower()}.pth")
    torch.save({
        'model_state_dict': best_model_state,
        'input_dim': input_dim,
        'num_classes': num_classes,
        'class_to_idx': class_to_idx,
        'extractor_name': extractor_name
    }, model_save_path)
    print(f"\nSaved baseline model to: {model_save_path}")
    
    history_save_path = os.path.join(RESULTS_DIR, f"baseline_history_{extractor_name.lower()}.json")
    with open(history_save_path, 'w') as f:
        json.dump(history, f, indent=2)
        
    # Final Evaluation on Unseen Test Set
    model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            x = batch['feature'].to(device)
            y = batch['label'].to(device)
            logits = model(x)
            preds = logits.argmax(dim=-1)
            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(y.cpu().numpy())
            
    test_acc = float(accuracy_score(test_targets, test_preds))
    prec, rec, f1, _ = precision_recall_fscore_support(test_targets, test_preds, average='macro', zero_division=0)
    cm = confusion_matrix(test_targets, test_preds).tolist()
    
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(num_classes)]
    
    metrics = {
        'model': f"Baseline MLP ({extractor_name.upper()})",
        'extractor': extractor_name,
        'test_accuracy': test_acc,
        'precision_macro': float(prec),
        'recall_macro': float(rec),
        'f1_macro': float(f1),
        'confusion_matrix': cm,
        'class_names': class_names,
        'num_test_samples': len(test_targets)
    }
    
    metrics_path = os.path.join(RESULTS_DIR, f"baseline_metrics_{extractor_name.lower()}.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
        
    print(f"\n--- Final Unseen Test Set Evaluation ({extractor_name.upper()}) ---")
    print(f"  Test Accuracy : {test_acc * 100:.2f}%")
    print(f"  Macro Precision: {prec:.4f}")
    print(f"  Macro Recall   : {rec:.4f}")
    print(f"  Macro F1-Score : {f1:.4f}")
    print(f"Saved test evaluation metrics to: {metrics_path}\n")
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['vgg19', 'resnet50', 'densenet121', 'efficientnet_b0'])
    args = parser.parse_args()
    
    all_metrics = {}
    for m in args.models:
        all_metrics[m] = train_baseline_model(m)
        
    summary_path = os.path.join(RESULTS_DIR, "baseline_all_models_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Saved baseline summary for all models to: {summary_path}")

if __name__ == '__main__':
    main()
