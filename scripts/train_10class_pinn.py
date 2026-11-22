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
from PIL import Image, ImageFile
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.append(r"D:\PINN\Hologram_Classification_10Class\physics")
from asm_module import ASMPhysicsLoss

BASE_DIR = r"D:\PINN"
FEATURES_DIR = r"D:\PINN\Hologram_Classification_10Class\features"
MODEL_SAVE_DIR = r"D:\PINN\Hologram_Classification_10Class\classifiers"
RESULTS_DIR = r"D:\PINN\Hologram_Classification_10Class\results"

def load_raw_hologram_fast(rel_path, raw_size=(64, 64)):
    full_path = os.path.join(BASE_DIR, rel_path) if not os.path.isabs(rel_path) else rel_path
    try:
        with Image.open(full_path) as img:
            img.draft('L', raw_size)
            img_gray = img.convert('L').resize(raw_size, Image.BILINEAR)
            arr = np.array(img_gray, dtype=np.float32) / 255.0
            t_raw = torch.from_numpy(arr).unsqueeze(0)
    except Exception:
        t_raw = torch.zeros((1, raw_size[0], raw_size[1]), dtype=torch.float32)
    return t_raw

class HologramFeatureAndRawDataset(Dataset):
    def __init__(self, pt_path, class_to_idx=None, raw_size=(64, 64)):
        data = torch.load(pt_path, weights_only=False)
        self.features = data['features'] # (N, D)
        self.class_labels = data['class_labels'] # List[str]
        self.image_paths = data['image_paths']
        self.z_values = data['z_values']
        self.wavelengths = data['wavelengths']
        self.L_values = data['L_values']
        self.raw_size = raw_size
        
        if class_to_idx is None:
            unique_classes = sorted(list(set(self.class_labels)))
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(unique_classes)}
        else:
            self.class_to_idx = class_to_idx
            
        self.labels = torch.tensor([self.class_to_idx[lbl] for lbl in self.class_labels], dtype=torch.long)
        
        # Parallel pre-loading of raw holograms into memory
        with ThreadPoolExecutor(max_workers=16) as executor:
            raw_tensors = list(executor.map(lambda p: load_raw_hologram_fast(p, raw_size), self.image_paths))
        self.raw_holograms = torch.stack(raw_tensors) # (N, 1, 64, 64)
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        return {
            'feature': self.features[idx],
            'label': self.labels[idx],
            'raw_hologram': self.raw_holograms[idx],
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

def train_pinn_model(extractor_name: str, lambda_asm: float = 0.01, max_epochs: int = 100, lr: float = 1e-3):
    print(f"\n==================================================")
    print(f"   Training 10-Class PINN (ASM): [{extractor_name.upper()}] (lambda={lambda_asm})")
    print(f"==================================================")
    
    train_pt = os.path.join(FEATURES_DIR, f"{extractor_name.lower()}_pca_train.pt")
    val_pt = os.path.join(FEATURES_DIR, f"{extractor_name.lower()}_pca_val.pt")
    test_pt = os.path.join(FEATURES_DIR, f"{extractor_name.lower()}_pca_test.pt")
    
    train_dataset = HologramFeatureAndRawDataset(train_pt)
    class_to_idx = train_dataset.class_to_idx
    val_dataset = HologramFeatureAndRawDataset(val_pt, class_to_idx=class_to_idx)
    test_dataset = HologramFeatureAndRawDataset(test_pt, class_to_idx=class_to_idx)
    
    input_dim = train_dataset.features.shape[1]
    num_classes = len(class_to_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLP10Class(input_dim=input_dim, num_classes=num_classes).to(device)
    
    ce_loss_fn = nn.CrossEntropyLoss()
    asm_loss_fn = ASMPhysicsLoss(lambda_asm=lambda_asm, target_size=(64, 64)).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    best_val_loss = float('inf')
    best_model_state = None
    patience = 25
    patience_counter = 0
    
    history = {
        'train_total_loss': [], 'train_ce_loss': [], 'train_asm_loss': [],
        'val_total_loss': [], 'val_ce_loss': [], 'val_asm_loss': [],
        'train_acc': [], 'val_acc': []
    }
    
    for epoch in range(1, max_epochs + 1):
        # Training
        model.train()
        t_total_loss, t_ce_loss, t_asm_loss = 0.0, 0.0, 0.0
        t_correct, t_total = 0, 0
        
        for batch in train_loader:
            x = batch['feature'].to(device)
            y = batch['label'].to(device)
            raw_holo = batch['raw_hologram'].to(device)
            z_v = batch['z_value'].to(device)
            w_v = batch['wavelength'].to(device)
            l_v = batch['L_value'].to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            
            l_ce = ce_loss_fn(logits, y)
            l_asm = asm_loss_fn(raw_holo, z_v, w_v, l_v)
            l_total = asm_loss_fn.total_loss(l_ce, l_asm)
            
            l_total.backward()
            optimizer.step()
            
            bs = len(y)
            t_total_loss += l_total.item() * bs
            t_ce_loss += l_ce.item() * bs
            t_asm_loss += l_asm.item() * bs
            
            preds = logits.argmax(dim=-1)
            t_correct += (preds == y).sum().item()
            t_total += bs
            
        ep_t_tot = t_total_loss / t_total
        ep_t_ce = t_ce_loss / t_total
        ep_t_asm = t_asm_loss / t_total
        ep_t_acc = t_correct / t_total
        
        # Validation
        model.eval()
        v_total_loss, v_ce_loss, v_asm_loss = 0.0, 0.0, 0.0
        v_correct, v_total = 0, 0
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch['feature'].to(device)
                y = batch['label'].to(device)
                raw_holo = batch['raw_hologram'].to(device)
                z_v = batch['z_value'].to(device)
                w_v = batch['wavelength'].to(device)
                l_v = batch['L_value'].to(device)
                
                logits = model(x)
                l_ce = ce_loss_fn(logits, y)
                l_asm = asm_loss_fn(raw_holo, z_v, w_v, l_v)
                l_total = asm_loss_fn.total_loss(l_ce, l_asm)
                
                bs = len(y)
                v_total_loss += l_total.item() * bs
                v_ce_loss += l_ce.item() * bs
                v_asm_loss += l_asm.item() * bs
                
                preds = logits.argmax(dim=-1)
                v_correct += (preds == y).sum().item()
                v_total += bs
                
        ep_v_tot = v_total_loss / v_total
        ep_v_ce = v_ce_loss / v_total
        ep_v_asm = v_asm_loss / v_total
        ep_v_acc = v_correct / v_total
        
        scheduler.step(ep_v_tot)
        
        history['train_total_loss'].append(ep_t_tot)
        history['train_ce_loss'].append(ep_t_ce)
        history['train_asm_loss'].append(ep_t_asm)
        history['val_total_loss'].append(ep_v_tot)
        history['val_ce_loss'].append(ep_v_ce)
        history['val_asm_loss'].append(ep_v_asm)
        history['train_acc'].append(ep_t_acc)
        history['val_acc'].append(ep_v_acc)
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch:03d}/{max_epochs}] | Train (Tot:{ep_t_tot:.4f}, CE:{ep_t_ce:.4f}, ASM:{ep_t_asm:.4f}, Acc:{ep_t_acc*100:.2f}%) | Val (Tot:{ep_v_tot:.4f}, CE:{ep_v_ce:.4f}, ASM:{ep_v_asm:.4f}, Acc:{ep_v_acc*100:.2f}%)")
            
        if ep_v_tot < best_val_loss:
            best_val_loss = ep_v_tot
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch}. Best Val Loss: {best_val_loss:.4f}")
                break
                
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Save PINN Model & History
    model_save_path = os.path.join(MODEL_SAVE_DIR, f"pinn_mlp_{extractor_name.lower()}.pth")
    torch.save({
        'model_state_dict': best_model_state,
        'input_dim': input_dim,
        'num_classes': num_classes,
        'class_to_idx': class_to_idx,
        'extractor_name': extractor_name,
        'lambda_asm': lambda_asm
    }, model_save_path)
    print(f"\nSaved PINN model to: {model_save_path}")
    
    history_save_path = os.path.join(RESULTS_DIR, f"pinn_history_{extractor_name.lower()}.json")
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
        'model': f"PINN MLP ({extractor_name.upper()})",
        'extractor': extractor_name,
        'lambda_asm': lambda_asm,
        'test_accuracy': test_acc,
        'precision_macro': float(prec),
        'recall_macro': float(rec),
        'f1_macro': float(f1),
        'confusion_matrix': cm,
        'class_names': class_names,
        'num_test_samples': len(test_targets)
    }
    
    metrics_path = os.path.join(RESULTS_DIR, f"pinn_metrics_{extractor_name.lower()}.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
        
    print(f"\n--- Final Unseen Test Set Evaluation PINN ({extractor_name.upper()}) ---")
    print(f"  Test Accuracy : {test_acc * 100:.2f}%")
    print(f"  Macro Precision: {prec:.4f}")
    print(f"  Macro Recall   : {rec:.4f}")
    print(f"  Macro F1-Score : {f1:.4f}")
    print(f"Saved PINN evaluation metrics to: {metrics_path}\n")
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['vgg19', 'resnet50', 'densenet121', 'efficientnet_b0'])
    parser.add_argument('--lambda_asm', type=float, default=0.01)
    args = parser.parse_args()
    
    all_metrics = {}
    for m in args.models:
        all_metrics[m] = train_pinn_model(m, lambda_asm=args.lambda_asm)
        
    summary_path = os.path.join(RESULTS_DIR, "pinn_all_models_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Saved PINN summary for all models to: {summary_path}")

if __name__ == '__main__':
    main()
