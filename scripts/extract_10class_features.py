import os
import sys
import time
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image, ImageFile
from concurrent.futures import ThreadPoolExecutor

ImageFile.LOAD_TRUNCATED_IMAGES = True

BASE_DIR = r"D:\PINN"
SPLIT_CSV = r"D:\PINN\Hologram_Classification_10Class\data\odp_10class_split.csv"
OUT_DIR = r"D:\PINN\Hologram_Classification_10Class\features"

class FeatureExtractorWrapper(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name.lower()
        
        if self.model_name == 'vgg19':
            weights = models.VGG19_Weights.DEFAULT
            vgg = models.vgg19(weights=weights)
            self.features = vgg.features
            self.avgpool = vgg.avgpool
            self.classifier = nn.Sequential(*list(vgg.classifier.children())[:5]) # 4096-dim
            self.feature_dim = 4096
            
        elif self.model_name == 'resnet50':
            weights = models.ResNet50_Weights.DEFAULT
            resnet = models.resnet50(weights=weights)
            self.backbone = nn.Sequential(*list(resnet.children())[:-1]) # 2048-dim
            self.feature_dim = 2048
            
        elif self.model_name == 'densenet121':
            weights = models.DenseNet121_Weights.DEFAULT
            densenet = models.densenet121(weights=weights)
            self.features = densenet.features
            self.pool = nn.AdaptiveAvgPool2d((1, 1)) # 1024-dim
            self.feature_dim = 1024
            
        elif self.model_name == 'efficientnet_b0':
            weights = models.EfficientNet_B0_Weights.DEFAULT
            effnet = models.efficientnet_b0(weights=weights)
            self.features = effnet.features
            self.avgpool = effnet.avgpool
            self.feature_dim = 1280
            
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.model_name == 'vgg19':
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
        elif self.model_name == 'resnet50':
            x = self.backbone(x)
            x = torch.flatten(x, 1)
        elif self.model_name == 'densenet121':
            x = self.features(x)
            x = F.relu(x, inplace=True)
            x = self.pool(x)
            x = torch.flatten(x, 1)
        elif self.model_name == 'efficientnet_b0':
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
        return x

# Pre-normalize transform
norm_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_single_image_fast(row):
    rel_path = str(row['image_path']).replace('\\', '/')
    full_path = os.path.join(BASE_DIR, rel_path) if not os.path.isabs(rel_path) else rel_path
    
    try:
        with Image.open(full_path) as img:
            img.draft('RGB', (224, 224))
            img_rgb = img.convert('RGB').resize((224, 224), Image.BILINEAR)
            t_img = norm_transform(img_rgb)
    except Exception:
        t_img = torch.zeros((3, 224, 224), dtype=torch.float32)
        
    return {
        'tensor': t_img,
        'rel_path': rel_path,
        'sample': str(row['sample']),
        'z_value': float(row['z_value']),
        'wavelength': float(row['wavelength']),
        'L_value': float(row['L_value'])
    }

def extract_for_model(model_name: str, device: torch.device, df_split: pd.DataFrame):
    print(f"\n==================================================")
    print(f"   Extracting Features for Architecture: [{model_name.upper()}]")
    print(f"==================================================")
    
    extractor = FeatureExtractorWrapper(model_name).to(device)
    extractor.eval()
    
    batch_size = 64
    
    for split_name in ['train', 'val', 'test']:
        out_path = os.path.join(OUT_DIR, f"{model_name.lower()}_{split_name}.pt")
        if os.path.exists(out_path):
            print(f"  [{model_name}_{split_name}.pt] already exists. Skipping.")
            continue
            
        sub_df = df_split[df_split['split'] == split_name].reset_index(drop=True)
        N = len(sub_df)
        print(f"Processing [{split_name.upper()}] split ({N} images) with fast parallel I/O...")
        
        start_t = time.time()
        
        # Parallel image loading
        rows_list = [row for _, row in sub_df.iterrows()]
        with ThreadPoolExecutor(max_workers=16) as executor:
            items = list(executor.map(load_single_image_fast, rows_list))
            
        extracted_feats = []
        image_paths = []
        class_labels = []
        z_values = []
        wavelengths = []
        l_values = []
        
        for i in range(0, N, batch_size):
            batch_items = items[i:i+batch_size]
            tensors = torch.stack([item['tensor'] for item in batch_items]).to(device)
            
            with torch.no_grad():
                feats = extractor(tensors).cpu()
                
            extracted_feats.append(feats)
            for item in batch_items:
                image_paths.append(item['rel_path'])
                class_labels.append(item['sample'])
                z_values.append(item['z_value'])
                wavelengths.append(item['wavelength'])
                l_values.append(item['L_value'])
                
        all_feats = torch.cat(extracted_feats, dim=0)
        assert len(all_feats) == N
        
        out_dict = {
            'features': all_feats,
            'image_paths': image_paths,
            'class_labels': class_labels,
            'z_values': torch.tensor(z_values, dtype=torch.float32),
            'wavelengths': torch.tensor(wavelengths, dtype=torch.float32),
            'L_values': torch.tensor(l_values, dtype=torch.float32),
            'split': split_name,
            'feature_dim': extractor.feature_dim
        }
        
        os.makedirs(OUT_DIR, exist_ok=True)
        torch.save(out_dict, out_path)
        print(f"Saved [{model_name}_{split_name}.pt]: shape {all_feats.shape} ({time.time()-start_t:.1f}s)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['vgg19', 'resnet50', 'densenet121', 'efficientnet_b0'])
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device for feature extraction: {device}")
    
    df_split = pd.read_csv(SPLIT_CSV)
    
    for m in args.models:
        extract_for_model(m, device, df_split)

if __name__ == '__main__':
    main()
