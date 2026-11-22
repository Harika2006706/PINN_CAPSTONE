import os
import pandas as pd
import yaml

def extract_sample(path):
    parts = path.replace('\\', '/').split('/')
    if 'SW-Dataset' in parts:
        try:
            return parts[parts.index('SW-Dataset') + 1]
        except IndexError:
            pass
    return 'Unknown'

def main():
    # Load config
    config_path = 'configs/dataset.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)['dataset']
        
    df = pd.read_csv(config['source_csv'])
    
    # Extract sample names
    df['sample'] = df['path'].apply(extract_sample)
    
    # Apply filtering criteria
    subset = df[(df['Wavelength'] == config['wavelength']) & 
                (df['sample'].isin(config['samples']))].copy()
                
    # Normalize paths (e.g. ./SW-Dataset/... -> data/SW-Dataset/...)
    subset['image_path'] = subset['path'].apply(lambda p: os.path.join('data', p.lstrip('./\\')))
    
    # Generate clean manifest
    manifest = subset[['image_path', 'sample', 'Wavelength', 'L_value', 'z_value']].rename(columns={'Wavelength': 'wavelength'})
    
    # Save manifest
    os.makedirs(os.path.dirname(config['manifest_output']), exist_ok=True)
    manifest.to_csv(config['manifest_output'], index=False)
    
    # Verify existence
    manifest['exists'] = manifest['image_path'].apply(os.path.exists)
    existing_count = manifest['exists'].sum()
    missing_count = len(manifest) - existing_count
    
    print("--- Manifest Summary ---")
    print(f"Total images selected: {len(manifest)}")
    print(f"Images verified on disk: {existing_count}")
    print(f"Images missing on disk: {missing_count} (To be downloaded)")
    
    print("\n--- Counts by Sample ---")
    print(manifest['sample'].value_counts().to_string())
    
    print("\n--- Counts by Z-Value (Overall) ---")
    print(manifest['z_value'].value_counts().sort_index().to_string())
    
    # Estimate total storage
    est_storage_mb = len(manifest) * 3
    print(f"\nEstimated compressed storage: ~{est_storage_mb} MB")

if __name__ == '__main__':
    main()
