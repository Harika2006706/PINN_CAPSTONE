import os
import pandas as pd

def main():
    manifest_path = 'data/odp_manifest.csv'
    if not os.path.exists(manifest_path):
        print(f"Manifest not found at {manifest_path}")
        return

    df = pd.read_csv(manifest_path)
    total_target = len(df)
    
    present_count = 0
    missing_paths = []

    for idx, row in df.iterrows():
        # Path might have mixed slashes, normalize it
        img_path = row['image_path'].replace('\\', '/')
        if os.path.exists(img_path):
            present_count += 1
        else:
            missing_paths.append(img_path)

    missing_count = len(missing_paths)

    print(f"Total target images: {total_target}")
    print(f"Local target images actually present: {present_count}")
    print(f"Exact missing count: {missing_count}")
    
    if missing_count > 0:
        print("\nMissing filenames/paths (first 50):")
        for path in missing_paths[:50]:
            print(f" - {path}")
        if missing_count > 50:
            print(f" ... and {missing_count - 50} more.")

if __name__ == '__main__':
    main()
