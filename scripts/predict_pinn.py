import sys
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F

sys.path.insert(0, ".")

from src.models.pinn import DLHMPINN


MODEL_PATH = "results/pinn_training/best_pinn.pth"


def predict(image_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DLHMPINN(
        use_fft_channel=True,
        input_size=(256, 256),
        base_channels=16
    ).to(device)

    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()

    image = Image.open(image_path).convert("L")

    x = torch.from_numpy(
        np.array(image, dtype=np.float32) / 255.0
    ).unsqueeze(0).unsqueeze(0)


    x = F.interpolate(
        x,
        size=(256, 256),
        mode="bilinear",
        align_corners=False
    ).to(device)

    with torch.no_grad():
        z_cm = model(x).item()

    z_mm = z_cm * 10.0

    print(f"\nImage: {image_path}")
    print(f"PINN Predicted z: {z_cm:.4f} cm ({z_mm:.2f} mm)")

    # Optional Ground-Truth Comparison if present in manifest
    try:
        import pandas as pd
        import os
        manifest_path = "Hologram_Classification_10Class/data/odp_10class_manifest.csv"
        if os.path.exists(manifest_path):
            df = pd.read_csv(manifest_path)
            rel_path = image_path.replace('\\', '/')
            match = df[df['image_path'].str.replace('\\', '/').str.endswith(rel_path)]
            if not match.empty:
                z_gt_cm = float(match['z_value'].values[0])
                z_gt_mm = z_gt_cm * 10.0
                err_cm = abs(z_cm - z_gt_cm)
                err_mm = err_cm * 10.0
                rel_err = (err_cm / z_gt_cm) * 100.0
                print(f"Original GT z:    {z_gt_cm:.4f} cm ({z_gt_mm:.2f} mm)")
                print(f"Absolute Error:   {err_cm:.4f} cm ({err_mm:.2f} mm)")
                print(f"Relative Error:   {rel_err:.2f}%")
    except Exception:
        pass


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/predict_pinn.py <image_path>")
        sys.exit(1)

    predict(sys.argv[1])