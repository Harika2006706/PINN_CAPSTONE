import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F

sys.path.insert(0, ".")

from src.models.pinn import DLHMPINN
from src.physics.asm import propagate_dlhm


MODEL_PATH = "results/pinn_training/best_pinn.pth"

IMAGE_PATH = sys.argv[1]

# Physical parameters from the manifest
WAVELENGTH = 0.633      # mm
L_VALUE = 18.81         # mm
DX_BASE = 0.0038
DX = DX_BASE * (4640.0 / 256.0)


# -------------------------
# Load PINN
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

model = DLHMPINN(
    use_fft_channel=True,
    input_size=(256, 256),
    base_channels=16
).to(device)

state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state)
model.eval()


# -------------------------
# Load hologram
# -------------------------
image = Image.open(IMAGE_PATH).convert("L")

original = np.array(image, dtype=np.float32) / 255.0

x = torch.from_numpy(original).unsqueeze(0).unsqueeze(0)

x = F.interpolate(
    x,
    size=(256, 256),
    mode="bilinear",
    align_corners=False
).to(device)


# -------------------------
# Predict z
# -------------------------
with torch.no_grad():
    z_pred = model(x).item()

print(f"Predicted z: {z_pred:.4f} mm")


# -------------------------
# ASM reconstruction
# -------------------------
with torch.no_grad():

    field, _, _ = propagate_dlhm(
        x,
        DX,
        WAVELENGTH,
        torch.tensor(z_pred, device=device),
        L_VALUE,
        backward=True
    )

    reconstruction = torch.abs(field)[0, 0].cpu().numpy()


# Normalize for visualization
reconstruction -= reconstruction.min()
reconstruction /= reconstruction.max() + 1e-8


# -------------------------
# Save reconstruction
# -------------------------
output_path = "results/final_pinn/pinn_reconstruction.png"

plt.figure(figsize=(8, 6))
plt.imshow(reconstruction, cmap="gray")
plt.axis("off")
plt.title(f"PINN Reconstruction — z = {z_pred:.4f} mm")
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved: {output_path}")