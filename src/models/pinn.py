import torch
import torch.nn as nn
from src.models.baseline import DLHMBaselineCNN

class DLHMPINN(nn.Module):
    """
    Physics-Informed DLHM Model.
    Reuses the baseline CNN feature extractor to predict z_value continuously.
    The physics incorporation happens in the loss function, not the forward pass, 
    to ensure the inference speed remains identical to the baseline while regularizing
    the training process with wave propagation physics.
    """
    def __init__(self, use_fft_channel=True, input_size=(256, 256), base_channels=16):
        super().__init__()
        # We reuse the verified baseline architecture exactly
        self.backbone = DLHMBaselineCNN(
            use_fft_channel=use_fft_channel, 
            input_size=input_size, 
            base_channels=base_channels
        )
        
    def forward(self, x):
        # Predict continuous raw scalar
        raw_z = self.backbone(x)
        # Constrain to physically valid dataset range using differentiable sigmoid
        # Avoids non-differentiable boundaries (e.g. clamp) while guaranteeing z > 0
        z_min, z_max = 1.1, 3.0
        z_pred = z_min + (z_max - z_min) * torch.sigmoid(raw_z)
        return z_pred
