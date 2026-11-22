import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

class DLHMBaselineCNN(nn.Module):
    """
    Supervised CNN baseline for Digital Lensless Holographic Microscopy (DLHM).
    Predicts the in-focus reconstruction distance (z_value) directly from the hologram.
    Loosely inspired by FocusNET.
    """
    def __init__(self, use_fft_channel=True, input_size=(256, 256), base_channels=8):
        super().__init__()
        self.use_fft_channel = use_fft_channel
        self.input_size = input_size
        self.in_channels = 2 if use_fft_channel else 1
        
        # Convolutional backbone
        self.features = nn.Sequential(
            self._make_block(self.in_channels, base_channels),              # 128
            self._make_block(base_channels, base_channels * 2),             # 64
            self._make_block(base_channels * 2, base_channels * 4),         # 32
            self._make_block(base_channels * 4, base_channels * 8),         # 16
            self._make_block(base_channels * 8, base_channels * 16)         # 8
        )
        
        # Calculate flatten size
        downsample_factor = 2 ** 5
        feat_h = input_size[0] // downsample_factor
        feat_w = input_size[1] // downsample_factor
        flatten_dim = (base_channels * 16) * feat_h * feat_w
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, 256),
            nn.SiLU(), # Swish activation
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1) # Predicts z_value
        )
        
    def _make_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.SiLU(),
            nn.MaxPool2d(2)
        )
        
    def forward(self, x):
        """
        x: (B, 1, H, W) where H, W can be large (e.g. 3506, 4640)
        """
        # Resize to network input size
        if x.shape[2:] != self.input_size:
            x = F.interpolate(x, size=self.input_size, mode='bilinear', align_corners=False)
            
        if self.use_fft_channel:
            # Generate Amplitude Fourier channel
            fft_x = torch.fft.fftn(x, dim=(-2, -1))
            amp = torch.log(torch.abs(fft_x) + 1e-8)
            # Basic instance normalization for the FFT channel to keep it stable
            mean = amp.mean(dim=(-2, -1), keepdim=True)
            std = amp.std(dim=(-2, -1), keepdim=True) + 1e-8
            amp = (amp - mean) / std
            
            x = torch.cat([x, amp], dim=1) # (B, 2, H', W')
            
        feat = self.features(x)
        out = self.regressor(feat)
        return out.squeeze(-1) # return shape (B,)
