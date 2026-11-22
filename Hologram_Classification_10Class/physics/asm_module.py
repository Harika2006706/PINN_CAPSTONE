import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile
import numpy as np

# Enable PIL to handle truncated image files gracefully
ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_frequencies(shape, dx, dy, device):
    """
    Generate 2D spatial frequency grids for the Fourier domain.
    
    Args:
        shape: (H, W) tuple of spatial dimensions.
        dx: Effective pixel pitch along W (x-axis).
        dy: Effective pixel pitch along H (y-axis).
        device: torch device.
        
    Returns:
        fx, fy: 2D frequency meshgrid tensors.
    """
    H, W = shape
    fx = torch.fft.fftfreq(W, d=dx, device=device)
    fy = torch.fft.fftfreq(H, d=dy, device=device)
    fy, fx = torch.meshgrid(fy, fx, indexing='ij')
    return fx, fy


def propagate_dlhm(u_in, dx, wavelength, z, L, backward=True):
    """
    Differentiable Equivalent Plane-Wave Angular Spectrum Method (ASM) for DLHM geometry.
    
    Args:
        u_in: Input raw hologram tensor of shape (B, C, H, W) or (B, H, W).
        dx: Camera pixel pitch in mm (e.g., 0.0038 mm for 3.8 µm pitch).
        wavelength: Illumination wavelength in mm (e.g., 0.000633 mm for 0.633 µm).
        z: Source-to-sample distance in mm (e.g., 11.0 mm for 1.1 cm).
        L: Source-to-camera distance in mm (e.g., 188.1 mm for 18.81 cm).
        backward: If True, backpropagates from camera to sample plane (z_eff is negative).
        
    Returns:
        u_out: Propagated complex optical field of shape (B, C, H, W).
        dx_eff: Effective pixel pitch at the reconstruction plane (mm).
        z_eff: Effective propagation distance (mm).
    """
    if u_in.dim() == 3:
        u_in = u_in.unsqueeze(1) # Ensure (B, C, H, W)
        
    B, C, H, W = u_in.shape
    device = u_in.device
    
    # Ensure float tensors for scalar physical parameters
    if not isinstance(wavelength, torch.Tensor):
        wavelength = torch.tensor(wavelength, device=device, dtype=torch.float32)
    if not isinstance(z, torch.Tensor):
        z = torch.tensor(z, device=device, dtype=torch.float32)
    if not isinstance(L, torch.Tensor):
        L = torch.tensor(L, device=device, dtype=torch.float32)
    if not isinstance(dx, torch.Tensor):
        dx = torch.tensor(dx, device=device, dtype=torch.float32)

    # 1. Map to Equivalent Plane-Wave Domain (DLHM Geometry)
    M = L / z                           # Magnification factor
    dx_eff = dx / M                     # Effective pixel pitch at reconstruction plane
    z_eff = z * (L - z) / L             # Equivalent plane-wave propagation distance
    
    if backward:
        z_eff = -z_eff

    # Convert real input image to complex field if needed
    if not u_in.is_complex():
        u_in_complex = u_in.to(torch.complex64)
    else:
        u_in_complex = u_in

    # 2. Compute 2D Frequency Meshgrid
    # Average dx_eff if batched tensor
    avg_dx_eff = float(dx_eff.mean().item()) if dx_eff.dim() > 0 else float(dx_eff.item())
    fx, fy = get_frequencies((H, W), avg_dx_eff, avg_dx_eff, device=device)
    
    # Expand frequency grid for batching: shape (1, 1, H, W)
    fx = fx.unsqueeze(0).unsqueeze(0)
    fy = fy.unsqueeze(0).unsqueeze(0)
    
    # Reshape z_eff & wavelength for batch broadcasting: (B, 1, 1, 1)
    if z_eff.dim() > 0:
        z_eff_b = z_eff.view(-1, 1, 1, 1)
    else:
        z_eff_b = z_eff
        
    if wavelength.dim() > 0:
        wave_b = wavelength.view(-1, 1, 1, 1)
    else:
        wave_b = wavelength

    # 3. Transfer Function (ASM Phase Propagator)
    k_term = 1.0 - (wave_b * fx)**2 - (wave_b * fy)**2
    
    # Filter out evanescent waves beyond the diffraction limit
    mask = (k_term > 0).float()
    k_term_clamped = torch.clamp(k_term, min=0.0) # Prevent NaNs in sqrt gradient
    
    phase = (2.0 * np.pi / wave_b) * z_eff_b * torch.sqrt(k_term_clamped)
    
    # Transfer function H_tf = exp(j * phase) * mask
    H_tf = torch.exp(1j * phase) * mask
    
    # 4. Propagation in Fourier Domain
    U_k = torch.fft.fftn(u_in_complex, dim=(-2, -1))
    U_k_prop = U_k * H_tf
    u_out = torch.fft.ifftn(U_k_prop, dim=(-2, -1))
    
    return u_out, dx_eff, z_eff


class ASMPhysicsLoss(nn.Module):
    """
    ASM Physics-Informed Loss Module.
    Calculates optical propagation at physical distance z and evaluates the focus/sharpness
    quality of the reconstructed optical field.
    """
    def __init__(self, lambda_asm: float = 0.01, dx_um: float = 3.8, target_size: tuple = (256, 256)):
        super().__init__()
        self.lambda_asm = lambda_asm
        self.dx_mm = dx_um * 1e-3 # Convert µm to mm (0.0038 mm)
        self.target_size = target_size

    def forward(self, holograms: torch.Tensor, z_cm: torch.Tensor, wavelength_um: torch.Tensor, L_cm: torch.Tensor) -> torch.Tensor:
        """
        Computes ASM physics loss L_ASM for a batch of raw holograms.
        
        Args:
            holograms: (B, 1, H, W) raw hologram intensity images [0, 1].
            z_cm: (B,) source-to-sample distance in cm.
            wavelength_um: (B,) illumination wavelength in µm.
            L_cm: (B,) source-to-camera distance in cm.
            
        Returns:
            l_asm: Scalar torch.Tensor containing the physics loss.
        """
        device = holograms.device
        B = holograms.shape[0]

        # Downsample holograms for fast FFT physics computation if needed
        if self.target_size is not None and (holograms.shape[-2] != self.target_size[0] or holograms.shape[-1] != self.target_size[1]):
            holo_proc = F.interpolate(holograms, size=self.target_size, mode='bilinear', align_corners=False)
        else:
            holo_proc = holograms

        # Convert physical units consistently to mm
        wave_mm = (wavelength_um * 1e-3).to(device).float() # µm -> mm
        L_mm = (L_cm * 10.0).to(device).float()             # cm -> mm
        z_mm = (z_cm * 10.0).to(device).float()             # cm -> mm
        dx_mm = torch.tensor(self.dx_mm, device=device, dtype=torch.float32)

        # Propagate hologram to object plane z
        u_out, _, _ = propagate_dlhm(holo_proc, dx_mm, wave_mm, z_mm, L_mm, backward=True)

        # Reconstructed optical amplitude field
        amplitude = torch.abs(u_out)

        # Standardize amplitude per image in batch
        mean_amp = amplitude.mean(dim=(-2, -1), keepdim=True)
        std_amp = amplitude.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        norm_amp = (amplitude - mean_amp) / std_amp

        # Spatial focus metric: Variance of normalized amplitude
        # In-focus optical fields exhibit maximized spatial variance / contrast
        variance = torch.var(norm_amp, dim=(-2, -1))

        # Loss is negative normalized variance (encouraging sharp, in-focus propagation)
        l_asm = -variance.mean()
        return l_asm

    def total_loss(self, l_ce: torch.Tensor, l_asm: torch.Tensor) -> torch.Tensor:
        """
        Combines conventional classification CrossEntropy loss with ASM physics loss:
        L_total = L_CE + λ_ASM * L_ASM
        """
        return l_ce + self.lambda_asm * l_asm
