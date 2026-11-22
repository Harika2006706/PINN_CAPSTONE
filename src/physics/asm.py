import torch

def get_frequencies(shape, device):
    """
    Generate unscaled 2D frequency grids for the Fourier domain.
    """
    H, W = shape
    fx = torch.fft.fftfreq(W, d=1.0, device=device)
    fy = torch.fft.fftfreq(H, d=1.0, device=device)
    fy, fx = torch.meshgrid(fy, fx, indexing='ij')
    return fx, fy

def propagate_dlhm(u_in, dx, wavelength, z, L, backward=True):
    """
    Differentiable Equivalent Plane-Wave ASM (Scaled ASM) for DLHM geometry.
    
    Args:
        u_in: Input tensor of shape (B, C, H, W).
        dx: Camera pixel pitch (mm).
        wavelength: Illumination wavelength (mm).
        z: Source-to-sample distance (mm).
        L: Source-to-camera distance (mm).
        backward: If True, backpropagates from camera to sample (z_eff is negative).
                  If False, forward propagates.
                  
    Returns:
        u_out: Propagated complex optical field (B, C, H, W).
        dx_eff: Effective pixel pitch at the reconstruction plane.
        z_eff: Effective propagation distance.
    """
    if wavelength <= 0: raise ValueError("Wavelength must be positive")
    if dx <= 0: raise ValueError("Pixel pitch must be positive")
    if L <= 0 or z <= 0: raise ValueError("L and z must be positive")
    
    B, C, H, W = u_in.shape
    device = u_in.device
    
    # 1. Map to Equivalent Plane-Wave Domain
    # These operations preserve the autograd graph for z
    M = L / z
    dx_eff = dx / M
    z_eff = z * (L - z) / L
    
    if backward:
        z_eff = -z_eff

    if not u_in.is_complex():
        u_in = u_in.to(torch.complex64)
        
    # Helper to broadcast batched metadata (B,) to (B, 1, 1, 1)
    def view_as_batch(x):
        if isinstance(x, torch.Tensor) and x.ndim > 0:
            return x.view(-1, 1, 1, 1)
        return x

    dx_eff_v = view_as_batch(dx_eff)
    z_eff_v = view_as_batch(z_eff)
    wave_v = view_as_batch(wavelength)

    # 2. Compute Frequencies (Unscaled)
    fx_base, fy_base = get_frequencies((H, W), device=device)
    
    # Scale frequencies differentiably to avoid passing tensors to torch.fft.fftfreq
    fx = fx_base / dx_eff_v
    fy = fy_base / dx_eff_v
    
    # 3. Transfer Function (Angular Spectrum Method)
    k_term = 1.0 - (wave_v * fx)**2 - (wave_v * fy)**2
    
    # Handle evanescent waves (frequencies beyond the diffraction limit)
    mask = k_term > 0
    k_term = torch.clamp(k_term, min=0.0) # Prevent NaNs in sqrt gradient
    
    phase = (2 * torch.pi / wave_v) * z_eff_v * torch.sqrt(k_term)
    
    # Transfer function matrix
    H_tf = torch.exp(1j * phase) * mask
    
    # 4. Propagation in Fourier Domain
    U_k = torch.fft.fftn(u_in, dim=(-2, -1))
    U_k_prop = U_k * H_tf
    u_out = torch.fft.ifftn(U_k_prop, dim=(-2, -1))
    
    return u_out, dx_eff, z_eff
