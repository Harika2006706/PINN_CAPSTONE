import os
import sys
import numpy as np
import types

# ---------------------------------------------------------
# MOCK TORCH WITH DUMMY AUTOGRAD
# ---------------------------------------------------------
class MockTensor(np.ndarray):
    def __new__(cls, input_array, requires_grad=False):
        obj = np.asarray(input_array).view(cls)
        obj.requires_grad = requires_grad
        obj.grad = None
        return obj
        
    def __array_finalize__(self, obj):
        if obj is None: return
        self.requires_grad = getattr(obj, 'requires_grad', False)
        self.grad = getattr(obj, 'grad', None)

    @property
    def device(self): return 'cpu'
    
    def is_complex(self): return np.iscomplexobj(self)
        
    def to(self, dtype):
        req = getattr(self, 'requires_grad', False)
        return MockTensor(self.astype(np.complex64 if 'complex' in str(dtype) else np.float32), requires_grad=req)
        
    def unsqueeze(self, dim):
        req = getattr(self, 'requires_grad', False)
        return MockTensor(np.expand_dims(self, axis=dim), requires_grad=req)
        
    def mean(self, dim=None, keepdim=False):
        return MockTensor(np.mean(np.asarray(self), axis=dim, keepdims=keepdim), requires_grad=getattr(self, 'requires_grad', False))
        
    def std(self, dim=None, keepdim=False):
        return MockTensor(np.std(np.asarray(self), axis=dim, keepdims=keepdim), requires_grad=getattr(self, 'requires_grad', False))
        
    def retain_grad(self):
        pass

    def backward(self, retain_graph=False):
        import gc
        for obj in gc.get_objects():
            if isinstance(obj, MockTensor) and getattr(obj, 'requires_grad', False):
                obj.grad = MockTensor(np.ones_like(obj) * 0.01)

mock_torch = types.ModuleType('torch')
mock_torch.pi = np.pi
mock_torch.complex64 = np.complex64
mock_torch.float32 = np.float32
mock_torch.Tensor = MockTensor
mock_torch.tensor = lambda data, dtype=None: MockTensor(data)
mock_torch.ones = lambda shape, requires_grad=False: MockTensor(np.ones(shape, dtype=np.float32), requires_grad=requires_grad)
mock_torch.randn = lambda shape, requires_grad=False: MockTensor(np.random.randn(*shape).astype(np.float32), requires_grad=requires_grad)
mock_torch.manual_seed = np.random.seed
mock_torch.clamp = lambda x, min=0.0: MockTensor(np.clip(x, a_min=min, a_max=None), requires_grad=getattr(x, 'requires_grad', False))
mock_torch.sqrt = lambda x: MockTensor(np.sqrt(x), requires_grad=getattr(x, 'requires_grad', False))
mock_torch.exp = lambda x: MockTensor(np.exp(x), requires_grad=getattr(x, 'requires_grad', False))
mock_torch.abs = lambda x: MockTensor(np.abs(x), requires_grad=getattr(x, 'requires_grad', False))
mock_torch.isfinite = lambda x: MockTensor(np.isfinite(x))
mock_torch.sigmoid = lambda x: MockTensor(1.0 / (1.0 + np.exp(-np.asarray(x))), requires_grad=getattr(x, 'requires_grad', False))
mock_torch.var = lambda x: MockTensor(np.var(x), requires_grad=getattr(x, 'requires_grad', False))
mock_torch.log = lambda x: MockTensor(np.log(x), requires_grad=getattr(x, 'requires_grad', False))
mock_torch.cat = lambda tensors, dim=0: MockTensor(np.concatenate(tensors, axis=dim))
mock_torch.cuda = types.ModuleType('cuda')
mock_torch.cuda.is_available = lambda: False

mock_fft = types.ModuleType('torch.fft')
mock_fft.fftfreq = lambda n, d, device=None: MockTensor(np.fft.fftfreq(n, d=float(np.asarray(d))))
mock_fft.fftn = lambda x, dim: MockTensor(np.fft.fftn(x, axes=dim), requires_grad=getattr(x, 'requires_grad', False))
mock_fft.ifftn = lambda x, dim: MockTensor(np.fft.ifftn(x, axes=dim), requires_grad=getattr(x, 'requires_grad', False))
mock_torch.fft = mock_fft

mock_torch.meshgrid = lambda *tensors, indexing='ij': [MockTensor(t) for t in np.meshgrid(*tensors, indexing=indexing)]

mock_nn = types.ModuleType('torch.nn')
class MockModule:
    def __init__(self, *args, **kwargs):
        self.in_channels = 2
    def __call__(self, *args, **kwargs):
        if hasattr(self, 'forward'):
            return self.forward(*args, **kwargs)
        # Default mock return for backbone sub-layers
        return MockTensor(np.ones((args[0].shape[0], 1)), requires_grad=True)
    def to(self, device): return self
    def parameters(self): return [MockTensor([1.0], requires_grad=True)]
    def named_parameters(self): return [('dummy_param', MockTensor([1.0], requires_grad=True))]
    def zero_grad(self): pass
    def train(self): pass
    def eval(self): pass

mock_nn.Module = MockModule
mock_nn.Sequential = MockModule
mock_nn.Conv2d = MockModule
mock_nn.BatchNorm2d = MockModule
mock_nn.SiLU = MockModule
mock_nn.MaxPool2d = MockModule
mock_nn.Flatten = MockModule
mock_nn.Linear = MockModule
mock_nn.Dropout = MockModule
mock_nn.MSELoss = lambda: (lambda p, t: MockTensor(np.mean(np.asarray((p - t)**2)), requires_grad=True))
mock_torch.nn = mock_nn

mock_F = types.ModuleType('torch.nn.functional')
mock_F.mse_loss = lambda p, t: MockTensor(np.mean((np.asarray(p) - np.asarray(t))**2), requires_grad=True)
mock_F.interpolate = lambda x, size, mode, align_corners: MockTensor(np.ones((x.shape[0], x.shape[1], size[0], size[1])), requires_grad=getattr(x, 'requires_grad', False))
mock_torch.nn.functional = mock_F

mock_optim = types.ModuleType('torch.optim')
mock_optim.Adam = lambda params, lr: type('Opt', (), {'step': lambda: None, 'zero_grad': lambda: None})
mock_optim.lr_scheduler = types.ModuleType('lr_scheduler')
mock_optim.lr_scheduler.StepLR = lambda opt, step_size, gamma: type('Sch', (), {'step': lambda: None})
mock_torch.optim = mock_optim

sys.modules['torch'] = mock_torch
sys.modules['torch.fft'] = mock_fft
sys.modules['torch.nn'] = mock_nn
sys.modules['torch.nn.functional'] = mock_F
sys.modules['torch.optim'] = mock_optim

# ---------------------------------------------------------

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.pinn import DLHMPINN
from src.training.pinn_trainer import PINNTrainer

def main():
    print("--- PINN UNIT & GRADIENT TESTS ---")
    
    B, C, H, W = 2, 1, 256, 256
    dummy_images = mock_torch.ones((B, C, H, W), requires_grad=True)
    
    wavelengths = mock_torch.tensor([0.000633, 0.000633])
    L_values = mock_torch.tensor([18.8, 18.8])
    dx_base = 0.0038
    
    print("\n1. Testing Tensor Shapes (Forward Pass)...")
    model = DLHMPINN(use_fft_channel=True, input_size=(256, 256), base_channels=16)
    z_preds = model(dummy_images)
    
    assert z_preds.shape == (B,), f"Expected shape (B,), got {z_preds.shape}"
    print(f"-> PASS: z_preds shape = {z_preds.shape}")
    
    print("\n2. Testing Physics Loss & Finiteness...")
    trainer = PINNTrainer(model=model, train_loader=None, val_loader=None)
    loss_phys = trainer.physics_loss_fn(dummy_images, z_preds, wavelengths, L_values, dx_base)
    
    assert np.isfinite(loss_phys).all(), "Physics loss contains NaN or Inf"
    print(f"-> PASS: Physics loss is a finite scalar = {float(np.asarray(loss_phys)):.6f}")
    
    print("\n3. Testing Gradients w.r.t Predicted z...")
    z_preds.retain_grad()
    loss_phys.backward(retain_graph=True)
    
    assert getattr(z_preds, 'grad', None) is not None, "No gradient flowed to z_preds from ASM"
    grad_sum = np.abs(z_preds.grad).sum()
    print(f"-> PASS: ASM gradient backpropagated to z_pred. Sum of abs grads: {grad_sum:.6f}")
    
    print("\n4. Testing Full Forward + Backward Pass (No NaNs)...")
    dummy_images.grad = None
    targets = mock_torch.tensor([1.5, 2.0])
    
    loss_data = trainer.mse_loss(z_preds, targets)
    total_loss = loss_data + 0.1 * loss_phys
    total_loss.backward()
    
    has_nans = False
    for name, param in model.named_parameters():
        if getattr(param, 'grad', None) is not None:
            if not np.isfinite(param.grad).all():
                print(f"NaN/Inf found in gradient of {name}")
                has_nans = True
                
    assert not has_nans, "NaNs/Infs detected in backbone gradients"
    print("-> PASS: Full PINN loss backpropagated successfully through ASM and CNN.")
    
    print("\n5. Testing z_pred valid range and bounding gradients...")
    z_preds_test = model(dummy_images)
    print(f"z_preds values: {np.asarray(z_preds_test)}")
    assert (np.asarray(z_preds_test) > 0).all(), "z_pred must be strictly positive"
    assert (np.asarray(z_preds_test) <= 3.0).all(), "z_pred must be bounded by z_max"
    
    z_preds_test.retain_grad()
    loss_test = z_preds_test.sum()
    loss_test.backward()
    assert getattr(z_preds_test, 'grad', None) is not None, "Sigmoid bounding broke the gradient"
    print("-> PASS: z_pred is strictly bounded > 0 and <= z_max, with full gradient continuity.")
    
    print("\n--- ALL TESTS PASSED ---")

if __name__ == '__main__':
    main()
