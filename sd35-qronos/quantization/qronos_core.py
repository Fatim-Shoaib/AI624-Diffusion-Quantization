"""
=============================================================================
Qronos Core Algorithm for Diffusion Models
=============================================================================

Implementation of the Qronos algorithm adapted from the Brevitas library
for use with Stable Diffusion 3.5 Medium's MMDiT architecture.

Reference:
- Paper: https://arxiv.org/abs/2505.11695
- Original Implementation: https://github.com/Xilinx/brevitas

Key Differences from LLM Implementation:
1. Calibration uses multiple timesteps (diffusion-specific)
2. Adapted for diffusers' SD3Transformer2DModel
3. Standalone implementation (no Brevitas dependency)
"""

import math
import warnings
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

try:
    from torch.linalg import LinAlgError
except ImportError:
    LinAlgError = RuntimeError


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _power_iteration(matrix: Tensor, num_iters: int = 30) -> float:
    """
    Estimate the largest singular value using power iteration.
    
    This is used for computing the dampening factor in Qronos,
    which differs from GPTQ's use of the average diagonal.
    
    Args:
        matrix: Square matrix to compute spectral norm of
        num_iters: Number of power iterations
        
    Returns:
        Estimated spectral norm (largest singular value)
    """
    n = matrix.shape[0]
    v = torch.randn(n, device=matrix.device, dtype=matrix.dtype)
    v = v / v.norm()
    
    for _ in range(num_iters):
        u = matrix @ v
        u_norm = u.norm()
        if u_norm > 0:
            u = u / u_norm
        v = matrix.T @ u
        v_norm = v.norm()
        if v_norm > 0:
            v = v / v_norm
    
    # Compute Rayleigh quotient
    sigma = (u @ matrix @ v).item()
    return abs(sigma)


def quantize_tensor(
    tensor: Tensor,
    scale: Tensor,
    zero_point: Tensor,
    num_bits: int,
    symmetric: bool = True,
) -> Tensor:
    """
    Quantize tensor to n-bit representation.
    
    Args:
        tensor: Input tensor
        scale: Quantization scale (per-channel or per-group)
        zero_point: Quantization zero point
        num_bits: Number of bits
        symmetric: If True, use symmetric quantization
        
    Returns:
        Quantized tensor (in dequantized form for fake quantization)
    """
    if symmetric:
        qmin = -(2 ** (num_bits - 1))
        qmax = 2 ** (num_bits - 1) - 1
    else:
        qmin = 0
        qmax = 2 ** num_bits - 1
    
    # Quantize
    q = torch.clamp(torch.round(tensor / scale) + zero_point, qmin, qmax)
    
    # Dequantize (fake quantization)
    return (q - zero_point) * scale


def compute_scale_zero_point(
    tensor: Tensor,
    num_bits: int,
    symmetric: bool = True,
    group_size: int = -1,
) -> Tuple[Tensor, Tensor]:
    """
    Compute quantization scale and zero point.
    
    Args:
        tensor: Weight tensor [out_features, in_features]
        num_bits: Number of bits
        symmetric: If True, use symmetric quantization
        group_size: Group size for per-group quantization (-1 for per-channel)
        
    Returns:
        Tuple of (scale, zero_point)
    """
    if group_size > 0:
        # Reshape for per-group quantization
        out_features, in_features = tensor.shape
        num_groups = in_features // group_size
        tensor_grouped = tensor.view(out_features, num_groups, group_size)
        
        if symmetric:
            max_val = tensor_grouped.abs().amax(dim=-1, keepdim=True)
            scale = max_val / (2 ** (num_bits - 1) - 1)
            scale = torch.clamp(scale, min=1e-8)
            zero_point = torch.zeros_like(scale)
        else:
            min_val = tensor_grouped.amin(dim=-1, keepdim=True)
            max_val = tensor_grouped.amax(dim=-1, keepdim=True)
            scale = (max_val - min_val) / (2 ** num_bits - 1)
            scale = torch.clamp(scale, min=1e-8)
            zero_point = torch.round(-min_val / scale)
        
        return scale, zero_point
    else:
        # Per-channel quantization
        if symmetric:
            max_val = tensor.abs().amax(dim=-1, keepdim=True)
            scale = max_val / (2 ** (num_bits - 1) - 1)
            scale = torch.clamp(scale, min=1e-8)
            zero_point = torch.zeros_like(scale)
        else:
            min_val = tensor.amin(dim=-1, keepdim=True)
            max_val = tensor.amax(dim=-1, keepdim=True)
            scale = (max_val - min_val) / (2 ** num_bits - 1)
            scale = torch.clamp(scale, min=1e-8)
            zero_point = torch.round(-min_val / scale)
        
        return scale, zero_point


# =============================================================================
# QRONOS ALGORITHM
# =============================================================================

class QronosQuantizer:
    """
    Qronos quantization algorithm for a single linear layer.
    
    This implements the core Qronos algorithm:
    1. Collect covariance matrices H (from quantized inputs) and G (cross-covariance)
    2. Apply two-phase quantization:
       - Phase 1: Special handling for first column
       - Phase 2: Cholesky-based updates for remaining columns
    """
    
    def __init__(
        self,
        layer: nn.Linear,
        layer_name: str,
        weight_bits: int = 4,
        group_size: int = 128,
        symmetric: bool = True,
        percdamp: float = 1e-5,
        num_blocks: int = 100,
        act_order: bool = False,
    ):
        """
        Initialize Qronos quantizer for a linear layer.
        
        Args:
            layer: The linear layer to quantize
            layer_name: Name of the layer (for logging)
            weight_bits: Number of bits for weight quantization
            group_size: Group size for quantization
            symmetric: Use symmetric quantization
            percdamp: Dampening factor (uses spectral norm)
            num_blocks: Number of sub-blocks for computation
            act_order: Whether to use activation ordering
        """
        self.layer = layer
        self.layer_name = layer_name
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.symmetric = symmetric
        self.percdamp = percdamp
        self.act_order = act_order
        
        # Get weight dimensions
        self.out_features = layer.out_features
        self.in_features = layer.in_features
        self.columns = self.in_features
        
        # Block size for efficient computation
        self.blocksize = math.ceil(self.columns / num_blocks)
        
        # Initialize covariance matrices (on CPU for memory efficiency)
        # H = X̃ᵀX̃ (Hessian from quantized inputs)
        self.H = torch.zeros(
            (self.columns, self.columns),
            dtype=torch.float32,
            device='cpu',
            pin_memory=torch.cuda.is_available()
        )
        
        # G = X̃ᵀX (cross-covariance between quantized and float inputs)
        self.G = torch.zeros(
            (self.columns, self.columns),
            dtype=torch.float32,
            device='cpu',
            pin_memory=torch.cuda.is_available()
        )
        
        # Buffer for efficient GPU->CPU transfer
        self.B = torch.zeros(
            (self.columns, self.columns),
            dtype=torch.float32,
            device='cpu',
            pin_memory=torch.cuda.is_available()
        )
        
        # Sample counter
        self.nsamples = 0
        
        # Storage for quantized input (for computing G)
        self.quant_input = None
        
        # Store original weights
        self.weight_orig = layer.weight.data.clone()
    
    def update_covariance_quant(self, inp: Tensor):
        """
        Update H matrix with quantized input.
        
        This is called first for each batch.
        
        Args:
            inp: Quantized input activations [batch, seq, features] or [batch, features]
        """
        # Flatten to 2D
        if len(inp.shape) > 2:
            inp = inp.reshape(-1, inp.shape[-1])
        
        batch_size = inp.shape[0]
        inp = inp.t().to(torch.float32)  # [features, batch]
        
        # Update H incrementally (numerically stable)
        self.nsamples += batch_size
        self.B.copy_(inp.cpu() @ inp.t().cpu())
        self.H *= (self.nsamples - batch_size) / self.nsamples
        self.H += self.B / self.nsamples
        
        # Store quantized input for G computation
        self.quant_input = inp
    
    def update_covariance_float(self, inp: Tensor):
        """
        Update G matrix with float input.
        
        This is called second for each batch, after update_covariance_quant.
        
        Args:
            inp: Float input activations [batch, seq, features] or [batch, features]
        """
        assert self.quant_input is not None, "Must call update_covariance_quant first"
        
        # Flatten to 2D
        if len(inp.shape) > 2:
            inp = inp.reshape(-1, inp.shape[-1])
        
        inp = inp.t().to(torch.float32)  # [features, batch]
        
        # Update G: G = X̃ᵀX
        batch_size = inp.shape[1]
        self.B.copy_(self.quant_input.cpu() @ inp.t().cpu())
        self.G *= (self.nsamples - batch_size) / self.nsamples
        self.G += self.B / self.nsamples
        
        # Clear stored quantized input
        self.quant_input = None
    
    def quantize_weight(self, w: Tensor, perm: Tensor) -> Tensor:
        """
        Quantize a single weight column.
        
        Args:
            w: Weight column [out_features]
            perm: Permutation index
            
        Returns:
            Quantized weight column
        """
        # Get the full weight matrix for scale computation
        weight = self.layer.weight.data
        
        # Compute scale and zero point
        if self.group_size > 0:
            # Per-group quantization
            scale, zp = compute_scale_zero_point(
                weight, self.weight_bits, self.symmetric, self.group_size
            )
            # Get the appropriate scale for this column
            group_idx = perm.item() // self.group_size
            col_scale = scale[:, group_idx:group_idx+1]
            col_zp = zp[:, group_idx:group_idx+1]
        else:
            # Per-channel quantization
            scale, zp = compute_scale_zero_point(
                weight, self.weight_bits, self.symmetric, -1
            )
            col_scale = scale
            col_zp = zp
        
        # Quantize the column
        return quantize_tensor(
            w.unsqueeze(1), col_scale, col_zp, 
            self.weight_bits, self.symmetric
        ).squeeze(1)
    
    def apply(self) -> nn.Linear:
        """
        Apply Qronos quantization to the layer.
        
        Returns:
            The quantized linear layer
        """
        del self.B  # Free memory
        
        weight = self.layer.weight.data.clone()
        weight_orig = self.weight_orig.clone()
        dev = weight.device
        dtype = weight.dtype
        
        # Convert to float32 for computation
        weight = weight.to(torch.float32)
        weight_orig = weight_orig.to(torch.float32)
        
        # Handle dead columns (zero diagonal in H)
        dead = self.H.diag() == 0
        weight[:, dead] = 0
        
        # Activation ordering (optional)
        if self.act_order:
            perm = torch.argsort(self.H.diag(), descending=True)
            self.H = self.H[perm, :][:, perm]
            self.G = self.G[perm, :][:, perm]
        else:
            perm = torch.arange(self.columns, device=dev)
        
        # Get diagonal elements
        Dh = self.H.diag().clone()
        Dhi = torch.where(Dh != 0, 1.0 / Dh, torch.zeros_like(Dh))  # D^{-1}
        Uh = torch.triu(self.H, 1)  # Upper triangular (for future weights)
        
        # Compute inverse Hessian with dampening
        iH = self.H.clone().to(dev)
        damp = self.percdamp * _power_iteration(self.H.to(dev), 30)
        diag_idx = torch.arange(self.columns, device=dev)
        iH[diag_idx, diag_idx] += damp
        
        try:
            iH = torch.linalg.cholesky(iH)
            iH = torch.cholesky_inverse(iH)
        except LinAlgError:
            warnings.warn(
                f"Failed to compute inverse Hessian for layer {self.layer_name}. "
                f"Falling back to RTN quantization."
            )
            # Fall back to simple RTN
            return self._apply_rtn()
        
        # =====================================================================
        # QRONOS PHASE 1: First Column (Special Handling)
        # =====================================================================
        
        # q_1 = Q((Gw - Uh*v) / H[0,0])
        Dhi = Dhi.to(dev)
        Uh = Uh.to(dev)
        self.G = self.G.to(dev)
        
        w = weight_orig[:, perm].to(dev)
        v = weight[:, perm].to(dev)
        
        Gw = w @ (self.G[:, 0] * Dhi[0])
        Uv = v @ (Uh[0, :] * Dhi[0])
        q_arg = Gw - Uv
        
        # Quantize first column
        q_first = self.quantize_weight(q_arg, perm[0]).to(dev)
        weight[:, perm[0]] = q_first.to(dtype)
        
        # Sherman-Morrison-Woodbury update for inverse Hessian
        # A = iH[1:,1:] - (b @ b.T) / c
        c = iH[0, 0]
        b = iH[1:, 0:1]
        A = iH[1:, 1:] - (b @ b.T) / c
        iH = A
        
        # Update weights for second column using least squares
        Ih = torch.diag(torch.full((self.columns,), damp, device=dev))
        Gh = self.G + Ih
        
        Gw_rest = w @ (Gh[:, 1:] @ iH)
        Hq = q_first.unsqueeze(1) @ (self.H[:1, 1:].to(dev) @ iH)
        weight[:, perm[1:]] = (Gw_rest - Hq.squeeze(0)).to(dtype)
        
        del self.G, self.H  # Memory management
        
        # =====================================================================
        # QRONOS PHASE 2: Cholesky-based Updates (t >= 2)
        # =====================================================================
        
        # Compute Cholesky decomposition of inverse Hessian
        c_factor = 1e4  # Stabilization constant
        try:
            L = torch.linalg.cholesky(iH * c_factor, upper=True) / math.sqrt(c_factor)
        except LinAlgError:
            warnings.warn(
                f"Failed Cholesky decomposition for layer {self.layer_name}. "
                f"Using fallback."
            )
            return self._finalize(weight, perm, dtype)
        
        del iH  # Memory management
        
        # Process remaining columns in blocks
        for i1 in range(1, self.columns, self.blocksize):
            i2 = min(i1 + self.blocksize, self.columns)
            count = i2 - i1
            
            error_block = torch.zeros(
                (self.out_features, count), 
                dtype=torch.float32, 
                device=dev
            )
            
            # Adjust indices (decrement by 1 due to Sherman-Morrison update)
            h_inv_block = L[i1-1:i2-1, i1-1:i2-1]
            
            # Process columns within block
            for i in range(count):
                col_idx = i1 + i
                
                # Round to nearest
                w_col = weight[:, perm[col_idx]].to(torch.float32)
                q_col = self.quantize_weight(w_col, perm[col_idx]).to(dev)
                
                # Compute error
                d = h_inv_block[i, i]
                error = (w_col - q_col) / d
                error_block[:, i] = error
                
                # Update weights within block (error diffusion)
                weight[:, perm[col_idx:i2]] -= (
                    error.unsqueeze(1) @ h_inv_block[i, i:i2-i1].unsqueeze(0)
                ).to(dtype)
            
            # Update weights outside block
            weight[:, perm[i2:]] -= (
                error_block @ L[i1-1:i2-1, i2-1:]
            ).to(dtype)
        
        del L  # Memory management
        
        return self._finalize(weight, perm, dtype)
    
    def _finalize(self, weight: Tensor, perm: Tensor, dtype: torch.dtype) -> nn.Linear:
        """Finalize quantization and update the layer."""
        # Undo permutation if act_order was used
        if self.act_order:
            inv_perm = torch.argsort(perm)
            weight = weight[:, inv_perm]
        
        # Update layer weights
        self.layer.weight.data = weight.to(dtype)
        
        return self.layer
    
    def _apply_rtn(self) -> nn.Linear:
        """Fallback: Apply simple round-to-nearest quantization."""
        weight = self.layer.weight.data
        scale, zp = compute_scale_zero_point(
            weight, self.weight_bits, self.symmetric, self.group_size
        )
        
        q_weight = quantize_tensor(weight, scale, zp, self.weight_bits, self.symmetric)
        self.layer.weight.data = q_weight
        
        return self.layer


# =============================================================================
# QRONOS MODE CONTEXT MANAGER
# =============================================================================

class QronosMode:
    """
    Context manager for applying Qronos quantization to a model.
    
    This handles the two-pass forward (quantized then float) required
    for computing the G matrix.
    
    Usage:
        with QronosMode(model, layer_names) as qronos:
            for batch in calibration_loader:
                qronos.collect_batch(batch)
            qronos.apply_quantization()
    """
    
    def __init__(
        self,
        transformer: nn.Module,
        layer_configs: Dict[str, dict],
        weight_bits: int = 4,
        group_size: int = 128,
        symmetric: bool = True,
        percdamp: float = 1e-5,
        num_blocks: int = 100,
        act_order: bool = False,
    ):
        """
        Initialize Qronos mode.
        
        Args:
            transformer: The transformer module to quantize
            layer_configs: Dict mapping layer names to their parent modules
            weight_bits: Number of bits for weight quantization
            group_size: Group size for quantization
            symmetric: Use symmetric quantization
            percdamp: Dampening factor
            num_blocks: Number of sub-blocks
            act_order: Use activation ordering
        """
        self.transformer = transformer
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.symmetric = symmetric
        self.percdamp = percdamp
        self.num_blocks = num_blocks
        self.act_order = act_order
        
        # Initialize quantizers for each layer
        self.quantizers: Dict[str, QronosQuantizer] = {}
        self.hooks = []
        
        for name, module in transformer.named_modules():
            if isinstance(module, nn.Linear):
                if any(skip in name for skip in ['time_embed', 'label_embed', 'proj_out', 'pos_embed']):
                    continue
                    
                self.quantizers[name] = QronosQuantizer(
                    layer=module,
                    layer_name=name,
                    weight_bits=weight_bits,
                    group_size=group_size,
                    symmetric=symmetric,
                    percdamp=percdamp,
                    num_blocks=num_blocks,
                    act_order=act_order,
                )
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Remove any hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def collect_batch_quant(self, layer_name: str, inp: Tensor):
        """Collect quantized input for a layer."""
        if layer_name in self.quantizers:
            self.quantizers[layer_name].update_covariance_quant(inp)
    
    def collect_batch_float(self, layer_name: str, inp: Tensor):
        """Collect float input for a layer."""
        if layer_name in self.quantizers:
            self.quantizers[layer_name].update_covariance_float(inp)
    
    def apply_quantization(self) -> nn.Module:
        """
        Apply Qronos quantization to all layers.
        
        Returns:
            The quantized transformer
        """
        print(f"\nApplying Qronos quantization to {len(self.quantizers)} layers...")
        
        for name, quantizer in tqdm(self.quantizers.items(), desc="Quantizing"):
            try:
                quantizer.apply()
            except Exception as e:
                warnings.warn(f"Failed to quantize {name}: {e}")
        
        return self.transformer
