"""
=============================================================================
Quantization Utilities
=============================================================================

Core quantization functions for weights and activations.
Supports both symmetric and asymmetric quantization with various granularities.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@torch.no_grad()
def quantize_tensor(
    x: torch.Tensor,
    bits: int = 8,
    symmetric: bool = True,
    per_channel: bool = False,
    channel_dim: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Quantize a tensor and return quantized values with scale/zero.
    
    Args:
        x: Input tensor to quantize
        bits: Number of bits (e.g., 4, 8)
        symmetric: Use symmetric quantization
        per_channel: Use per-channel quantization
        channel_dim: Dimension for per-channel quantization
        
    Returns:
        Tuple of (quantized_tensor, scale, zero_point)
        For symmetric, zero_point is None
    """
    if symmetric:
        qmin = -(2 ** (bits - 1))
        qmax = 2 ** (bits - 1) - 1
    else:
        qmin = 0
        qmax = 2 ** bits - 1
    
    if per_channel:
        # Move channel dim to last, flatten others
        x_perm = x.movedim(channel_dim, -1)
        orig_shape = x_perm.shape
        x_flat = x_perm.reshape(-1, orig_shape[-1])
        
        if symmetric:
            max_val = x_flat.abs().amax(dim=0, keepdim=True).clamp(min=1e-5)
            scale = max_val / qmax
            x_q = torch.clamp(torch.round(x_flat / scale), qmin, qmax)
            x_deq = x_q * scale
            zero = None
        else:
            min_val = x_flat.amin(dim=0, keepdim=True)
            max_val = x_flat.amax(dim=0, keepdim=True)
            scale = (max_val - min_val).clamp(min=1e-5) / qmax
            zero = torch.round(-min_val / scale).clamp(qmin, qmax)
            x_q = torch.clamp(torch.round(x_flat / scale) + zero, qmin, qmax)
            x_deq = (x_q - zero) * scale
        
        # Reshape back
        x_deq = x_deq.reshape(orig_shape).movedim(-1, channel_dim)
        scale = scale.squeeze()
        if zero is not None:
            zero = zero.squeeze()
    else:
        # Per-tensor quantization
        if symmetric:
            max_val = x.abs().amax().clamp(min=1e-5)
            scale = max_val / qmax
            x_q = torch.clamp(torch.round(x / scale), qmin, qmax)
            x_deq = x_q * scale
            zero = None
        else:
            min_val = x.amin()
            max_val = x.amax()
            scale = (max_val - min_val).clamp(min=1e-5) / qmax
            zero = torch.round(-min_val / scale).clamp(qmin, qmax)
            x_q = torch.clamp(torch.round(x / scale) + zero, qmin, qmax)
            x_deq = (x_q - zero) * scale
    
    return x_deq, scale, zero


@torch.no_grad()
def quantize_tensor_per_channel(
    x: torch.Tensor,
    bits: int = 8,
    symmetric: bool = True,
    channel_dim: int = 0,
) -> torch.Tensor:
    """
    Quantize tensor with per-channel scaling and return dequantized result.
    
    This is a simplified version that returns only the dequantized tensor.
    
    Args:
        x: Input tensor
        bits: Number of bits
        symmetric: Use symmetric quantization  
        channel_dim: Channel dimension
        
    Returns:
        Fake-quantized (quantized then dequantized) tensor
    """
    x_deq, _, _ = quantize_tensor(
        x, bits=bits, symmetric=symmetric, 
        per_channel=True, channel_dim=channel_dim
    )
    return x_deq


@torch.no_grad()
def quantize_activation(
    x: torch.Tensor,
    bits: int = 8,
    symmetric: bool = False,
    per_token: bool = False,
) -> torch.Tensor:
    """
    Quantize activations (typically 8-bit).
    
    For activations, we typically use:
    - Asymmetric quantization (activations often have positive bias)
    - Per-token or per-tensor granularity
    
    Args:
        x: Activation tensor [..., hidden_dim]
        bits: Number of bits (typically 8)
        symmetric: Use symmetric quantization
        per_token: Use per-token quantization
        
    Returns:
        Fake-quantized activation tensor
    """
    if per_token:
        # Per-token: quantize along last dimension
        x_deq, _, _ = quantize_tensor(
            x, bits=bits, symmetric=symmetric,
            per_channel=True, channel_dim=-1
        )
    else:
        # Per-tensor
        x_deq, _, _ = quantize_tensor(
            x, bits=bits, symmetric=symmetric,
            per_channel=False
        )
    
    return x_deq


class ActivationQuantizer(nn.Module):
    """
    Module wrapper for activation quantization.
    
    Can be inserted between layers to quantize intermediate activations.
    Supports both static (calibrated) and dynamic quantization.
    """
    
    def __init__(
        self,
        bits: int = 8,
        symmetric: bool = False,
        per_token: bool = True,
        static: bool = False,
    ):
        """
        Initialize activation quantizer.
        
        Args:
            bits: Number of bits
            symmetric: Use symmetric quantization
            per_token: Use per-token quantization
            static: Use static (pre-calibrated) scales
        """
        super().__init__()
        
        self.bits = bits
        self.symmetric = symmetric
        self.per_token = per_token
        self.static = static
        
        # For static quantization
        self.register_buffer('scale', None)
        self.register_buffer('zero', None)
        self.calibrated = False
        
        # Quantization range
        if symmetric:
            self.qmin = -(2 ** (bits - 1))
            self.qmax = 2 ** (bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** bits - 1
    
    def calibrate(self, x: torch.Tensor) -> None:
        """
        Calibrate static quantization parameters from sample data.
        
        Args:
            x: Sample activation tensor for calibration
        """
        if not self.static:
            logger.warning("Calibration only needed for static quantization")
            return
        
        with torch.no_grad():
            if self.symmetric:
                max_val = x.abs().amax().clamp(min=1e-5)
                self.scale = max_val / self.qmax
                self.zero = None
            else:
                min_val = x.amin()
                max_val = x.amax()
                self.scale = (max_val - min_val).clamp(min=1e-5) / self.qmax
                self.zero = torch.round(-min_val / self.scale)
        
        self.calibrated = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Quantize activation tensor.
        
        Args:
            x: Input activation tensor
            
        Returns:
            Fake-quantized activation tensor
        """
        if self.static and self.calibrated:
            # Use pre-calibrated scales
            if self.symmetric:
                x_q = torch.clamp(torch.round(x / self.scale), self.qmin, self.qmax)
                return x_q * self.scale
            else:
                x_q = torch.clamp(
                    torch.round(x / self.scale) + self.zero,
                    self.qmin, self.qmax
                )
                return (x_q - self.zero) * self.scale
        else:
            # Dynamic quantization
            return quantize_activation(
                x,
                bits=self.bits,
                symmetric=self.symmetric,
                per_token=self.per_token,
            )
    
    def extra_repr(self) -> str:
        return (
            f"bits={self.bits}, symmetric={self.symmetric}, "
            f"per_token={self.per_token}, static={self.static}"
        )


class SmoothQuantScaler(nn.Module):
    """
    SmoothQuant-style scaling for activation-aware quantization.
    
    SmoothQuant migrates quantization difficulty from activations to weights
    by applying channel-wise scaling: Y = (X / s) @ (s * W)
    
    Reference: https://arxiv.org/abs/2211.10438
    """
    
    def __init__(self, num_features: int, alpha: float = 0.5):
        """
        Initialize SmoothQuant scaler.
        
        Args:
            num_features: Number of features/channels
            alpha: Migration strength (0 = all to weights, 1 = all to activations)
        """
        super().__init__()
        
        self.alpha = alpha
        self.register_buffer(
            'scale',
            torch.ones(num_features)
        )
        self.calibrated = False
    
    def calibrate(
        self,
        activation_scales: torch.Tensor,
        weight_scales: torch.Tensor,
    ) -> None:
        """
        Compute smoothing scales from activation and weight statistics.
        
        Args:
            activation_scales: Per-channel activation max values
            weight_scales: Per-channel weight max values
        """
        # s = activation_scales^alpha / weight_scales^(1-alpha)
        self.scale = (
            activation_scales.pow(self.alpha) /
            weight_scales.pow(1 - self.alpha)
        ).clamp(min=1e-5)
        self.calibrated = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply smoothing to activations."""
        if self.calibrated:
            return x / self.scale
        return x
    
    def get_weight_scale(self) -> torch.Tensor:
        """Get scale to apply to weights."""
        return self.scale


def compute_activation_scales(
    model: nn.Module,
    calibration_loader,
    device: torch.device,
) -> dict:
    """
    Compute per-channel activation scales for all linear layers.
    
    Args:
        model: Model to analyze
        calibration_loader: DataLoader with calibration data
        device: Device for computation
        
    Returns:
        Dictionary mapping layer names to activation scales
    """
    activation_scales = {}
    hooks = []
    
    def make_hook(name):
        def hook(module, inp, out):
            x = inp[0]
            if x.dim() == 3:
                # [batch, seq, hidden] -> max over batch and seq
                scales = x.abs().amax(dim=(0, 1))
            elif x.dim() == 2:
                scales = x.abs().amax(dim=0)
            else:
                scales = x.abs().amax()
            
            if name in activation_scales:
                activation_scales[name] = torch.maximum(
                    activation_scales[name], scales
                )
            else:
                activation_scales[name] = scales
        return hook
    
    # Register hooks
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(make_hook(name)))
    
    # Run calibration
    model.eval()
    with torch.no_grad():
        for batch in calibration_loader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(device)
            model(batch)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return activation_scales
