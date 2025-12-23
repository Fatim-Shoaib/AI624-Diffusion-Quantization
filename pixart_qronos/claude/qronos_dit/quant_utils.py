"""
Quantization utilities for Qronos-DiT.
Contains basic quantization functions and helpers.
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple
import math


def quantize_tensor(
    w: torch.Tensor,
    n_bits: int = 8,
    sym: bool = True,
    group_size: int = -1,
    clip_ratio: float = 1.0,
) -> torch.Tensor:
    """
    Quantize a tensor using uniform quantization.
    
    Args:
        w: Input tensor to quantize
        n_bits: Number of bits for quantization
        sym: Whether to use symmetric quantization
        group_size: Group size for quantization (-1 for per-channel)
        clip_ratio: Ratio to clip the range
    
    Returns:
        Quantized tensor (fake quantized - still in float)
    """
    if n_bits >= 16:
        return w
    
    saved_shape = w.shape
    w = w.squeeze()
    
    if not w.is_contiguous():
        w = w.contiguous()
    
    if group_size > 0:
        assert w.shape[-1] % group_size == 0
        w = w.reshape(-1, group_size)
    
    if w.dim() == 1:
        w = w.unsqueeze(0)
    
    assert w.dim() == 2, f"Weight format should be [num_groups, group_size], got {w.shape}"
    
    if sym:
        w_max = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
        q_max = (2 ** (n_bits - 1) - 1)
        q_min = -(2 ** (n_bits - 1))
        
        if clip_ratio < 1.0:
            w_max = w_max * clip_ratio
        
        scales = w_max / q_max
        base = torch.zeros_like(scales)
    else:
        w_max = w.amax(dim=-1, keepdim=True)
        w_min = w.amin(dim=-1, keepdim=True)
        
        q_max = (2 ** n_bits - 1)
        q_min = 0
        
        if clip_ratio < 1.0:
            w_max = w_max * clip_ratio
            w_min = w_min * clip_ratio
        
        scales = (w_max - w_min).clamp(min=1e-5) / q_max
        base = torch.round(-w_min / scales).clamp_(min=q_min, max=q_max)
    
    # Quantize and dequantize
    w = (torch.clamp(torch.round(w / scales) + base, q_min, q_max) - base) * scales
    
    return w.reshape(saved_shape)


def quantize_gptq(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    maxq: torch.Tensor,
    channel_group: int = 1,
) -> torch.Tensor:
    """
    GPTQ-style quantization function.
    
    Args:
        x: Input tensor to quantize (usually a weight column)
        scale: Quantization scale
        zero: Zero point
        maxq: Maximum quantization value
        channel_group: Number of channels grouped together
    
    Returns:
        Quantized tensor
    """
    if maxq < 0:
        return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
    
    shape = x.shape
    if channel_group > 1:
        assert len(shape) == 2, "Only support 2D input when using multiple channel group"
        x = x.reshape((int(x.shape[0] / channel_group), -1))
    
    # Uniform affine mapping
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    q = scale * (q - zero)
    
    return q.reshape(shape)


class Quantizer(nn.Module):
    """
    Quantizer class for managing quantization parameters.
    Similar to Q-DiT's Quantizer_GPTQ class.
    """
    
    def __init__(self, shape: int = 1):
        super().__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))
        self.channel_group = 1
        self.sym = True
        self.bits = 8
    
    def configure(
        self,
        bits: int,
        perchannel: bool = True,
        channel_group: int = 1,
        sym: bool = True,
        clip_ratio: float = 1.0,
    ):
        """Configure the quantizer parameters."""
        self.bits = bits
        self.maxq = torch.tensor(2 ** bits - 1)
        self.perchannel = perchannel
        self.channel_group = channel_group
        self.sym = sym
        self.clip_ratio = clip_ratio
    
    def find_params(self, x: torch.Tensor, weight: bool = False):
        """
        Find quantization parameters (scale and zero point) for input tensor.
        
        Args:
            x: Input tensor
            weight: Whether this is a weight tensor
        """
        dev = x.device
        self.maxq = self.maxq.to(dev)
        
        shape = x.shape
        if self.perchannel:
            if weight:
                x = x.flatten(1)
                if self.channel_group > 1:
                    x = x.reshape(int(shape[0] / self.channel_group), -1)
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3]).flatten(1)
                elif len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                elif len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)
        
        tmp = torch.zeros(x.shape[0], device=dev, dtype=x.dtype)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)
        
        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1
        
        if self.maxq < 0:
            self.scale = xmax
            self.zero = xmin
        else:
            self.scale = (xmax - xmin) * self.clip_ratio / self.maxq
            if self.sym:
                self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
            else:
                self.zero = torch.round(-xmin / self.scale)
        
        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)
        
        if weight:
            shape_out = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape_out)
            self.zero = self.zero.reshape(shape_out)
            return
        
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        elif len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1))
        elif len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)
    
    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize the input tensor using stored parameters."""
        if self.ready():
            return quantize_gptq(x, self.scale, self.zero, self.maxq, self.channel_group)
        return x
    
    def ready(self) -> bool:
        """Check if quantizer is ready (has valid scales)."""
        return torch.all(self.scale != 0)


def compute_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute mean squared error between predictions and targets."""
    return ((pred - target) ** 2).mean().item()


def power_iteration(H: torch.Tensor, num_iters: int = 30) -> float:
    """
    Estimate the maximum singular value of H using power iteration.
    Used for dampening in Qronos.
    
    Args:
        H: Input matrix (Hessian)
        num_iters: Number of power iteration steps
    
    Returns:
        Estimated maximum singular value
    """
    n = H.shape[0]
    v = torch.randn(n, 1, device=H.device, dtype=H.dtype)
    v = v / v.norm()
    
    for _ in range(num_iters):
        v = H @ v
        v = v / v.norm()
    
    # Rayleigh quotient
    sigma = (v.T @ H @ v).item()
    return abs(sigma)
