"""
=============================================================================
Quantized Linear Layer Implementation
=============================================================================

This module provides a quantized linear layer that stores weights in low-bit
format and performs quantized matrix multiplication.

For inference, we support two modes:
1. Simulated quantization (fake quantization): Weights are stored in FP16 but
   quantized/dequantized on the fly. Useful for measuring quality impact.
2. True quantization: Weights stored in int4/int8 format with custom kernels.
   Provides actual memory and speed benefits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def find_linear_layers(
    model: nn.Module,
    prefix: str = "",
    skip_patterns: Optional[list] = None,
) -> Dict[str, nn.Linear]:
    """
    Recursively find all nn.Linear layers in a model.
    
    Args:
        model: PyTorch model to search
        prefix: Current name prefix for recursion
        skip_patterns: List of patterns to skip (e.g., ["embed", "norm"])
        
    Returns:
        Dictionary mapping full layer names to nn.Linear modules
    """
    if skip_patterns is None:
        skip_patterns = []
    
    linear_layers = {}
    
    for name, module in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        
        # Check if should skip
        should_skip = any(pattern in full_name.lower() for pattern in skip_patterns)
        
        if isinstance(module, nn.Linear) and not should_skip:
            linear_layers[full_name] = module
        else:
            # Recurse into children
            linear_layers.update(
                find_linear_layers(module, full_name, skip_patterns)
            )
    
    return linear_layers


class QuantLinear(nn.Module):
    """
    Quantized Linear layer with support for low-bit weights.
    
    This layer can operate in two modes:
    1. Fake quantization: Weights stored in original dtype, quantized on-the-fly
    2. True quantization: Weights stored in packed low-bit format
    
    Attributes:
        in_features: Input dimension
        out_features: Output dimension
        bits: Number of bits for weight quantization
        group_size: Group size for group-wise quantization
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        bits: int = 4,
        group_size: int = 128,
        symmetric: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize quantized linear layer.
        
        Args:
            in_features: Size of input features
            out_features: Size of output features
            bias: Whether to include bias
            bits: Number of bits for quantization
            group_size: Group size for group-wise quantization
            symmetric: Use symmetric quantization
            device: Device for parameters
            dtype: Data type for computations
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size if group_size > 0 else in_features
        self.symmetric = symmetric
        
        # Calculate number of groups
        self.num_groups = (in_features + self.group_size - 1) // self.group_size
        
        # Quantization range
        if symmetric:
            self.qmin = -(2 ** (bits - 1))
            self.qmax = 2 ** (bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** bits - 1
        
        # Register buffers for quantized weights
        # For fake quantization, we store dequantized weights
        self.register_buffer(
            'weight',
            torch.zeros(out_features, in_features, device=device, dtype=dtype)
        )
        
        # Scale and zero-point for each group
        # Shape: [out_features, num_groups]
        self.register_buffer(
            'scales',
            torch.zeros(out_features, self.num_groups, device=device, dtype=dtype)
        )
        
        if not symmetric:
            self.register_buffer(
                'zeros',
                torch.zeros(out_features, self.num_groups, device=device, dtype=dtype)
            )
        else:
            self.zeros = None
        
        # Bias
        if bias:
            self.register_buffer(
                'bias',
                torch.zeros(out_features, device=device, dtype=dtype)
            )
        else:
            self.bias = None
        
        # Track if layer has been quantized
        self.quantized = False
    
    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        bits: int = 4,
        group_size: int = 128,
        symmetric: bool = True,
    ) -> "QuantLinear":
        """
        Create a QuantLinear from an existing nn.Linear.
        
        Args:
            linear: Source linear layer
            bits: Number of bits for quantization
            group_size: Group size for quantization
            symmetric: Use symmetric quantization
            
        Returns:
            New QuantLinear with copied weights (not yet quantized)
        """
        quant_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            bits=bits,
            group_size=group_size,
            symmetric=symmetric,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        
        # Copy original weights (will be quantized later)
        quant_linear.weight.copy_(linear.weight.data)
        
        if linear.bias is not None:
            quant_linear.bias.copy_(linear.bias.data)
        
        return quant_linear
    
    def quantize_weights(self) -> None:
        """
        Quantize the weights in-place.
        
        This computes optimal scales and zero-points, then stores
        the fake-quantized weights (quantized then dequantized).
        """
        if self.quantized:
            logger.warning("Layer already quantized, skipping")
            return
        
        W = self.weight.data.float()
        
        # Process each group
        for g in range(self.num_groups):
            start_idx = g * self.group_size
            end_idx = min((g + 1) * self.group_size, self.in_features)
            
            W_group = W[:, start_idx:end_idx]
            
            if self.symmetric:
                # Symmetric quantization: scale = max(|W|) / qmax
                max_val = W_group.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
                scale = max_val / self.qmax
                
                # Quantize and dequantize
                W_q = torch.clamp(
                    torch.round(W_group / scale),
                    self.qmin,
                    self.qmax
                )
                W_deq = W_q * scale
                
                self.scales[:, g] = scale.squeeze()
            else:
                # Asymmetric quantization
                min_val = W_group.amin(dim=1, keepdim=True)
                max_val = W_group.amax(dim=1, keepdim=True)
                
                scale = (max_val - min_val).clamp(min=1e-5) / self.qmax
                zero = torch.round(-min_val / scale).clamp(self.qmin, self.qmax)
                
                # Quantize and dequantize
                W_q = torch.clamp(
                    torch.round(W_group / scale) + zero,
                    self.qmin,
                    self.qmax
                )
                W_deq = (W_q - zero) * scale
                
                self.scales[:, g] = scale.squeeze()
                self.zeros[:, g] = zero.squeeze()
            
            # Store dequantized weights
            self.weight.data[:, start_idx:end_idx] = W_deq.to(self.weight.dtype)
        
        self.quantized = True
        logger.debug(f"Quantized layer: {self.out_features}x{self.in_features}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with quantized weights.
        
        Args:
            x: Input tensor [..., in_features]
            
        Returns:
            Output tensor [..., out_features]
        """
        return F.linear(x, self.weight, self.bias)
    
    def extra_repr(self) -> str:
        """String representation for printing."""
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"bits={self.bits}, "
            f"group_size={self.group_size}, "
            f"quantized={self.quantized}"
        )


def replace_linear_with_quantized(
    model: nn.Module,
    bits: int = 4,
    group_size: int = 128,
    symmetric: bool = True,
    skip_patterns: Optional[list] = None,
) -> Dict[str, QuantLinear]:
    """
    Replace all nn.Linear layers in a model with QuantLinear.
    
    Args:
        model: Model to modify (in-place)
        bits: Number of bits for quantization
        group_size: Group size for quantization
        symmetric: Use symmetric quantization
        skip_patterns: Patterns in layer names to skip
        
    Returns:
        Dictionary of replaced layers
    """
    if skip_patterns is None:
        skip_patterns = []
    
    replaced = {}
    
    def replace_module(parent: nn.Module, name: str, module: nn.Module):
        """Helper to replace a module in parent."""
        setattr(parent, name, module)
    
    def recursive_replace(module: nn.Module, prefix: str = ""):
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            
            if isinstance(child, nn.Linear):
                # Check if should skip
                should_skip = any(p in full_name.lower() for p in skip_patterns)
                
                if not should_skip:
                    quant_linear = QuantLinear.from_linear(
                        child,
                        bits=bits,
                        group_size=group_size,
                        symmetric=symmetric,
                    )
                    replace_module(module, name, quant_linear)
                    replaced[full_name] = quant_linear
                    logger.debug(f"Replaced {full_name}")
            else:
                recursive_replace(child, full_name)
    
    recursive_replace(model)
    logger.info(f"Replaced {len(replaced)} linear layers with QuantLinear")
    
    return replaced


def compute_layer_sensitivity(
    layer: nn.Linear,
    calibration_data: torch.Tensor,
    bits_range: list = [2, 3, 4, 8],
) -> Dict[int, float]:
    """
    Compute sensitivity of a layer to different quantization bit-widths.
    
    This helps identify layers that are particularly sensitive to quantization
    and might need higher precision.
    
    Args:
        layer: Linear layer to analyze
        calibration_data: Sample inputs [num_samples, in_features]
        bits_range: List of bit-widths to test
        
    Returns:
        Dictionary mapping bits to MSE loss
    """
    sensitivity = {}
    
    with torch.no_grad():
        # Get FP output as reference
        fp_output = layer(calibration_data)
        
        for bits in bits_range:
            # Create temporary quantized version
            quant_layer = QuantLinear.from_linear(
                layer,
                bits=bits,
                group_size=128,
                symmetric=True,
            )
            quant_layer.quantize_weights()
            
            # Get quantized output
            q_output = quant_layer(calibration_data)
            
            # Compute MSE
            mse = F.mse_loss(q_output, fp_output).item()
            sensitivity[bits] = mse
            
            del quant_layer
    
    return sensitivity
