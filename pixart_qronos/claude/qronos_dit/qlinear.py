"""
Quantized Linear Layer wrapper for Qronos-DiT.

Provides a wrapper around nn.Linear that supports quantization.
"""
import torch
import torch.nn as nn
from typing import Optional
from copy import deepcopy

from .quant_utils import quantize_tensor


def find_linear_layers(module: nn.Module, name: str = '', layers_to_skip: list = None):
    """
    Recursively find all linear layers in a module.
    
    Args:
        module: The module to search
        name: Current name prefix
        layers_to_skip: List of layer name patterns to skip
    
    Returns:
        Dictionary mapping layer names to layer modules
    """
    if layers_to_skip is None:
        layers_to_skip = []
    
    res = {}
    
    for child_name, child in module.named_children():
        full_name = f"{name}.{child_name}" if name else child_name
        
        # Check if this layer should be skipped
        skip = False
        for skip_pattern in layers_to_skip:
            if skip_pattern in full_name:
                skip = True
                break
        
        if skip:
            continue
        
        if isinstance(child, nn.Linear):
            res[full_name] = child
        else:
            res.update(find_linear_layers(child, full_name, layers_to_skip))
    
    return res


class QLinearLayer(nn.Module):
    """
    Quantized Linear Layer wrapper.
    
    This wraps a standard nn.Linear layer and provides quantization functionality.
    The weight is stored as a buffer and can be quantized using various methods.
    """
    
    def __init__(
        self,
        original_layer: nn.Linear,
        bits: int = 8,
        sym: bool = True,
        group_size: int = -1,
        enable_quant: bool = True,
    ):
        super().__init__()
        
        self.bits = bits
        self.sym = sym
        self.group_size = group_size
        self.enable_quant = enable_quant
        
        # Store weight as buffer
        self.register_buffer('weight', original_layer.weight.data.clone())
        
        if original_layer.bias is not None:
            self.register_buffer('bias', original_layer.bias.data.clone())
        else:
            self.bias = None
        
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        
        self.quantized = False
    
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using stored weights."""
        return torch.nn.functional.linear(x, self.weight, self.bias)
    
    @torch.no_grad()
    def quantize_weights(self):
        """Quantize the weights using simple RTN quantization."""
        if not self.enable_quant or self.bits >= 16:
            return
        
        self.weight.data = quantize_tensor(
            self.weight.data,
            n_bits=self.bits,
            sym=self.sym,
            group_size=self.group_size,
        )
        self.quantized = True
    
    def extra_repr(self) -> str:
        return (
            f'in_features={self.in_features}, out_features={self.out_features}, '
            f'bits={self.bits}, sym={self.sym}, group_size={self.group_size}, '
            f'quantized={self.quantized}'
        )


def replace_linear_with_qlinear(
    module: nn.Module,
    bits: int = 8,
    sym: bool = True,
    group_size: int = -1,
    layers_to_skip: list = None,
    name: str = '',
) -> nn.Module:
    """
    Replace all nn.Linear layers with QLinearLayer.
    
    Args:
        module: Module to modify
        bits: Quantization bits
        sym: Symmetric quantization
        group_size: Group size for quantization
        layers_to_skip: List of layer name patterns to skip
        name: Current name prefix
    
    Returns:
        Modified module with QLinearLayer replacements
    """
    if layers_to_skip is None:
        layers_to_skip = []
    
    for child_name, child in module.named_children():
        full_name = f"{name}.{child_name}" if name else child_name
        
        # Check if this layer should be skipped
        skip = False
        for skip_pattern in layers_to_skip:
            if skip_pattern in full_name:
                skip = True
                break
        
        if skip:
            continue
        
        if isinstance(child, nn.Linear):
            q_layer = QLinearLayer(
                child,
                bits=bits,
                sym=sym,
                group_size=group_size,
                enable_quant=True,
            )
            setattr(module, child_name, q_layer)
        else:
            replace_linear_with_qlinear(
                child, bits, sym, group_size, layers_to_skip, full_name
            )
    
    return module


def get_named_linears(module: nn.Module, name: str = '') -> dict:
    """
    Get all named linear layers (both nn.Linear and QLinearLayer).
    
    Args:
        module: Module to search
        name: Current name prefix
    
    Returns:
        Dictionary mapping names to linear layers
    """
    res = {}
    
    for child_name, child in module.named_children():
        full_name = f"{name}.{child_name}" if name else child_name
        
        if isinstance(child, (nn.Linear, QLinearLayer)):
            res[full_name] = child
        else:
            res.update(get_named_linears(child, full_name))
    
    return res
