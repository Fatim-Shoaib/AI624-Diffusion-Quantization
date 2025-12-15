"""
=============================================================================
SD 3.5 Medium Model Loading Utilities
=============================================================================

This module provides utilities for loading and managing the SD 3.5 Medium model,
including functions for accessing transformer layers for quantization.

SD 3.5 Medium Architecture:
- Transformer: MMDiT (Multimodal Diffusion Transformer) with ~2.5B parameters
- Text Encoders: CLIP-L/14, CLIP-G/14, T5-XXL
- VAE: SD 3 VAE
"""

import gc
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn

from diffusers import StableDiffusion3Pipeline
from diffusers.models import SD3Transformer2DModel


def load_sd35_pipeline(
    model_id: str = "stabilityai/stable-diffusion-3.5-medium",
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    enable_model_cpu_offload: bool = False,
) -> StableDiffusion3Pipeline:
    """
    Load the complete SD 3.5 Medium pipeline.
    
    Args:
        model_id: Hugging Face model ID
        device: Device to load to
        dtype: Data type for model weights
        enable_model_cpu_offload: If True, enable CPU offloading for low VRAM
        
    Returns:
        StableDiffusion3Pipeline ready for inference
    """
    print(f"Loading SD 3.5 Medium from {model_id}...")
    
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
    )
    
    if enable_model_cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)
    
    # Enable memory optimizations
    pipe.enable_vae_slicing()
    
    print(f"Pipeline loaded successfully")
    print(f"  Transformer: {get_model_size(pipe.transformer):.2f} GB")
    print(f"  Device: {next(pipe.transformer.parameters()).device}")
    
    return pipe


def load_sd35_transformer(
    model_id: str = "stabilityai/stable-diffusion-3.5-medium",
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> SD3Transformer2DModel:
    """
    Load only the SD 3.5 Medium transformer (for quantization).
    
    Args:
        model_id: Hugging Face model ID
        device: Device to load to
        dtype: Data type for model weights
        
    Returns:
        SD3Transformer2DModel
    """
    print(f"Loading SD 3.5 Medium transformer from {model_id}...")
    
    transformer = SD3Transformer2DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        torch_dtype=dtype,
    ).to(device)
    
    transformer.eval()
    
    print(f"Transformer loaded: {get_model_size(transformer):.2f} GB")
    
    return transformer


def get_transformer_linear_layers(
    transformer: SD3Transformer2DModel,
    skip_layers: Optional[List[str]] = None,
) -> Dict[str, nn.Linear]:
    """
    Get all linear layers in the transformer for quantization.
    
    Args:
        transformer: The SD3Transformer2DModel
        skip_layers: List of layer name patterns to skip
        
    Returns:
        Dict mapping layer names to nn.Linear modules
    """
    skip_layers = skip_layers or ['time_embed', 'label_embed', 'proj_out', 'pos_embed']
    
    linear_layers = {}
    
    for name, module in transformer.named_modules():
        if isinstance(module, nn.Linear):
            # Skip specified layers
            if any(skip in name for skip in skip_layers):
                continue
            
            linear_layers[name] = module
    
    return linear_layers


def get_model_size(model: nn.Module, unit: str = "GB") -> float:
    """
    Calculate model size in memory.
    
    Args:
        model: PyTorch model
        unit: "GB", "MB", or "B"
        
    Returns:
        Model size in specified unit
    """
    total_bytes = 0
    for param in model.parameters():
        total_bytes += param.numel() * param.element_size()
    for buffer in model.buffers():
        total_bytes += buffer.numel() * buffer.element_size()
    
    if unit == "GB":
        return total_bytes / (1024 ** 3)
    elif unit == "MB":
        return total_bytes / (1024 ** 2)
    else:
        return total_bytes


def count_parameters(model: nn.Module) -> int:
    """Count total parameters in model."""
    return sum(p.numel() for p in model.parameters())


def count_linear_layers(model: nn.Module) -> int:
    """Count linear layers in model."""
    return sum(1 for m in model.modules() if isinstance(m, nn.Linear))


def save_quantized_transformer(
    transformer: SD3Transformer2DModel,
    output_dir: Path,
    save_config: bool = True,
    quantization_config: Optional[dict] = None,
):
    """
    Save a quantized transformer to disk.
    
    Args:
        transformer: The quantized transformer
        output_dir: Directory to save to
        save_config: Whether to save the model config
        quantization_config: Quantization configuration to save
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save state dict
    state_dict_path = output_dir / "transformer_state_dict.pt"
    torch.save(transformer.state_dict(), state_dict_path)
    print(f"Saved state dict to {state_dict_path}")
    
    # Save config
    if save_config:
        config_path = output_dir / "config.json"
        transformer.config.to_json_file(config_path)
        print(f"Saved config to {config_path}")
    
    # Save quantization config
    if quantization_config:
        quant_config_path = output_dir / "quantization_config.json"
        with open(quant_config_path, 'w') as f:
            json.dump(quantization_config, f, indent=2)
        print(f"Saved quantization config to {quant_config_path}")
    
    # Calculate and save size info
    size_info = {
        "total_parameters": count_parameters(transformer),
        "model_size_gb": get_model_size(transformer, "GB"),
        "num_linear_layers": count_linear_layers(transformer),
    }
    
    size_path = output_dir / "size_info.json"
    with open(size_path, 'w') as f:
        json.dump(size_info, f, indent=2)
    
    print(f"Quantized model size: {size_info['model_size_gb']:.2f} GB")


def load_quantized_transformer(
    quantized_dir: Path,
    model_id: str = "stabilityai/stable-diffusion-3.5-medium",
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> SD3Transformer2DModel:
    """
    Load a quantized transformer from disk.
    
    Args:
        quantized_dir: Directory containing quantized model
        model_id: Original model ID (for architecture)
        device: Device to load to
        dtype: Data type
        
    Returns:
        Loaded transformer with quantized weights
    """
    quantized_dir = Path(quantized_dir)
    
    # Load base architecture
    transformer = SD3Transformer2DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        torch_dtype=dtype,
    )
    
    # Load quantized state dict
    state_dict_path = quantized_dir / "transformer_state_dict.pt"
    state_dict = torch.load(state_dict_path, map_location=device)
    transformer.load_state_dict(state_dict)
    
    transformer = transformer.to(device)
    transformer.eval()
    
    print(f"Loaded quantized transformer from {quantized_dir}")
    print(f"Model size: {get_model_size(transformer):.2f} GB")
    
    return transformer


def replace_pipeline_transformer(
    pipe: StableDiffusion3Pipeline,
    transformer: SD3Transformer2DModel,
) -> StableDiffusion3Pipeline:
    """
    Replace the transformer in a pipeline with a quantized version.
    
    Args:
        pipe: Original pipeline
        transformer: New (quantized) transformer
        
    Returns:
        Pipeline with replaced transformer
    """
    # Get the device of the original transformer
    device = next(pipe.transformer.parameters()).device
    
    # Delete old transformer to free memory
    del pipe.transformer
    torch.cuda.empty_cache()
    gc.collect()
    
    # Replace with new transformer
    pipe.transformer = transformer.to(device)
    
    return pipe


def print_transformer_structure(transformer: SD3Transformer2DModel, max_depth: int = 3):
    """
    Print the structure of the transformer for debugging.
    
    Args:
        transformer: The transformer to inspect
        max_depth: Maximum depth to print
    """
    def _print_module(module: nn.Module, prefix: str = "", depth: int = 0):
        if depth > max_depth:
            return
        
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            if isinstance(child, nn.Linear):
                print(f"  {full_name}: Linear({child.in_features}, {child.out_features})")
            elif isinstance(child, nn.LayerNorm):
                print(f"  {full_name}: LayerNorm({child.normalized_shape})")
            elif isinstance(child, nn.Conv2d):
                print(f"  {full_name}: Conv2d({child.in_channels}, {child.out_channels})")
            else:
                child_type = type(child).__name__
                print(f"  {full_name}: {child_type}")
                _print_module(child, full_name, depth + 1)
    
    print(f"\nSD 3.5 Medium Transformer Structure:")
    print(f"  Total parameters: {count_parameters(transformer):,}")
    print(f"  Linear layers: {count_linear_layers(transformer)}")
    print(f"\nModule hierarchy:")
    _print_module(transformer)
