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

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
import os
import gc

from diffusers import StableDiffusion3Pipeline
from diffusers.models import SD3Transformer2DModel

logger = logging.getLogger(__name__)


def load_sd35_pipeline(
    model_id: str = "stabilityai/stable-diffusion-3.5-medium",
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    enable_model_cpu_offload: bool = False,
) -> StableDiffusion3Pipeline:
    """
    Load the complete SD 3.5 Medium pipeline.
    
    Args:
        model_id: Hugging Face model identifier
        device: Device to load the model on
        dtype: Data type for model weights
        enable_model_cpu_offload: Enable CPU offloading for low VRAM
        
    Returns:
        StableDiffusion3Pipeline ready for inference
    """
    logger.info(f"Loading SD 3.5 Medium pipeline from {model_id}")
    logger.info(f"Device: {device}, dtype: {dtype}")
    
    # Load the pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        use_safetensors=True,
    )
    
    if enable_model_cpu_offload:
        logger.info("Enabling model CPU offload")
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)
    
    # Enable VAE slicing for memory efficiency (if available)
    if hasattr(pipe, 'enable_vae_slicing'):
        pipe.enable_vae_slicing()
        logger.info("VAE slicing enabled")
    else:
        logger.info("VAE slicing not available for this pipeline")
    
    logger.info("Pipeline loaded successfully")
    log_model_info(pipe)
    
    return pipe


def load_sd35_transformer(
    model_id: str = "stabilityai/stable-diffusion-3.5-medium",
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> SD3Transformer2DModel:
    """
    Load only the SD 3.5 transformer (MMDiT) component.
    
    This is useful for quantization where we only need to process the transformer.
    
    Args:
        model_id: Hugging Face model identifier
        device: Device to load the model on
        dtype: Data type for model weights
        
    Returns:
        SD3Transformer2DModel (the MMDiT model)
    """
    logger.info(f"Loading SD 3.5 transformer from {model_id}")
    
    transformer = SD3Transformer2DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        torch_dtype=dtype,
        use_safetensors=True,
    )
    
    transformer = transformer.to(device)
    transformer.eval()
    
    logger.info(f"Transformer loaded: {count_parameters(transformer) / 1e9:.2f}B parameters")
    
    return transformer


def get_transformer_layers(
    transformer: SD3Transformer2DModel,
    include_embeddings: bool = False,
) -> Dict[str, nn.Module]:
    """
    Get all quantizable layers from the SD 3.5 transformer.
    
    The MMDiT architecture contains:
    - Joint Transformer Blocks with:
        - Self-attention (Q, K, V projections + output projection)
        - Cross-attention for text conditioning
        - MLP/FFN blocks
        - AdaLN (adaptive layer norm) modulation layers
    
    Args:
        transformer: The SD3Transformer2DModel
        include_embeddings: Whether to include embedding layers
        
    Returns:
        Dictionary mapping layer names to nn.Linear modules
    """
    linear_layers = {}
    
    for name, module in transformer.named_modules():
        # Get all Linear layers
        if isinstance(module, nn.Linear):
            # Skip certain sensitive layers
            skip_patterns = ["time_text_embed", "context_embedder", "pos_embed"]
            
            should_skip = any(pattern in name for pattern in skip_patterns)
            
            if not include_embeddings and should_skip:
                logger.debug(f"Skipping embedding layer: {name}")
                continue
                
            linear_layers[name] = module
            
    logger.info(f"Found {len(linear_layers)} quantizable linear layers")
    
    return linear_layers


def get_transformer_blocks(transformer: SD3Transformer2DModel) -> List[nn.Module]:
    """
    Get the transformer blocks from MMDiT for block-wise quantization.
    
    Args:
        transformer: The SD3Transformer2DModel
        
    Returns:
        List of transformer blocks
    """
    # SD 3.5 Medium has transformer blocks in transformer.transformer_blocks
    if hasattr(transformer, 'transformer_blocks'):
        blocks = list(transformer.transformer_blocks)
        logger.info(f"Found {len(blocks)} transformer blocks")
        return blocks
    else:
        logger.warning("Could not find transformer_blocks attribute")
        return []


def get_linear_layers_in_block(block: nn.Module) -> Dict[str, nn.Linear]:
    """
    Get all Linear layers within a single transformer block.
    
    Args:
        block: A single transformer block
        
    Returns:
        Dictionary mapping relative names to Linear modules
    """
    linear_layers = {}
    
    for name, module in block.named_modules():
        if isinstance(module, nn.Linear):
            linear_layers[name] = module
            
    return linear_layers


def log_model_info(pipe: StableDiffusion3Pipeline) -> None:
    """
    Log detailed information about the loaded pipeline.
    
    Args:
        pipe: The loaded pipeline
    """
    logger.info("=" * 60)
    logger.info("SD 3.5 Medium Pipeline Information")
    logger.info("=" * 60)
    
    # Transformer info
    if hasattr(pipe, 'transformer') and pipe.transformer is not None:
        transformer = pipe.transformer
        logger.info(f"Transformer: {count_parameters(transformer) / 1e9:.2f}B parameters")
        logger.info(f"  - dtype: {next(transformer.parameters()).dtype}")
        logger.info(f"  - device: {next(transformer.parameters()).device}")
        
        if hasattr(transformer, 'transformer_blocks'):
            logger.info(f"  - num_blocks: {len(transformer.transformer_blocks)}")
    
    # Text encoder info
    if hasattr(pipe, 'text_encoder') and pipe.text_encoder is not None:
        logger.info(f"Text Encoder (CLIP-L): {count_parameters(pipe.text_encoder) / 1e6:.1f}M parameters")
        
    if hasattr(pipe, 'text_encoder_2') and pipe.text_encoder_2 is not None:
        logger.info(f"Text Encoder 2 (CLIP-G): {count_parameters(pipe.text_encoder_2) / 1e6:.1f}M parameters")
        
    if hasattr(pipe, 'text_encoder_3') and pipe.text_encoder_3 is not None:
        logger.info(f"Text Encoder 3 (T5-XXL): {count_parameters(pipe.text_encoder_3) / 1e9:.2f}B parameters")
    
    # VAE info
    if hasattr(pipe, 'vae') and pipe.vae is not None:
        logger.info(f"VAE: {count_parameters(pipe.vae) / 1e6:.1f}M parameters")
    
    logger.info("=" * 60)


def count_parameters(model: nn.Module) -> int:
    """
    Count the total number of parameters in a model.
    
    Args:
        model: PyTorch module
        
    Returns:
        Total number of parameters
    """
    return sum(p.numel() for p in model.parameters())


def get_model_size(model: nn.Module, unit: str = "GB") -> float:
    """
    Calculate the size of model parameters in memory.
    
    Args:
        model: PyTorch module
        unit: Size unit ("B", "KB", "MB", "GB")
        
    Returns:
        Size in specified unit
    """
    total_bytes = 0
    
    for param in model.parameters():
        # Calculate bytes based on dtype
        dtype_size = param.element_size()
        total_bytes += param.numel() * dtype_size
    
    divisors = {
        "B": 1,
        "KB": 1024,
        "MB": 1024 ** 2,
        "GB": 1024 ** 3,
    }
    
    return total_bytes / divisors.get(unit, 1)


def save_transformer_state(
    transformer: SD3Transformer2DModel,
    save_path: Path,
    save_format: str = "safetensors",
) -> None:
    """
    Save the transformer state dict.
    
    Args:
        transformer: The transformer model
        save_path: Path to save the model
        save_format: Format to save ("safetensors" or "pt")
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    if save_format == "safetensors":
        from safetensors.torch import save_file
        state_dict = {k: v.contiguous() for k, v in transformer.state_dict().items()}
        save_file(state_dict, save_path / "transformer.safetensors")
    else:
        torch.save(transformer.state_dict(), save_path / "transformer.pt")
    
    logger.info(f"Saved transformer to {save_path}")


def load_transformer_state(
    transformer: SD3Transformer2DModel,
    load_path: Path,
    load_format: str = "safetensors",
) -> SD3Transformer2DModel:
    """
    Load transformer state from a saved checkpoint.
    
    Args:
        transformer: The transformer model (architecture)
        load_path: Path to load from
        load_format: Format of saved model
        
    Returns:
        Transformer with loaded weights
    """
    load_path = Path(load_path)
    
    if load_format == "safetensors":
        from safetensors.torch import load_file
        state_dict = load_file(load_path / "transformer.safetensors")
    else:
        state_dict = torch.load(load_path / "transformer.pt")
    
    transformer.load_state_dict(state_dict)
    logger.info(f"Loaded transformer from {load_path}")
    
    return transformer


def replace_pipeline_transformer(
    pipe: StableDiffusion3Pipeline,
    new_transformer: SD3Transformer2DModel,
) -> StableDiffusion3Pipeline:
    """
    Replace the transformer in a pipeline with a new one (e.g., quantized).
    
    Args:
        pipe: The original pipeline
        new_transformer: The new transformer to use
        
    Returns:
        Pipeline with replaced transformer
    """
    # Clear old transformer
    old_transformer = pipe.transformer
    pipe.transformer = new_transformer
    
    # Clean up old transformer
    del old_transformer
    gc.collect()
    torch.cuda.empty_cache()
    
    logger.info("Replaced pipeline transformer")
    
    return pipe


def get_gpu_memory_info() -> Dict[str, float]:
    """
    Get current GPU memory usage information.
    
    Returns:
        Dictionary with memory stats in GB
    """
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    return {
        "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
        "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
        "max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "total_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
    }