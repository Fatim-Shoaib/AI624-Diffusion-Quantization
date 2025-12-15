#!/usr/bin/env python3
"""
=============================================================================
Calibration Data Collection for GPTQ
=============================================================================

This script collects calibration data (intermediate activations) needed for
GPTQ quantization. The calibration data captures the input distributions
to each layer during typical inference.

For SD 3.5 Medium (MMDiT), we need to collect:
- Inputs to transformer blocks at various timesteps
- Different prompts to capture diverse inputs

Usage:
    python 01_collect_calibration_data.py --num-samples 256 --output-dir ./calibration_data

Requirements:
    - RTX 4090 (24GB VRAM) recommended
    - ~10GB disk space for calibration data
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import gc

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    MODEL_CONFIG,
    CALIBRATION_DATA_DIR,
    PROMPTS_DIR,
    DEFAULT_VISUAL_PROMPTS,
)
from models.sd35_loader import (
    load_sd35_pipeline,
    get_transformer_blocks,
    get_gpu_memory_info,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


class ActivationCollector:
    """
    Collects intermediate activations during forward passes.
    
    Used to gather calibration data for GPTQ quantization.
    """
    
    def __init__(self, max_samples: int = 256):
        """
        Initialize activation collector.
        
        Args:
            max_samples: Maximum number of samples to collect per layer
        """
        self.max_samples = max_samples
        self.activations: Dict[str, List[torch.Tensor]] = {}
        self.hooks = []
        self.sample_count = 0
        
    def _make_hook(self, name: str):
        """Create a forward hook for a specific layer."""
        def hook(module, inp, out):
            if name not in self.activations:
                self.activations[name] = []
            
            # Get input tensor
            x = inp[0] if isinstance(inp, tuple) else inp
            
            # Only store if we haven't reached max samples
            if len(self.activations[name]) < self.max_samples:
                # Detach and move to CPU to save GPU memory
                self.activations[name].append(x.detach().cpu())
                
        return hook
    
    def register_hooks(self, model: nn.Module, layer_types: tuple = (nn.Linear,)):
        """
        Register forward hooks on specified layer types.
        
        Args:
            model: Model to register hooks on
            layer_types: Tuple of layer types to hook
        """
        for name, module in model.named_modules():
            if isinstance(module, layer_types):
                hook = module.register_forward_hook(self._make_hook(name))
                self.hooks.append(hook)
                
        logger.info(f"Registered {len(self.hooks)} hooks")
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_calibration_data(self) -> Dict[str, torch.Tensor]:
        """
        Get collected activations as concatenated tensors.
        
        Returns:
            Dictionary mapping layer names to activation tensors
        """
        calibration_data = {}
        
        for name, activations in self.activations.items():
            if activations:
                # Concatenate along batch dimension
                calibration_data[name] = torch.cat(activations, dim=0)
                
        return calibration_data
    
    def clear(self):
        """Clear collected activations."""
        self.activations = {}
        self.sample_count = 0


def get_diverse_prompts(num_prompts: int) -> List[str]:
    """
    Get a diverse set of prompts for calibration.
    
    Args:
        num_prompts: Number of prompts needed
        
    Returns:
        List of prompts
    """
    # Start with visual inspection prompts
    prompts = DEFAULT_VISUAL_PROMPTS.copy()
    
    # Add more diverse prompts for better calibration coverage
    additional_prompts = [
        # Simple objects
        "a red apple on a white table",
        "a blue car parked on a street",
        "a green tree in a park",
        "a yellow flower in a garden",
        
        # Complex scenes
        "a busy city intersection with pedestrians and cars",
        "a peaceful countryside landscape with rolling hills",
        "an underwater scene with colorful coral and fish",
        "a space station orbiting Earth",
        
        # Different styles
        "abstract art with vibrant colors and geometric shapes",
        "a watercolor painting of a sunset over the ocean",
        "a pencil sketch of a human face",
        "a digital art piece in cyberpunk style",
        
        # Various subjects
        "a cute golden retriever puppy playing",
        "a majestic eagle soaring through clouds",
        "a detailed macro shot of a butterfly wing",
        "a portrait of an elderly woman with wrinkles",
        
        # Technical/challenging
        "hands holding a crystal ball",
        "a mirror reflecting a complex scene",
        "text that says 'HELLO WORLD' on a sign",
        "multiple people at a dinner table",
    ]
    
    prompts.extend(additional_prompts)
    
    # Repeat if needed
    if num_prompts > len(prompts):
        repeats = num_prompts // len(prompts) + 1
        prompts = (prompts * repeats)
    
    return prompts[:num_prompts]


def collect_transformer_activations(
    pipe,
    prompts: List[str],
    num_inference_steps: int = 28,
    guidance_scale: float = 4.5,
    seed: int = 42,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """
    Collect activations from the transformer during generation.
    
    Args:
        pipe: SD3 pipeline
        prompts: List of prompts to run
        num_inference_steps: Inference steps
        guidance_scale: CFG scale
        seed: Random seed
        device: Device
        
    Returns:
        Dictionary mapping layer names to activation tensors
    """
    # Initialize collector
    collector = ActivationCollector(max_samples=len(prompts) * num_inference_steps)
    
    # Register hooks on transformer's linear layers
    collector.register_hooks(pipe.transformer, layer_types=(nn.Linear,))
    
    generator = torch.Generator(device=device).manual_seed(seed)
    
    logger.info(f"Collecting activations for {len(prompts)} prompts...")
    
    try:
        for i, prompt in enumerate(tqdm(prompts, desc="Collecting calibration data")):
            generator.manual_seed(seed + i)
            
            with torch.no_grad():
                _ = pipe(
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    output_type="latent",  # Skip VAE decode to save time
                )
            
            # Periodic memory cleanup
            if (i + 1) % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()
                
    finally:
        collector.remove_hooks()
    
    return collector.get_calibration_data()


def save_calibration_data(
    calibration_data: Dict[str, torch.Tensor],
    output_dir: Path,
    chunk_size_mb: int = 500,
) -> None:
    """
    Save calibration data to disk.
    
    Large tensors are saved in chunks to avoid memory issues.
    
    Args:
        calibration_data: Dictionary of activations
        output_dir: Output directory
        chunk_size_mb: Maximum chunk size in MB
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metadata
    metadata = {
        "num_layers": len(calibration_data),
        "layers": {},
    }
    
    for name, tensor in tqdm(calibration_data.items(), desc="Saving calibration data"):
        # Create safe filename
        safe_name = name.replace(".", "_").replace("/", "_")
        
        # Calculate tensor size
        tensor_size_mb = tensor.numel() * tensor.element_size() / (1024 ** 2)
        
        # Save tensor
        tensor_path = output_dir / f"{safe_name}.pt"
        torch.save(tensor, tensor_path)
        
        metadata["layers"][name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "size_mb": tensor_size_mb,
            "file": str(tensor_path.name),
        }
        
        logger.debug(f"Saved {name}: {tensor.shape}, {tensor_size_mb:.1f} MB")
    
    # Save metadata
    import json
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved calibration data for {len(calibration_data)} layers")
    logger.info(f"Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect calibration data for GPTQ quantization",
    )
    
    parser.add_argument(
        "--num-samples", type=int, default=256,
        help="Number of calibration samples (prompts) to collect (default: 256)"
    )
    parser.add_argument(
        "--num-inference-steps", type=int, default=28,
        help="Inference steps per sample (default: 28)"
    )
    parser.add_argument(
        "--guidance-scale", type=float, default=4.5,
        help="Guidance scale (default: 4.5)"
    )
    parser.add_argument(
        "--model-id", type=str,
        default="stabilityai/stable-diffusion-3.5-medium",
        help="Hugging Face model ID"
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(CALIBRATION_DATA_DIR),
        help="Output directory for calibration data"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Calibration Data Collection")
    logger.info("=" * 60)
    
    # Check GPU
    if not torch.cuda.is_available():
        logger.error("CUDA not available!")
        return 1
    
    device = "cuda"
    logger.info(f"GPU: {torch.cuda.get_device_name()}")
    logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Get prompts
    prompts = get_diverse_prompts(args.num_samples)
    logger.info(f"Using {len(prompts)} prompts for calibration")
    
    # Load model
    logger.info("\nLoading SD 3.5 Medium...")
    pipe = load_sd35_pipeline(
        model_id=args.model_id,
        device=device,
        dtype=torch.float16,
    )
    
    # Collect activations
    logger.info("\nCollecting activations...")
    start_time = time.time()
    
    calibration_data = collect_transformer_activations(
        pipe=pipe,
        prompts=prompts,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        device=device,
    )
    
    collection_time = time.time() - start_time
    logger.info(f"Collection completed in {collection_time / 60:.1f} minutes")
    
    # Free GPU memory
    del pipe
    torch.cuda.empty_cache()
    gc.collect()
    
    # Save calibration data
    logger.info("\nSaving calibration data...")
    output_dir = Path(args.output_dir)
    save_calibration_data(calibration_data, output_dir)
    
    # Calculate total size
    total_size_mb = sum(
        t.numel() * t.element_size() / (1024 ** 2) 
        for t in calibration_data.values()
    )
    logger.info(f"Total calibration data size: {total_size_mb / 1024:.2f} GB")
    
    logger.info("\nCalibration data collection complete!")
    logger.info(f"Output: {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
