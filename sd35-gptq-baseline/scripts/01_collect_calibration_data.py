#!/usr/bin/env python3
"""
=============================================================================
Calibration Data Collection for GPTQ (Optimized)
=============================================================================

This script collects calibration data needed for GPTQ quantization.
Instead of storing all activations (which uses massive memory), we compute
the Hessian matrix (H = X^T X) incrementally during inference.

This is mathematically equivalent to storing all activations and computing
H later, but uses O(d^2) memory per layer instead of O(n*d) where:
- d = layer input dimension
- n = number of samples (which can be millions with timesteps)

Usage:
    python 01_collect_calibration_data.py --num-samples 64 --output-dir ./calibration_data

For higher quality (takes longer):
    python 01_collect_calibration_data.py --num-samples 128 --num-inference-steps 28
"""

import argparse
import logging
import sys
import time
import json
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
    get_gpu_memory_info,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


class HessianCollector:
    """
    Collects Hessian matrices (H = X^T X) incrementally during forward passes.
    
    This is memory-efficient because we only store the d×d Hessian matrix
    per layer, not all the n×d activations.
    
    For GPTQ, we need H to compute the optimal quantization. Computing H
    incrementally is mathematically equivalent to storing all activations.
    """
    
    def __init__(self, device: torch.device = torch.device("cuda")):
        """
        Initialize Hessian collector.
        
        Args:
            device: Device to store Hessian matrices on
        """
        self.device = device
        self.hessians: Dict[str, torch.Tensor] = {}
        self.nsamples: Dict[str, int] = {}
        self.hooks = []
        self.enabled = True
        
    def _make_hook(self, name: str):
        """Create a forward hook that accumulates Hessian incrementally."""
        def hook(module, inp, out):
            if not self.enabled:
                return
                
            # Get input tensor
            x = inp[0] if isinstance(inp, tuple) else inp
            
            # Handle different input shapes
            if x.dim() == 2:
                # [batch, features]
                x = x.float()
            elif x.dim() == 3:
                # [batch, seq_len, features] -> flatten to [batch*seq_len, features]
                x = x.reshape(-1, x.shape[-1]).float()
            elif x.dim() == 4:
                # [batch, channels, h, w] -> [batch*h*w, channels]
                x = x.permute(0, 2, 3, 1).reshape(-1, x.shape[1]).float()
            else:
                return  # Skip unsupported shapes
            
            # Number of samples in this batch
            n = x.shape[0]
            
            # Initialize Hessian if first time
            if name not in self.hessians:
                d = x.shape[1]
                self.hessians[name] = torch.zeros((d, d), device=self.device, dtype=torch.float32)
                self.nsamples[name] = 0
            
            # Incremental Hessian update: H_new = (n_old * H_old + X^T X) / (n_old + n_new)
            # But for numerical stability, we accumulate X^T X and divide at the end
            x = x.to(self.device)
            self.hessians[name].add_(x.T @ x)
            self.nsamples[name] += n
                
        return hook
    
    def register_hooks(self, model: nn.Module, layer_names: Optional[List[str]] = None):
        """
        Register forward hooks on Linear layers.
        
        Args:
            model: Model to register hooks on
            layer_names: Optional list of specific layer names to hook (if None, hook all Linear)
        """
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Skip certain sensitive layers
                skip_patterns = ["time_text_embed", "context_embedder", "pos_embed", "proj_out"]
                should_skip = any(pattern in name for pattern in skip_patterns)
                
                if should_skip:
                    continue
                
                if layer_names is not None and name not in layer_names:
                    continue
                    
                hook = module.register_forward_hook(self._make_hook(name))
                self.hooks.append(hook)
                
        logger.info(f"Registered {len(self.hooks)} hooks for Hessian collection")
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_hessians(self) -> Dict[str, torch.Tensor]:
        """
        Get normalized Hessian matrices.
        
        Returns:
            Dictionary mapping layer names to normalized Hessian matrices
        """
        normalized = {}
        for name, H in self.hessians.items():
            n = self.nsamples[name]
            if n > 0:
                # Normalize by number of samples
                normalized[name] = H / n
            else:
                normalized[name] = H
        return normalized
    
    def clear(self):
        """Clear collected Hessians."""
        self.hessians = {}
        self.nsamples = {}


def get_diverse_prompts(num_prompts: int) -> List[str]:
    """
    Get a diverse set of prompts for calibration.
    
    Quality note: Using diverse prompts ensures the Hessian captures
    the full range of activations the model will see during inference.
    """
    # Start with visual inspection prompts (high quality, curated)
    prompts = DEFAULT_VISUAL_PROMPTS.copy()
    
    # Add more diverse prompts for better calibration coverage
    additional_prompts = [
        # Simple objects
        "a red apple on a white table",
        "a blue car parked on a street",
        "a green tree in a park",
        "a yellow flower in a garden",
        "a black cat sitting on a windowsill",
        "a white dog running in a field",
        
        # Complex scenes
        "a busy city intersection with pedestrians and cars",
        "a peaceful countryside landscape with rolling hills",
        "an underwater scene with colorful coral and fish",
        "a space station orbiting Earth with stars in background",
        "a medieval castle on a hilltop at sunset",
        "a modern kitchen with stainless steel appliances",
        
        # Different styles
        "abstract art with vibrant colors and geometric shapes",
        "a watercolor painting of a sunset over the ocean",
        "a pencil sketch of a human face with detailed shading",
        "a digital art piece in cyberpunk style with neon lights",
        "an oil painting in impressionist style of a garden",
        "a minimalist line drawing of a mountain landscape",
        
        # Various subjects
        "a cute golden retriever puppy playing with a ball",
        "a majestic eagle soaring through clouds",
        "a detailed macro shot of a butterfly wing",
        "a portrait of an elderly woman with kind eyes",
        "a bowl of fresh fruit on a wooden table",
        "a vintage car from the 1950s in cherry red",
        
        # Technical/challenging
        "human hands holding a glowing crystal",
        "a mirror reflecting a beautiful sunset",
        "text that says HELLO on a neon sign",
        "multiple people having dinner at a restaurant",
        "a glass of water with ice cubes and lemon",
        "a bookshelf filled with colorful books",
        
        # Nature
        "a waterfall in a tropical rainforest",
        "northern lights over a snowy mountain",
        "a field of sunflowers under blue sky",
        "ocean waves crashing on rocky shore",
        "a dense bamboo forest with morning mist",
        "autumn leaves falling in a park",
        
        # Architecture
        "a modern glass skyscraper reflecting clouds",
        "an ancient Roman colosseum at golden hour",
        "a cozy wooden cabin in snowy mountains",
        "a Japanese temple with cherry blossoms",
        "a futuristic city with flying vehicles",
        "a rustic farmhouse with a red barn",
    ]
    
    prompts.extend(additional_prompts)
    
    # Shuffle for variety (but deterministically for reproducibility)
    import random
    rng = random.Random(42)
    rng.shuffle(prompts)
    
    # Repeat if needed
    if num_prompts > len(prompts):
        repeats = num_prompts // len(prompts) + 1
        prompts = (prompts * repeats)
    
    return prompts[:num_prompts]


def save_hessians(
    hessians: Dict[str, torch.Tensor],
    nsamples: Dict[str, int],
    output_dir: Path,
    config: dict,
) -> None:
    """
    Save Hessian matrices to disk.
    
    Args:
        hessians: Dictionary of Hessian matrices
        nsamples: Dictionary of sample counts per layer
        output_dir: Output directory
        config: Configuration used for collection
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metadata
    metadata = {
        "num_layers": len(hessians),
        "config": config,
        "layers": {},
    }
    
    # Save each Hessian
    for name, H in tqdm(hessians.items(), desc="Saving Hessians"):
        # Create safe filename
        safe_name = name.replace(".", "_").replace("/", "_")
        
        # Save as float32 for precision during quantization
        H = H.cpu().float()
        
        tensor_path = output_dir / f"hessian_{safe_name}.pt"
        torch.save(H, tensor_path)
        
        metadata["layers"][name] = {
            "shape": list(H.shape),
            "file": str(tensor_path.name),
            "nsamples": nsamples.get(name, 0),
        }
    
    # Save metadata
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Calculate total size
    total_size_mb = sum(
        H.numel() * 4 / (1024 ** 2)  # float32 = 4 bytes
        for H in hessians.values()
    )
    
    logger.info(f"Saved Hessians for {len(hessians)} layers")
    logger.info(f"Total size: {total_size_mb:.1f} MB")
    logger.info(f"Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect calibration data (Hessians) for GPTQ quantization",
    )
    
    parser.add_argument(
        "--num-samples", type=int, default=64,
        help="Number of calibration prompts (default: 64, recommended: 64-128)"
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
    logger.info("Calibration Data Collection (Optimized)")
    logger.info("=" * 60)
    logger.info(f"Number of prompts: {args.num_samples}")
    logger.info(f"Inference steps: {args.num_inference_steps}")
    
    # Check GPU
    if not torch.cuda.is_available():
        logger.error("CUDA not available!")
        return 1
    
    device = "cuda"
    gpu_info = get_gpu_memory_info()
    logger.info(f"GPU: {torch.cuda.get_device_name()}")
    logger.info(f"VRAM: {gpu_info['total_gb']:.1f} GB")
    
    # Get prompts
    prompts = get_diverse_prompts(args.num_samples)
    logger.info(f"Using {len(prompts)} diverse prompts for calibration")
    
    # Load model
    logger.info("\nLoading SD 3.5 Medium...")
    pipe = load_sd35_pipeline(
        model_id=args.model_id,
        device=device,
        dtype=torch.float16,
    )
    
    # Initialize Hessian collector
    logger.info("\nCollecting Hessian matrices...")
    start_time = time.time()
    
    collector = HessianCollector(device=torch.device(device))
    collector.register_hooks(pipe.transformer)
    
    generator = torch.Generator(device=device).manual_seed(args.seed)
    
    # Estimate time
    estimated_time = len(prompts) * 8 / 60  # ~8 seconds per prompt
    logger.info(f"Estimated time: {estimated_time:.1f} minutes")
    
    for i, prompt in enumerate(tqdm(prompts, desc="Collecting calibration data")):
        generator.manual_seed(args.seed + i)
        
        with torch.no_grad():
            _ = pipe(
                prompt=prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
                output_type="latent",  # Skip VAE decode
            )
        
        # Log progress periodically
        if (i + 1) % 10 == 0:
            mem_gb = torch.cuda.memory_allocated() / (1024**3)
            elapsed = time.time() - start_time
            eta = elapsed / (i + 1) * (len(prompts) - i - 1)
            logger.info(f"  Progress: {i+1}/{len(prompts)}, "
                       f"GPU mem: {mem_gb:.1f} GB, "
                       f"ETA: {eta/60:.1f} min")
    
    collector.remove_hooks()
    
    collection_time = time.time() - start_time
    logger.info(f"\nCollection completed in {collection_time / 60:.1f} minutes")
    
    # Get normalized Hessians
    hessians = collector.get_hessians()
    logger.info(f"Collected Hessians for {len(hessians)} layers")
    
    # Free GPU memory
    del pipe
    torch.cuda.empty_cache()
    gc.collect()
    
    # Save calibration data
    logger.info("\nSaving calibration data...")
    output_dir = Path(args.output_dir)
    
    config = {
        "num_samples": args.num_samples,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "model_id": args.model_id,
        "seed": args.seed,
        "collection_time_minutes": collection_time / 60,
    }
    
    save_hessians(hessians, collector.nsamples, output_dir, config)
    
    logger.info("\n" + "=" * 60)
    logger.info("Calibration data collection complete!")
    logger.info("=" * 60)
    logger.info(f"Output: {output_dir}")
    logger.info(f"Layers: {len(hessians)}")
    logger.info(f"Time: {collection_time / 60:.1f} minutes")
    logger.info("\nNext step: Run 02_quantize_model.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())