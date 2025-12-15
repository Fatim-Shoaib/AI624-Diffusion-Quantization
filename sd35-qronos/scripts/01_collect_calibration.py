#!/usr/bin/env python3
"""
=============================================================================
Step 1: Collect Calibration Data for Qronos
=============================================================================

This script collects the covariance matrices (H and G) required for Qronos
quantization of SD 3.5 Medium.

Key Differences from GPTQ Calibration:
- Qronos requires BOTH H (from quantized inputs) and G (cross-covariance)
- We collect across multiple timesteps to capture diffusion dynamics

Usage:
    python 01_collect_calibration.py --num-samples 256

Requirements:
    - RTX 4090 (24GB VRAM) recommended
    - ~30-60 minutes for 256 samples
"""

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    MODEL_CONFIG,
    QRONOS_CONFIG,
    CALIBRATION_DIR,
    DEFAULT_VISUAL_PROMPTS,
)
from models.sd35_loader import load_sd35_pipeline, get_model_size
from quantization.calibration import (
    CalibrationDataCollector,
    collect_calibration_for_diffusion,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


def get_calibration_prompts(num_prompts: int) -> list:
    """
    Get prompts for calibration.
    
    Uses the 50 visual inspection prompts plus additional diverse prompts
    for better coverage of the activation space.
    """
    prompts = DEFAULT_VISUAL_PROMPTS.copy()
    
    # Add additional prompts for better calibration coverage
    additional_prompts = [
        # Simple objects
        "a red apple on a white table",
        "a blue car parked on a street",
        "a green tree in a park",
        "a yellow flower in a garden",
        "a brown dog sitting on grass",
        
        # Complex scenes
        "a busy city intersection with pedestrians and cars",
        "a peaceful countryside landscape with rolling hills",
        "an underwater scene with colorful coral and fish",
        "a space station orbiting Earth",
        "a medieval castle on a cliff",
        
        # Different styles
        "abstract art with vibrant colors and geometric shapes",
        "a watercolor painting of a sunset over the ocean",
        "a pencil sketch of a human face",
        "a digital art piece in cyberpunk style",
        "an oil painting in impressionist style",
        
        # Various subjects
        "a cute golden retriever puppy playing",
        "a majestic eagle soaring through clouds",
        "a detailed macro shot of a butterfly wing",
        "a portrait of an elderly woman with wrinkles",
        "a newborn baby sleeping peacefully",
        
        # Technical/challenging
        "hands holding a crystal ball",
        "a mirror reflecting a complex scene",
        "text that says 'HELLO WORLD' on a sign",
        "multiple people at a dinner table",
        "a glass of water with light refraction",
        
        # More diverse content
        "a steampunk mechanical owl",
        "a neon-lit tokyo alley at night",
        "a viking ship on stormy seas",
        "a cozy cabin in snowy mountains",
        "a futuristic robot gardening",
        "an ancient egyptian temple at sunrise",
        "a jazz musician playing saxophone",
        "a colorful hot air balloon festival",
        "a deep sea anglerfish in darkness",
        "a fairy tale mushroom village",
    ]
    
    prompts.extend(additional_prompts)
    
    # Repeat if we need more
    if num_prompts > len(prompts):
        repeats = (num_prompts // len(prompts)) + 1
        prompts = prompts * repeats
    
    return prompts[:num_prompts]


def main():
    parser = argparse.ArgumentParser(description="Collect Qronos calibration data")
    parser.add_argument(
        "--num-samples", type=int, default=QRONOS_CONFIG.num_calibration_samples,
        help=f"Number of calibration samples (default: {QRONOS_CONFIG.num_calibration_samples})"
    )
    parser.add_argument(
        "--num-timesteps", type=int, default=QRONOS_CONFIG.num_timesteps_per_sample,
        help=f"Timesteps per sample (default: {QRONOS_CONFIG.num_timesteps_per_sample})"
    )
    parser.add_argument(
        "--timestep-strategy", type=str, default=QRONOS_CONFIG.timestep_strategy,
        choices=["uniform", "linear", "quadratic"],
        help="Timestep sampling strategy"
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(CALIBRATION_DIR),
        help="Output directory for calibration data"
    )
    parser.add_argument(
        "--model-id", type=str, default=MODEL_CONFIG.model_id,
        help="Hugging Face model ID"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Check CUDA
    if not torch.cuda.is_available():
        logger.error("CUDA not available! GPU required for calibration.")
        return 1
    
    device = "cuda"
    logger.info(f"GPU: {torch.cuda.get_device_name()}")
    logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Get prompts
    prompts = get_calibration_prompts(args.num_samples)
    logger.info(f"Using {len(prompts)} prompts for calibration")
    logger.info(f"Timesteps per sample: {args.num_timesteps}")
    logger.info(f"Total forward passes: ~{len(prompts) * args.num_timesteps * 2}")
    
    # Load pipeline
    logger.info("\nLoading SD 3.5 Medium pipeline...")
    pipe = load_sd35_pipeline(
        model_id=args.model_id,
        device=device,
        dtype=torch.float16,
    )
    
    # Collect calibration data
    logger.info("\nCollecting calibration data...")
    logger.info("This collects both H (Hessian) and G (cross-covariance) matrices for Qronos")
    
    start_time = time.time()
    
    collector = collect_calibration_for_diffusion(
        pipe=pipe,
        prompts=prompts,
        num_timesteps_per_sample=args.num_timesteps,
        timestep_strategy=args.timestep_strategy,
        num_inference_steps=MODEL_CONFIG.default_num_inference_steps,
        guidance_scale=MODEL_CONFIG.default_guidance_scale,
        height=MODEL_CONFIG.default_height,
        width=MODEL_CONFIG.default_width,
        seed=args.seed,
        device=device,
    )
    
    collection_time = time.time() - start_time
    logger.info(f"\nCalibration collection completed in {collection_time / 60:.1f} minutes")
    
    # Save calibration data
    output_dir = Path(args.output_dir)
    logger.info(f"\nSaving calibration data to {output_dir}...")
    collector.save_calibration_data(output_dir)
    
    # Save collection metadata
    metadata = {
        "num_samples": args.num_samples,
        "num_timesteps_per_sample": args.num_timesteps,
        "timestep_strategy": args.timestep_strategy,
        "collection_time_minutes": collection_time / 60,
        "model_id": args.model_id,
        "seed": args.seed,
        "num_inference_steps": MODEL_CONFIG.default_num_inference_steps,
        "guidance_scale": MODEL_CONFIG.default_guidance_scale,
        "height": MODEL_CONFIG.default_height,
        "width": MODEL_CONFIG.default_width,
    }
    
    with open(output_dir / "collection_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Calculate storage size
    total_size_mb = sum(
        f.stat().st_size for f in output_dir.glob("*.pt")
    ) / (1024 ** 2)
    logger.info(f"Total calibration data size: {total_size_mb / 1024:.2f} GB")
    
    # Cleanup
    del pipe, collector
    torch.cuda.empty_cache()
    gc.collect()
    
    logger.info("\nCalibration data collection complete!")
    logger.info(f"Output directory: {output_dir}")
    logger.info("\nNext step: Run 02_apply_qronos.py to apply quantization")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
