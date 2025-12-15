#!/usr/bin/env python3
"""
=============================================================================
SD 3.5 Medium Baseline Benchmark Script
=============================================================================

This script runs the complete benchmark pipeline:
1. Load SD 3.5 Medium model
2. Generate images using provided prompts
3. Calculate all metrics (FID, CLIP, VRAM, timing)
4. Save results and comparison

Usage:
    python 03_benchmark.py --num-images 5000 --output-dir ./results

For quick testing:
    python 03_benchmark.py --num-images 50 --skip-fid

Requirements:
    - RTX 4090 (24GB VRAM) recommended
    - ~30GB disk space for generated images
    - Hugging Face login for SD 3.5 access
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    MODEL_CONFIG,
    EVAL_CONFIG,
    OUTPUTS_DIR,
    RESULTS_DIR,
    PROMPTS_DIR,
    DEFAULT_VISUAL_PROMPTS,
)
from models.sd35_loader import (
    load_sd35_pipeline,
    get_model_size,
    get_gpu_memory_info,
)
from evaluation.metrics import (
    BenchmarkMetrics,
    VRAMTracker,
    save_metrics,
    get_model_size_gb,
)
from evaluation.fid_score import FIDCalculator
from evaluation.clip_score import CLIPScoreCalculator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


def setup_prompts(args) -> List[str]:
    """
    Load or generate prompts for benchmarking.
    
    Args:
        args: Command line arguments
        
    Returns:
        List of prompts
    """
    prompts_file = Path(args.prompts_file) if args.prompts_file else None
    
    # Try to load from file
    if prompts_file and prompts_file.exists():
        logger.info(f"Loading prompts from {prompts_file}")
        with open(prompts_file) as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        # Use default visual inspection prompts
        logger.info("Using default visual inspection prompts")
        prompts = DEFAULT_VISUAL_PROMPTS.copy()
        
        # Save them for reference
        prompts_file = PROMPTS_DIR / "visual_inspection.txt"
        with open(prompts_file, 'w') as f:
            f.write('\n'.join(prompts))
        logger.info(f"Saved prompts to {prompts_file}")
    
    # Limit number of prompts
    if args.num_images < len(prompts):
        prompts = prompts[:args.num_images]
    elif args.num_images > len(prompts):
        # Repeat prompts if needed
        repeats = args.num_images // len(prompts) + 1
        prompts = (prompts * repeats)[:args.num_images]
    
    logger.info(f"Using {len(prompts)} prompts")
    return prompts


def generate_images(
    pipe,
    prompts: List[str],
    output_dir: Path,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    device: str,
    height: int = 1024,
    width: int = 1024,
) -> tuple:
    """
    Generate images and collect timing information.
    
    Returns:
        Tuple of (generated_images, inference_times, peak_vram_gb)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generated_images = []
    inference_times = []
    
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # Reset VRAM tracking
    torch.cuda.reset_peak_memory_stats()
    
    logger.info(f"Generating {len(prompts)} images...")
    logger.info(f"  Steps: {num_inference_steps}, Guidance: {guidance_scale}, Resolution: {height}x{width}")
    
    for i, prompt in enumerate(tqdm(prompts, desc="Generating images")):
        # Reset generator for reproducibility
        generator.manual_seed(seed + i)
        
        # Synchronize and time
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        with torch.no_grad():
            result = pipe(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                generator=generator,
            )
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        image = result.images[0]
        generated_images.append(image)
        inference_times.append(end_time - start_time)
        
        # Save image
        image_path = output_dir / f"{i:06d}.png"
        image.save(image_path)
        
        # Log progress periodically
        if (i + 1) % 10 == 0:
            avg_time = np.mean(inference_times[-10:])
            logger.info(f"  Generated {i+1}/{len(prompts)}, avg time: {avg_time:.2f}s")
    
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    
    return generated_images, inference_times, peak_vram_gb


def run_benchmark(args):
    """
    Run the complete benchmark pipeline.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Setup output directories
    run_dir = RESULTS_DIR / f"benchmark_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    images_dir = run_dir / "images"
    
    logger.info("=" * 60)
    logger.info("SD 3.5 Medium Baseline Benchmark")
    logger.info("=" * 60)
    logger.info(f"Output directory: {run_dir}")
    
    # Check GPU
    if not torch.cuda.is_available():
        logger.error("CUDA not available!")
        return
    
    device = "cuda"
    gpu_info = get_gpu_memory_info()
    logger.info(f"GPU: {torch.cuda.get_device_name()}")
    logger.info(f"Total VRAM: {gpu_info['total_gb']:.1f} GB")
    
    # Setup prompts
    prompts = setup_prompts(args)
    
    # Save prompts used
    with open(run_dir / "prompts_used.txt", 'w') as f:
        f.write('\n'.join(prompts))
    
    # =========================================================================
    # Load Model
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("Loading SD 3.5 Medium...")
    logger.info("=" * 60)
    
    load_start = time.time()
    
    pipe = load_sd35_pipeline(
        model_id=args.model_id,
        device=device,
        dtype=torch.float16,
    )
    
    load_time = time.time() - load_start
    logger.info(f"Model loaded in {load_time:.1f}s")
    
    # Get model size
    transformer_size_gb = get_model_size(pipe.transformer, "GB")
    logger.info(f"Transformer size: {transformer_size_gb:.2f} GB")
    
    # Initialize metrics
    metrics = BenchmarkMetrics(
        model_name="SD 3.5 Medium",
        quantization_config="FP16 (Baseline)",
        num_images=len(prompts),
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        resolution=f"{args.resolution}x{args.resolution}",
        transformer_size_gb=transformer_size_gb,
    )
    
    # =========================================================================
    # Generate Images
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("Generating Images...")
    logger.info("=" * 60)
    
    generated_images, inference_times, peak_vram = generate_images(
        pipe=pipe,
        prompts=prompts,
        output_dir=images_dir,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        device=device,
        height=args.resolution,
        width=args.resolution,
    )
    
    # Record timing metrics
    metrics.inference_times = inference_times
    metrics.inference_time_mean = float(np.mean(inference_times))
    metrics.inference_time_std = float(np.std(inference_times))
    metrics.images_per_second = 1.0 / metrics.inference_time_mean
    metrics.peak_vram_gb = peak_vram
    
    logger.info(f"\nGeneration complete:")
    logger.info(f"  Mean inference time: {metrics.inference_time_mean:.2f}s")
    logger.info(f"  Std inference time:  {metrics.inference_time_std:.2f}s")
    logger.info(f"  Peak VRAM:           {peak_vram:.2f} GB")
    
    # Free pipeline memory for evaluation
    del pipe
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    
    # =========================================================================
    # Calculate CLIP Score
    # =========================================================================
    if not args.skip_clip:
        logger.info("\n" + "=" * 60)
        logger.info("Calculating CLIP Score...")
        logger.info("=" * 60)
        
        clip_calc = CLIPScoreCalculator(
            model_name=args.clip_model,
            device=torch.device(device),
        )
        
        clip_mean, clip_scores = clip_calc.calculate_score_batch(
            generated_images,
            prompts,
            batch_size=32,
        )
        
        metrics.clip_score_mean = clip_mean
        metrics.clip_score_std = float(np.std(clip_scores))
        metrics.clip_scores = clip_scores
        
        logger.info(f"CLIP Score: {clip_mean:.4f} ± {metrics.clip_score_std:.4f}")
        
        # Cleanup
        del clip_calc
        torch.cuda.empty_cache()
        gc.collect()
    
    # =========================================================================
    # Calculate FID Score
    # =========================================================================
    if not args.skip_fid and args.reference_path:
        logger.info("\n" + "=" * 60)
        logger.info("Calculating FID Score...")
        logger.info("=" * 60)
        
        ref_path = Path(args.reference_path)
        if not ref_path.exists():
            logger.warning(f"Reference path not found: {ref_path}")
            logger.warning("Skipping FID calculation")
        else:
            fid_calc = FIDCalculator(
                device=torch.device(device),
                cache_dir=RESULTS_DIR / "fid_cache",
            )
            
            fid_score = fid_calc.calculate_fid(
                images_dir,
                ref_path,
                max_images=len(prompts),
            )
            
            metrics.fid_score = fid_score
            logger.info(f"FID Score: {fid_score:.2f}")
            
            # Cleanup
            del fid_calc
            torch.cuda.empty_cache()
            gc.collect()
    
    # =========================================================================
    # Save Results
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("Saving Results...")
    logger.info("=" * 60)
    
    # Save metrics
    metrics_path = run_dir / "metrics.json"
    save_metrics(metrics, metrics_path)
    
    # Save summary
    summary_path = run_dir / "summary.txt"
    with open(summary_path, 'w') as f:
        f.write(metrics.summary())
    
    # Save detailed timing
    timing_path = run_dir / "timing_details.json"
    with open(timing_path, 'w') as f:
        json.dump({
            "inference_times": inference_times,
            "mean": metrics.inference_time_mean,
            "std": metrics.inference_time_std,
        }, f, indent=2)
    
    # Save CLIP scores if available
    if metrics.clip_scores:
        clip_path = run_dir / "clip_scores.json"
        with open(clip_path, 'w') as f:
            json.dump({
                "prompts": prompts,
                "scores": metrics.clip_scores,
                "mean": metrics.clip_score_mean,
                "std": metrics.clip_score_std,
            }, f, indent=2)
    
    # Print final summary
    logger.info("\n" + metrics.summary())
    logger.info(f"\nResults saved to: {run_dir}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Run SD 3.5 Medium baseline benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (50 images, skip FID)
  python 03_benchmark.py --num-images 50 --skip-fid

  # Full benchmark (5000 images with FID)
  python 03_benchmark.py --num-images 5000 --reference-path ./coco_val2017

  # Custom settings
  python 03_benchmark.py --num-images 1000 --num-inference-steps 50 --guidance-scale 5.0
        """
    )
    
    # Generation settings
    parser.add_argument(
        "--num-images", type=int, default=50,
        help="Number of images to generate (default: 50)"
    )
    parser.add_argument(
        "--num-inference-steps", type=int, default=28,
        help="Number of inference steps (default: 28)"
    )
    parser.add_argument(
        "--guidance-scale", type=float, default=4.5,
        help="Guidance scale for CFG (default: 4.5)"
    )
    parser.add_argument(
        "--resolution", type=int, default=1024,
        help="Image resolution (default: 1024)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    
    # Model settings
    parser.add_argument(
        "--model-id", type=str,
        default="stabilityai/stable-diffusion-3.5-medium",
        help="Hugging Face model ID"
    )
    
    # Prompt settings
    parser.add_argument(
        "--prompts-file", type=str, default=None,
        help="Path to prompts file (one prompt per line)"
    )
    
    # Evaluation settings
    parser.add_argument(
        "--reference-path", type=str, default=None,
        help="Path to reference images for FID calculation"
    )
    parser.add_argument(
        "--clip-model", type=str, default="ViT-B-32",
        help="CLIP model for score calculation (default: ViT-B-32)"
    )
    parser.add_argument(
        "--skip-fid", action="store_true",
        help="Skip FID calculation"
    )
    parser.add_argument(
        "--skip-clip", action="store_true",
        help="Skip CLIP score calculation"
    )
    
    # Output settings
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: auto-generated)"
    )
    
    args = parser.parse_args()
    
    try:
        metrics = run_benchmark(args)
        return 0
    except Exception as e:
        logger.exception(f"Benchmark failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())