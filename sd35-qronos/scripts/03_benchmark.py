#!/usr/bin/env python3
"""
=============================================================================
Step 3: Benchmark Qronos-Quantized SD 3.5 Medium
=============================================================================

This script benchmarks the Qronos-quantized model and compares it against:
1. FP16 baseline (unquantized)
2. GPTQ baseline (if available)

Metrics:
- FID Score (image quality)
- CLIP Score (text-image alignment)
- Model Size (GB)
- Peak VRAM Usage (GB)
- Inference Time (seconds)
- Visual Inspection (saved images)

Usage:
    # Quick test (10 images, no FID)
    python 03_benchmark.py --num-images 10 --skip-fid
    
    # Full benchmark (5000 images with FID)
    python 03_benchmark.py --num-images 5000 --reference-path ./coco_val2017

Requirements:
    - Quantized model from 02_apply_qronos.py
    - COCO val2017 images for FID (optional)
"""

import argparse
import gc
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    MODEL_CONFIG,
    QRONOS_CONFIG,
    EVAL_CONFIG,
    QUANTIZED_MODEL_DIR,
    RESULTS_DIR,
    OUTPUTS_DIR,
    DEFAULT_VISUAL_PROMPTS,
)
from models.sd35_loader import (
    load_sd35_pipeline,
    load_quantized_transformer,
    replace_pipeline_transformer,
    get_model_size,
)
from evaluation.metrics import (
    VRAMTracker,
    InferenceTimer,
    CLIPScorer,
    FIDCalculator,
    generate_images_for_evaluation,
    run_full_evaluation,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


def load_reference_images(reference_path: Path, max_images: int = 5000) -> list:
    """Load reference images for FID calculation."""
    reference_path = Path(reference_path)
    
    if not reference_path.exists():
        logger.warning(f"Reference path not found: {reference_path}")
        return []
    
    # Get image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    image_files = [
        f for f in reference_path.iterdir()
        if f.suffix.lower() in image_extensions
    ]
    
    # Limit number of images
    image_files = sorted(image_files)[:max_images]
    
    logger.info(f"Loading {len(image_files)} reference images...")
    images = []
    for f in tqdm(image_files, desc="Loading reference images"):
        try:
            img = Image.open(f).convert("RGB")
            images.append(img)
        except Exception as e:
            logger.warning(f"Failed to load {f}: {e}")
    
    return images


def save_comparison_grid(
    images: dict,
    prompts: list,
    output_path: Path,
    images_per_row: int = 3,
):
    """
    Save a comparison grid of images from different models.
    
    Args:
        images: Dict mapping model names to list of images
        prompts: List of prompts
        output_path: Path to save grid
        images_per_row: Number of images per row
    """
    from PIL import ImageDraw, ImageFont
    
    model_names = list(images.keys())
    num_models = len(model_names)
    num_prompts = min(len(prompts), len(list(images.values())[0]))
    
    # Get image size from first image
    sample_img = list(images.values())[0][0]
    img_width, img_height = sample_img.size
    
    # Calculate grid size
    thumb_size = 512  # Thumbnail size
    padding = 10
    text_height = 60
    
    grid_width = (thumb_size + padding) * num_models + padding
    grid_height = (thumb_size + text_height + padding) * min(num_prompts, 10) + padding
    
    # Create grid
    grid = Image.new('RGB', (grid_width, grid_height), 'white')
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        font = ImageFont.load_default()
    
    draw = ImageDraw.Draw(grid)
    
    # Add images
    for prompt_idx in range(min(num_prompts, 10)):
        y = padding + prompt_idx * (thumb_size + text_height + padding)
        
        for model_idx, model_name in enumerate(model_names):
            x = padding + model_idx * (thumb_size + padding)
            
            # Resize image
            img = images[model_name][prompt_idx].copy()
            img.thumbnail((thumb_size, thumb_size))
            
            # Paste image
            grid.paste(img, (x, y + text_height))
            
            # Add model name
            draw.text((x, y), model_name, fill='black', font=font)
            
            # Add prompt (truncated)
            prompt_text = prompts[prompt_idx][:50] + "..." if len(prompts[prompt_idx]) > 50 else prompts[prompt_idx]
            draw.text((x, y + 20), prompt_text, fill='gray', font=font)
    
    grid.save(output_path)
    logger.info(f"Saved comparison grid to {output_path}")


def benchmark_model(
    pipe,
    model_name: str,
    prompts: list,
    reference_images: list = None,
    output_dir: Path = None,
    num_inference_steps: int = 28,
    guidance_scale: float = 4.5,
    seed: int = 42,
    compute_fid: bool = True,
    compute_clip: bool = True,
) -> dict:
    """
    Benchmark a single model.
    
    Returns:
        Dictionary of metrics
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Benchmarking: {model_name}")
    logger.info(f"{'='*60}")
    
    device = "cuda"
    
    # Get model size
    model_size_gb = get_model_size(pipe.transformer, "GB")
    logger.info(f"Transformer size: {model_size_gb:.2f} GB")
    
    # Run evaluation
    results, images = run_full_evaluation(
        pipe=pipe,
        prompts=prompts,
        reference_images=reference_images if compute_fid else None,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        device=device,
        compute_fid=compute_fid and reference_images is not None,
        compute_clip=compute_clip,
    )
    
    results["model_name"] = model_name
    results["model_size_gb"] = model_size_gb
    
    # Save individual images
    if output_dir:
        img_dir = output_dir / model_name.lower().replace(" ", "_")
        img_dir.mkdir(parents=True, exist_ok=True)
        
        for i, (img, prompt) in enumerate(zip(images, prompts)):
            # Create safe filename
            safe_prompt = "".join(c if c.isalnum() or c in " -_" else "_" for c in prompt[:50])
            img_path = img_dir / f"{i:04d}_{safe_prompt}.png"
            img.save(img_path)
        
        logger.info(f"Saved {len(images)} images to {img_dir}")
    
    # Print results
    logger.info(f"\nResults for {model_name}:")
    logger.info(f"  Model Size: {results['model_size_gb']:.2f} GB")
    logger.info(f"  Peak VRAM: {results.get('peak_vram_gb', 0):.2f} GB")
    logger.info(f"  Mean Inference Time: {results.get('mean_inference_time', 0):.2f}s")
    if results.get('mean_clip_score'):
        logger.info(f"  CLIP Score: {results['mean_clip_score']:.4f} ± {results.get('std_clip_score', 0):.4f}")
    if results.get('fid_score'):
        logger.info(f"  FID Score: {results['fid_score']:.2f}")
    
    return results, images


def main():
    parser = argparse.ArgumentParser(description="Benchmark Qronos-quantized model")
    parser.add_argument(
        "--num-images", type=int, default=50,
        help="Number of images to generate (default: 50)"
    )
    parser.add_argument(
        "--quantized-model", type=str, default=str(QUANTIZED_MODEL_DIR),
        help="Path to quantized model"
    )
    parser.add_argument(
        "--reference-path", type=str, default=None,
        help="Path to reference images for FID"
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(RESULTS_DIR),
        help="Output directory for results"
    )
    parser.add_argument(
        "--model-id", type=str, default=MODEL_CONFIG.model_id,
        help="Hugging Face model ID"
    )
    parser.add_argument(
        "--skip-fid", action="store_true",
        help="Skip FID calculation"
    )
    parser.add_argument(
        "--skip-clip", action="store_true",
        help="Skip CLIP score calculation"
    )
    parser.add_argument(
        "--skip-fp16", action="store_true",
        help="Skip FP16 baseline benchmark"
    )
    parser.add_argument(
        "--seed", type=int, default=EVAL_CONFIG.eval_seed,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Check CUDA
    if not torch.cuda.is_available():
        logger.error("CUDA not available!")
        return 1
    
    device = "cuda"
    logger.info(f"GPU: {torch.cuda.get_device_name()}")
    logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"benchmark_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Get prompts
    prompts = DEFAULT_VISUAL_PROMPTS[:args.num_images]
    if args.num_images > len(DEFAULT_VISUAL_PROMPTS):
        # Repeat prompts if needed
        repeats = (args.num_images // len(DEFAULT_VISUAL_PROMPTS)) + 1
        prompts = (DEFAULT_VISUAL_PROMPTS * repeats)[:args.num_images]
    
    logger.info(f"Using {len(prompts)} prompts")
    
    # Load reference images for FID
    reference_images = None
    if args.reference_path and not args.skip_fid:
        reference_images = load_reference_images(Path(args.reference_path), args.num_images)
        if not reference_images:
            logger.warning("No reference images loaded, FID will be skipped")
    
    all_results = {}
    all_images = {}
    
    # =========================================================================
    # Benchmark FP16 Baseline
    # =========================================================================
    
    if not args.skip_fp16:
        logger.info("\n" + "="*60)
        logger.info("Loading FP16 Baseline...")
        logger.info("="*60)
        
        pipe_fp16 = load_sd35_pipeline(
            model_id=args.model_id,
            device=device,
            dtype=torch.float16,
        )
        
        results_fp16, images_fp16 = benchmark_model(
            pipe=pipe_fp16,
            model_name="FP16 Baseline",
            prompts=prompts,
            reference_images=reference_images,
            output_dir=run_dir,
            num_inference_steps=MODEL_CONFIG.default_num_inference_steps,
            guidance_scale=MODEL_CONFIG.default_guidance_scale,
            seed=args.seed,
            compute_fid=not args.skip_fid,
            compute_clip=not args.skip_clip,
        )
        
        all_results["fp16"] = results_fp16
        all_images["FP16 Baseline"] = images_fp16
        
        # Cleanup
        del pipe_fp16
        torch.cuda.empty_cache()
        gc.collect()
    
    # =========================================================================
    # Benchmark Qronos Quantized Model
    # =========================================================================
    
    quantized_path = Path(args.quantized_model)
    if quantized_path.exists():
        logger.info("\n" + "="*60)
        logger.info("Loading Qronos-Quantized Model...")
        logger.info("="*60)
        
        # Load base pipeline
        pipe_qronos = load_sd35_pipeline(
            model_id=args.model_id,
            device=device,
            dtype=torch.float16,
        )
        
        # Load and replace with quantized transformer
        quantized_transformer = load_quantized_transformer(
            quantized_dir=quantized_path,
            model_id=args.model_id,
            device=device,
            dtype=torch.float16,
        )
        
        pipe_qronos = replace_pipeline_transformer(pipe_qronos, quantized_transformer)
        
        results_qronos, images_qronos = benchmark_model(
            pipe=pipe_qronos,
            model_name="Qronos W4A8",
            prompts=prompts,
            reference_images=reference_images,
            output_dir=run_dir,
            num_inference_steps=MODEL_CONFIG.default_num_inference_steps,
            guidance_scale=MODEL_CONFIG.default_guidance_scale,
            seed=args.seed,
            compute_fid=not args.skip_fid,
            compute_clip=not args.skip_clip,
        )
        
        all_results["qronos"] = results_qronos
        all_images["Qronos W4A8"] = images_qronos
        
        # Cleanup
        del pipe_qronos
        torch.cuda.empty_cache()
        gc.collect()
    else:
        logger.warning(f"Quantized model not found at {quantized_path}")
        logger.warning("Run 02_apply_qronos.py first to create quantized model")
    
    # =========================================================================
    # Save Results
    # =========================================================================
    
    # Save comparison grid
    if len(all_images) > 1:
        grid_path = run_dir / "comparison_grid.png"
        save_comparison_grid(all_images, prompts, grid_path)
    
    # Save results JSON
    results_path = run_dir / "benchmark_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {results_path}")
    
    # Print summary table
    logger.info("\n" + "="*60)
    logger.info("BENCHMARK SUMMARY")
    logger.info("="*60)
    
    headers = ["Metric", "FP16 Baseline", "Qronos W4A8", "Change"]
    rows = []
    
    fp16 = all_results.get("fp16", {})
    qronos = all_results.get("qronos", {})
    
    # Model Size
    fp16_size = fp16.get("model_size_gb", 0)
    qronos_size = qronos.get("model_size_gb", 0)
    if fp16_size and qronos_size:
        compression = fp16_size / qronos_size if qronos_size > 0 else 0
        rows.append(["Model Size (GB)", f"{fp16_size:.2f}", f"{qronos_size:.2f}", f"{compression:.1f}x"])
    
    # Peak VRAM
    fp16_vram = fp16.get("peak_vram_gb", 0)
    qronos_vram = qronos.get("peak_vram_gb", 0)
    if fp16_vram and qronos_vram:
        vram_reduction = (fp16_vram - qronos_vram) / fp16_vram * 100 if fp16_vram > 0 else 0
        rows.append(["Peak VRAM (GB)", f"{fp16_vram:.2f}", f"{qronos_vram:.2f}", f"-{vram_reduction:.1f}%"])
    
    # Inference Time
    fp16_time = fp16.get("mean_inference_time", 0)
    qronos_time = qronos.get("mean_inference_time", 0)
    if fp16_time and qronos_time:
        time_change = (qronos_time - fp16_time) / fp16_time * 100 if fp16_time > 0 else 0
        sign = "+" if time_change > 0 else ""
        rows.append(["Inference Time (s)", f"{fp16_time:.2f}", f"{qronos_time:.2f}", f"{sign}{time_change:.1f}%"])
    
    # CLIP Score
    fp16_clip = fp16.get("mean_clip_score")
    qronos_clip = qronos.get("mean_clip_score")
    if fp16_clip and qronos_clip:
        clip_change = (qronos_clip - fp16_clip) / fp16_clip * 100 if fp16_clip > 0 else 0
        sign = "+" if clip_change > 0 else ""
        rows.append(["CLIP Score", f"{fp16_clip:.4f}", f"{qronos_clip:.4f}", f"{sign}{clip_change:.2f}%"])
    
    # FID Score
    fp16_fid = fp16.get("fid_score")
    qronos_fid = qronos.get("fid_score")
    if fp16_fid and qronos_fid:
        fid_change = qronos_fid - fp16_fid
        sign = "+" if fid_change > 0 else ""
        rows.append(["FID Score", f"{fp16_fid:.2f}", f"{qronos_fid:.2f}", f"{sign}{fid_change:.2f}"])
    
    # Print table
    col_widths = [20, 15, 15, 12]
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    separator = "-" * len(header_line)
    
    logger.info(header_line)
    logger.info(separator)
    for row in rows:
        row_line = " | ".join(str(v).ljust(w) for v, w in zip(row, col_widths))
        logger.info(row_line)
    
    logger.info("\n" + "="*60)
    logger.info(f"Full results saved to: {run_dir}")
    logger.info("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
