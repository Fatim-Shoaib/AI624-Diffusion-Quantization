#!/usr/bin/env python3
"""
=============================================================================
Generate Sample Images for Visual Comparison
=============================================================================

This script generates sample images for visual inspection and side-by-side
comparison between FP16 baseline and quantized models.

Usage:
    # Generate samples with FP16 only
    python generate_samples.py --num-images 10
    
    # Generate comparison (FP16 vs Quantized)
    python generate_samples.py --compare --quantized-model ./quantized_model

    # Use custom prompts
    python generate_samples.py --prompts-file ./my_prompts.txt
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional
import gc

import torch
from PIL import Image
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    MODEL_CONFIG,
    OUTPUTS_DIR,
    PROMPTS_DIR,
    DEFAULT_VISUAL_PROMPTS,
)
from models.sd35_loader import (
    load_sd35_pipeline,
    load_transformer_state,
    replace_pipeline_transformer,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


def load_prompts(prompts_file: Optional[Path], num_images: int) -> List[str]:
    """Load prompts from file or use defaults."""
    if prompts_file and prompts_file.exists():
        with open(prompts_file) as f:
            prompts = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    else:
        prompts = DEFAULT_VISUAL_PROMPTS.copy()
    
    # Adjust to requested number
    if num_images < len(prompts):
        prompts = prompts[:num_images]
    elif num_images > len(prompts):
        repeats = num_images // len(prompts) + 1
        prompts = (prompts * repeats)[:num_images]
    
    return prompts


def generate_images(
    pipe,
    prompts: List[str],
    output_dir: Path,
    num_inference_steps: int = 28,
    guidance_scale: float = 4.5,
    seed: int = 42,
    device: str = "cuda",
) -> List[Image.Image]:
    """
    Generate images from prompts.
    
    Returns list of generated PIL Images.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    images = []
    generator = torch.Generator(device=device).manual_seed(seed)
    
    for i, prompt in enumerate(tqdm(prompts, desc="Generating")):
        generator.manual_seed(seed + i)
        
        with torch.no_grad():
            result = pipe(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
        
        image = result.images[0]
        images.append(image)
        
        # Save individual image
        image_path = output_dir / f"{i:04d}.png"
        image.save(image_path)
        
        # Save prompt
        prompt_path = output_dir / f"{i:04d}_prompt.txt"
        with open(prompt_path, 'w') as f:
            f.write(prompt)
    
    return images


def create_comparison_grid(
    fp16_images: List[Image.Image],
    quant_images: List[Image.Image],
    prompts: List[str],
    output_path: Path,
) -> None:
    """
    Create side-by-side comparison images.
    """
    output_path.mkdir(parents=True, exist_ok=True)
    
    for i, (fp16_img, quant_img, prompt) in enumerate(zip(fp16_images, quant_images, prompts)):
        # Create side-by-side comparison
        width = fp16_img.width + quant_img.width + 20  # 20px gap
        height = max(fp16_img.height, quant_img.height) + 80  # Space for labels
        
        comparison = Image.new('RGB', (width, height), 'white')
        
        # Paste images
        comparison.paste(fp16_img, (0, 60))
        comparison.paste(quant_img, (fp16_img.width + 20, 60))
        
        # Add labels (would need PIL ImageDraw for text, keeping simple for now)
        
        # Save comparison
        comparison_path = output_path / f"comparison_{i:04d}.png"
        comparison.save(comparison_path)
        
        # Save prompt
        prompt_path = output_path / f"comparison_{i:04d}_prompt.txt"
        with open(prompt_path, 'w') as f:
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Left: FP16 Baseline\n")
            f.write(f"Right: Quantized (W4A8)\n")
    
    logger.info(f"Saved {len(fp16_images)} comparison images to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate sample images for visual inspection",
    )
    
    # Generation settings
    parser.add_argument(
        "--num-images", type=int, default=10,
        help="Number of images to generate (default: 10)"
    )
    parser.add_argument(
        "--num-inference-steps", type=int, default=28,
        help="Inference steps (default: 28)"
    )
    parser.add_argument(
        "--guidance-scale", type=float, default=4.5,
        help="Guidance scale (default: 4.5)"
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
    parser.add_argument(
        "--quantized-model", type=str, default=None,
        help="Path to quantized model for comparison"
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Generate comparison between FP16 and quantized"
    )
    
    # Prompt settings
    parser.add_argument(
        "--prompts-file", type=str, default=None,
        help="Path to prompts file"
    )
    
    # Output settings
    parser.add_argument(
        "--output-dir", type=str, default=str(OUTPUTS_DIR / "samples"),
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Sample Image Generation")
    logger.info("=" * 60)
    
    # Check GPU
    if not torch.cuda.is_available():
        logger.error("CUDA not available!")
        return 1
    
    device = "cuda"
    logger.info(f"GPU: {torch.cuda.get_device_name()}")
    
    # Load prompts
    prompts_file = Path(args.prompts_file) if args.prompts_file else None
    prompts = load_prompts(prompts_file, args.num_images)
    logger.info(f"Using {len(prompts)} prompts")
    
    output_dir = Path(args.output_dir)
    
    # =========================================================================
    # Generate FP16 Baseline Images
    # =========================================================================
    logger.info("\nGenerating FP16 baseline images...")
    
    pipe = load_sd35_pipeline(
        model_id=args.model_id,
        device=device,
        dtype=torch.float16,
    )
    
    fp16_dir = output_dir / "fp16"
    fp16_images = generate_images(
        pipe=pipe,
        prompts=prompts,
        output_dir=fp16_dir,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        device=device,
    )
    
    logger.info(f"Saved FP16 images to {fp16_dir}")
    
    # =========================================================================
    # Generate Quantized Images (if requested)
    # =========================================================================
    quant_images = None
    
    if args.compare and args.quantized_model:
        logger.info("\nGenerating quantized model images...")
        
        quant_model_path = Path(args.quantized_model)
        if not quant_model_path.exists():
            logger.warning(f"Quantized model not found: {quant_model_path}")
            logger.warning("Skipping quantized comparison")
        else:
            # Load quantized transformer
            from models.sd35_loader import load_sd35_transformer
            
            quant_transformer = load_sd35_transformer(
                model_id=args.model_id,
                device=device,
                dtype=torch.float16,
            )
            quant_transformer = load_transformer_state(quant_transformer, quant_model_path)
            
            # Replace transformer in pipeline
            pipe = replace_pipeline_transformer(pipe, quant_transformer)
            
            quant_dir = output_dir / "quantized"
            quant_images = generate_images(
                pipe=pipe,
                prompts=prompts,
                output_dir=quant_dir,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed,
                device=device,
            )
            
            logger.info(f"Saved quantized images to {quant_dir}")
    
    # =========================================================================
    # Create Comparison Grid (if both available)
    # =========================================================================
    if quant_images and len(fp16_images) == len(quant_images):
        logger.info("\nCreating comparison images...")
        comparison_dir = output_dir / "comparison"
        create_comparison_grid(fp16_images, quant_images, prompts, comparison_dir)
    
    # Save prompts used
    prompts_output = output_dir / "prompts_used.txt"
    with open(prompts_output, 'w') as f:
        for i, p in enumerate(prompts):
            f.write(f"{i:04d}: {p}\n")
    
    logger.info(f"\nGeneration complete! Output: {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
