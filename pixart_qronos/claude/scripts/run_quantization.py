#!/usr/bin/env python3
"""
Main script for running Qronos-DiT quantization on PixArt-α.

This script provides a complete pipeline for:
1. Loading the PixArt model
2. Collecting calibration data from COCO
3. Quantizing the model using Qronos
4. Evaluating the quantized model
5. Saving results and visualizations

Usage:
    python run_quantization.py --help
    
    # Full quantization with evaluation
    python run_quantization.py \
        --model_id PixArt-alpha/PixArt-XL-2-512x512 \
        --bits 8 \
        --num_calibration_samples 256 \
        --coco_captions_path /path/to/captions_val2017.json \
        --output_dir ./output \
        --evaluate

    # Resume from checkpoint
    python run_quantization.py \
        --resume_from ./checkpoints/checkpoint_block_12 \
        --output_dir ./output
"""

import os
import sys
import argparse
import json
import gc
from pathlib import Path
from datetime import datetime

import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from qronos_dit import (
    PixArtQuantizer,
    Evaluator,
    CLIPScorer,
    load_coco_captions,
    get_default_eval_prompts,
    measure_peak_vram,
    reset_vram_stats,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Qronos-DiT: Quantize PixArt-α using Qronos algorithm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Model settings
    parser.add_argument(
        "--model_id",
        type=str,
        default="PixArt-alpha/PixArt-XL-2-512x512",
        help="HuggingFace model ID or local path",
    )
    
    # Quantization settings
    parser.add_argument(
        "--bits",
        type=int,
        default=8,
        help="Quantization bit-width (default: 8)",
    )
    parser.add_argument(
        "--sym",
        action="store_true",
        default=True,
        help="Use symmetric quantization (default: True)",
    )
    parser.add_argument(
        "--no_sym",
        action="store_false",
        dest="sym",
        help="Use asymmetric quantization",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=-1,
        help="Group size for quantization (-1 for per-channel, default: -1)",
    )
    parser.add_argument(
        "--blocksize",
        type=int,
        default=128,
        help="Block size for GPTQ algorithm (default: 128)",
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percentage damping for Hessian (default: 0.01)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1e-6,
        help="Qronos damping factor (default: 1e-6)",
    )
    
    # Calibration settings
    parser.add_argument(
        "--num_calibration_samples",
        type=int,
        default=256,
        help="Number of calibration samples (default: 256)",
    )
    parser.add_argument(
        "--coco_captions_path",
        type=str,
        default=None,
        help="Path to COCO captions JSON file (optional, uses default prompts if not provided)",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=20,
        help="Number of diffusion steps for calibration (default: 20)",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=4.5,
        help="Classifier-free guidance scale (default: 4.5)",
    )
    
    # Output settings
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Output directory (default: ./output)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="Checkpoint directory (default: ./checkpoints)",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=4,
        help="Save checkpoint every N blocks (default: 4)",
    )
    
    # Resume settings
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    
    # Evaluation settings
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation after quantization",
    )
    parser.add_argument(
        "--num_eval_samples",
        type=int,
        default=50,
        help="Number of samples for evaluation (default: 50)",
    )
    parser.add_argument(
        "--eval_seed",
        type=int,
        default=42,
        help="Random seed for evaluation (default: 42)",
    )
    
    # Other settings
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)",
    )
    parser.add_argument(
        "--skip_layers",
        type=str,
        nargs="+",
        default=["to_k", "to_v"],
        help="Layer patterns to skip quantization (default: to_k to_v)",
    )
    
    return parser.parse_args()


def download_coco_captions(output_path: str) -> str:
    """
    Download COCO captions if not present.
    
    Args:
        output_path: Path to save the captions file
    
    Returns:
        Path to the captions file
    """
    output_path = Path(output_path)
    captions_file = output_path / "captions_val2017.json"
    
    if captions_file.exists():
        return str(captions_file)
    
    print("Downloading COCO captions...")
    import urllib.request
    import zipfile
    
    url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    zip_path = output_path / "annotations.zip"
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        urllib.request.urlretrieve(url, zip_path)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_path)
        
        # Move file to expected location
        annotations_dir = output_path / "annotations"
        if annotations_dir.exists():
            for f in annotations_dir.glob("captions_*.json"):
                f.rename(output_path / f.name)
        
        zip_path.unlink()  # Remove zip file
        
        return str(captions_file)
    except Exception as e:
        print(f"Failed to download COCO captions: {e}")
        return None


def main():
    args = parse_args()
    
    print("=" * 70)
    print("Qronos-DiT: Quantization for PixArt-α Diffusion Models")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: {args.model_id}")
    print(f"Bits: {args.bits}")
    print(f"Symmetric: {args.sym}")
    print(f"Skip layers: {args.skip_layers}")
    print("=" * 70)
    
    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config = vars(args)
    config['start_time'] = datetime.now().isoformat()
    
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Get calibration prompts
    if args.coco_captions_path:
        prompts = load_coco_captions(args.coco_captions_path, args.num_calibration_samples)
    else:
        # Try to download COCO captions
        coco_path = download_coco_captions(output_dir / "coco")
        if coco_path:
            prompts = load_coco_captions(coco_path, args.num_calibration_samples)
        else:
            print("Using default prompts (COCO captions not available)")
            prompts = get_default_eval_prompts(args.num_calibration_samples)
    
    print(f"Loaded {len(prompts)} calibration prompts")
    
    # Initialize quantizer
    quantizer = PixArtQuantizer(
        model_id=args.model_id,
        bits=args.bits,
        sym=args.sym,
        group_size=args.group_size,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        skip_layers=args.skip_layers,
    )
    
    # Load model
    quantizer.load_model()
    
    # Keep reference to FP16 model for evaluation
    if args.evaluate:
        from copy import deepcopy
        from diffusers import PixArtAlphaPipeline
        
        print("Loading FP16 baseline for comparison...")
        fp16_pipe = PixArtAlphaPipeline.from_pretrained(
            args.model_id,
            torch_dtype=torch.float16,
        )
    
    # Reset VRAM stats
    reset_vram_stats()
    
    # Run quantization
    print("\n" + "=" * 70)
    print("STARTING QUANTIZATION")
    print("=" * 70)
    
    stats = quantizer.quantize_full_model(
        prompts=prompts,
        blocksize=args.blocksize,
        percdamp=args.percdamp,
        alpha=args.alpha,
        checkpoint_interval=args.checkpoint_interval,
        resume_from=args.resume_from,
    )
    
    # Save quantized model
    quantizer.save_quantized_model(output_dir / "quantized_model")
    
    # Measure VRAM
    quant_vram = measure_peak_vram()
    stats['peak_vram_gb'] = quant_vram
    
    # Save stats
    with open(output_dir / "quantization_stats.json", 'w') as f:
        # Convert non-serializable items
        stats_json = {
            'total_layers': stats['total_layers'],
            'total_loss': stats['total_loss'],
            'total_time': stats['total_time'],
            'peak_vram_gb': stats['peak_vram_gb'],
        }
        json.dump(stats_json, f, indent=2)
    
    print(f"\nQuantization completed!")
    print(f"Total layers: {stats['total_layers']}")
    print(f"Total loss: {stats['total_loss']:.4f}")
    print(f"Total time: {stats['total_time']:.2f}s")
    print(f"Peak VRAM: {quant_vram:.2f} GB")
    
    # Run evaluation
    if args.evaluate:
        print("\n" + "=" * 70)
        print("RUNNING EVALUATION")
        print("=" * 70)
        
        # Get evaluation prompts
        eval_prompts = get_default_eval_prompts(args.num_eval_samples)
        
        # Create evaluator
        evaluator = Evaluator(
            fp16_pipe=fp16_pipe,
            quantized_pipe=quantizer.get_pipeline(),
            device=args.device,
            output_dir=output_dir / "evaluation",
        )
        
        # Run evaluation
        eval_results = evaluator.evaluate(
            prompts=eval_prompts,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            seed=args.eval_seed,
        )
        
        # Update stats with evaluation results
        stats['evaluation'] = eval_results
    
    # Final summary
    print("\n" + "=" * 70)
    print("QUANTIZATION COMPLETE")
    print("=" * 70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output saved to: {output_dir}")
    print("=" * 70)
    
    return stats


if __name__ == "__main__":
    main()
