#!/usr/bin/env python3
"""
Quickstart script for Qronos-DiT quantization.

This is a simplified version for quick testing.
For full control, use scripts/run_quantization.py

Usage:
    python quickstart.py
"""

import os
import sys
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import gc
from datetime import datetime

# Configuration - Edit these values
CONFIG = {
    # Model
    "model_id": "PixArt-alpha/PixArt-XL-2-512x512",
    
    # Quantization
    "bits": 8,
    "symmetric": True,
    "group_size": -1,  # -1 for per-channel
    
    # Layers to SKIP quantization (K and V projections)
    "skip_layers": ["to_k", "to_v"],
    
    # Calibration
    "num_calibration_samples": 256,  # Number of prompts for calibration
    
    # Checkpointing
    "checkpoint_interval": 4,  # Save every N blocks
    "checkpoint_dir": "./checkpoints",
    
    # Output
    "output_dir": "./output",
    
    # Evaluation
    "run_evaluation": True,
    "num_eval_samples": 50,
}


def get_calibration_prompts(num_samples: int = 256):
    """Get calibration prompts."""
    # Default diverse prompts
    base_prompts = [
        "A photo of a cat sitting on a windowsill",
        "A beautiful sunset over the ocean",
        "A modern city skyline at night",
        "A peaceful forest path in autumn",
        "A delicious pizza on a wooden table",
        "A cute puppy playing in the park",
        "A mountain landscape with snow-capped peaks",
        "A colorful bouquet of flowers",
        "A vintage car parked on a street",
        "A cozy coffee shop interior",
        "A tropical beach with palm trees",
        "A starry night sky over a desert",
        "A bowl of fresh fruit",
        "A person reading a book",
        "A butterfly on a flower",
        "A rainy day in a city",
        "A beautiful garden with roses",
        "A modern living room",
        "A waterfall in a jungle",
        "A street market",
        "A hot air balloon over a valley",
        "A snowy cabin in the mountains",
        "A lighthouse on a rocky coast",
        "A field of sunflowers",
        "A medieval castle on a hill",
        "A Japanese garden with koi pond",
        "A coral reef underwater",
        "A steaming cup of coffee",
        "A bicycle by a canal",
        "A fireworks display over a city",
        "A peaceful lake at dawn",
        "A busy train station",
        "A wedding cake",
        "Elephants in the savanna",
        "A futuristic robot",
        "A cozy bedroom",
        "A plate of sushi",
        "Cherry blossoms in spring",
        "A vintage typewriter",
        "A beautiful waterfall",
    ]
    
    # Expand prompts with variations
    variations = [
        "A photo of {}",
        "An image showing {}",
        "A picture of {}",
        "{} in high quality",
        "{} with beautiful lighting",
        "Professional photo of {}",
    ]
    
    subjects = [
        "a golden retriever", "a tabby cat", "a red rose", "a blue car",
        "a green forest", "a white house", "a black bird", "a yellow sunflower",
        "a purple butterfly", "a brown horse", "mountains", "the ocean",
        "a rainbow", "clouds", "snow", "rain", "sunshine", "moonlight",
        "a river", "a lake", "a bridge", "a tower", "a garden", "a park",
        "a street", "a market", "a restaurant", "a cafe", "a library", "a museum",
    ]
    
    prompts = list(base_prompts)
    
    # Add variations
    import itertools
    for var, subj in itertools.product(variations, subjects):
        prompts.append(var.format(subj))
        if len(prompts) >= num_samples:
            break
    
    return prompts[:num_samples]


def main():
    print("=" * 70)
    print("Qronos-DiT Quickstart")
    print("=" * 70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: {CONFIG['model_id']}")
    print(f"Bits: {CONFIG['bits']}")
    print(f"Skip layers: {CONFIG['skip_layers']}")
    print("=" * 70)
    
    # Import after showing config (faster startup feedback)
    from qronos_dit import PixArtQuantizer, Evaluator, get_default_eval_prompts
    
    # Create output directories
    Path(CONFIG['output_dir']).mkdir(parents=True, exist_ok=True)
    Path(CONFIG['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    
    # Get calibration prompts
    print("\nPreparing calibration prompts...")
    prompts = get_calibration_prompts(CONFIG['num_calibration_samples'])
    print(f"Using {len(prompts)} calibration prompts")
    
    # Initialize quantizer
    print("\nInitializing quantizer...")
    quantizer = PixArtQuantizer(
        model_id=CONFIG['model_id'],
        bits=CONFIG['bits'],
        sym=CONFIG['symmetric'],
        group_size=CONFIG['group_size'],
        device="cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir=CONFIG['checkpoint_dir'],
        skip_layers=CONFIG['skip_layers'],
    )
    
    # Load model
    print("\nLoading model...")
    quantizer.load_model()
    
    # Save FP16 reference if doing evaluation
    if CONFIG['run_evaluation']:
        print("\nKeeping FP16 reference for evaluation...")
        from diffusers import PixArtAlphaPipeline
        fp16_pipe = PixArtAlphaPipeline.from_pretrained(
            CONFIG['model_id'],
            torch_dtype=torch.float16,
        )
    
    # Run quantization
    print("\n" + "=" * 70)
    print("STARTING QUANTIZATION")
    print("=" * 70)
    
    stats = quantizer.quantize_full_model(
        prompts=prompts,
        checkpoint_interval=CONFIG['checkpoint_interval'],
    )
    
    # Save quantized model
    output_path = Path(CONFIG['output_dir']) / "quantized_model"
    quantizer.save_quantized_model(output_path)
    
    print(f"\nQuantization complete!")
    print(f"  Layers quantized: {stats['total_layers']}")
    print(f"  Total loss: {stats['total_loss']:.4f}")
    print(f"  Total time: {stats['total_time']:.2f}s")
    
    # Run evaluation
    if CONFIG['run_evaluation']:
        print("\n" + "=" * 70)
        print("RUNNING EVALUATION")
        print("=" * 70)
        
        eval_prompts = get_default_eval_prompts(CONFIG['num_eval_samples'])
        
        evaluator = Evaluator(
            fp16_pipe=fp16_pipe,
            quantized_pipe=quantizer.get_pipeline(),
            device="cuda",
            output_dir=Path(CONFIG['output_dir']) / "evaluation",
        )
        
        eval_results = evaluator.evaluate(
            prompts=eval_prompts,
            num_inference_steps=20,
            guidance_scale=4.5,
            seed=42,
        )
    
    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)
    print(f"Results saved to: {CONFIG['output_dir']}")


if __name__ == "__main__":
    main()
