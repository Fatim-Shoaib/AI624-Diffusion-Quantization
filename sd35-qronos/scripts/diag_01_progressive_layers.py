#!/usr/bin/env python3
"""
=============================================================================
HYPOTHESIS 1: Progressive Layer Quantization
=============================================================================

Test if black images appear when we quantize N layers together.
Start with 1 layer, add more progressively until we see degradation.

This will tell us:
- How many layers can be quantized before quality collapses
- If there's a specific "tipping point"
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from PIL import Image
import json
from tqdm import tqdm

from config import DEFAULT_VISUAL_PROMPTS
from models.sd35_loader import load_sd35_pipeline


def simple_quantize_layer(layer: nn.Linear, bits: int = 4):
    """Simple RTN quantization."""
    weight = layer.weight.data.float()
    max_val = weight.abs().amax(dim=-1, keepdim=True)
    scale = max_val / (2 ** (bits - 1) - 1)
    scale = torch.clamp(scale, min=1e-8)
    
    qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    q = torch.clamp(torch.round(weight / scale), qmin, qmax)
    layer.weight.data = (q * scale).to(layer.weight.dtype)


def generate_image(pipe, prompt, seed=42):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    with torch.no_grad():
        result = pipe(prompt=prompt, height=512, width=512, 
                     num_inference_steps=20, guidance_scale=4.5, generator=generator)
    return result.images[0]


def check_image_quality(image):
    pixels = list(image.getdata())
    mean_val = sum(sum(p) for p in pixels) / (len(pixels) * 3)
    return mean_val


def main():
    output_dir = Path("diagnosis_progressive")
    output_dir.mkdir(exist_ok=True)
    
    prompt = DEFAULT_VISUAL_PROMPTS[0]
    results = []
    
    # Get all linear layers
    print("Loading pipeline...")
    pipe = load_sd35_pipeline(device="cuda", dtype=torch.float16)
    
    linear_layers = []
    for name, module in pipe.transformer.named_modules():
        if isinstance(module, nn.Linear):
            linear_layers.append((name, module))
    
    print(f"Found {len(linear_layers)} linear layers")
    
    # Save original weights
    original_weights = {}
    for name, layer in linear_layers:
        original_weights[name] = layer.weight.data.clone()
    
    # Test: Quantize progressively more layers
    test_counts = [1, 5, 10, 20, 50, 100, 150, 200, 250, 300, 350, len(linear_layers)]
    
    for n_layers in test_counts:
        if n_layers > len(linear_layers):
            n_layers = len(linear_layers)
            
        print(f"\n=== Testing with {n_layers} quantized layers ===")
        
        # Reset all weights
        for name, layer in linear_layers:
            layer.weight.data = original_weights[name].clone()
        
        # Quantize first N layers
        for i in range(n_layers):
            name, layer = linear_layers[i]
            simple_quantize_layer(layer)
        
        # Generate image
        img = generate_image(pipe, prompt)
        quality = check_image_quality(img)
        
        img.save(output_dir / f"progressive_{n_layers:03d}_layers.png")
        
        result = {
            "n_layers": n_layers,
            "image_mean": quality,
            "is_black": quality < 10,
        }
        results.append(result)
        
        print(f"  Image mean: {quality:.1f}, Is black: {quality < 10}")
        
        if quality < 10:
            print(f"  ⚠️ BLACK IMAGE at {n_layers} layers!")
            break
    
    # Save results
    with open(output_dir / "progressive_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()