#!/usr/bin/env python3
"""
=============================================================================
Quick Test: Compare Different Quantization Approaches
=============================================================================

This script quickly tests:
1. FP16 baseline
2. Simple RTN quantization (all layers)
3. Simple RTN with K/V skip
4. Our Qronos implementation

To identify if the issue is with Qronos specifically or quantization in general.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from PIL import Image

from config import DEFAULT_VISUAL_PROMPTS
from models.sd35_loader import load_sd35_pipeline


def simple_rtn_quantize(weight: torch.Tensor, bits: int = 4) -> torch.Tensor:
    """Simple per-channel RTN quantization."""
    weight_float = weight.float()
    max_val = weight_float.abs().amax(dim=-1, keepdim=True)
    scale = max_val / (2 ** (bits - 1) - 1)
    scale = torch.clamp(scale, min=1e-8)
    
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1
    q = torch.clamp(torch.round(weight_float / scale), qmin, qmax)
    
    return (q * scale).to(weight.dtype)


def quantize_transformer(transformer, skip_patterns=None, bits=4):
    """Quantize all linear layers except those matching skip patterns."""
    skip_patterns = skip_patterns or []
    quantized = 0
    skipped = 0
    
    for name, module in transformer.named_modules():
        if isinstance(module, nn.Linear):
            # Check if should skip
            should_skip = any(pattern in name for pattern in skip_patterns)
            
            if should_skip:
                skipped += 1
                continue
            
            # Quantize
            module.weight.data = simple_rtn_quantize(module.weight.data, bits)
            quantized += 1
    
    print(f"Quantized {quantized} layers, skipped {skipped}")
    return transformer


def generate_image(pipe, prompt, seed=42):
    """Generate a single image."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    with torch.no_grad():
        result = pipe(
            prompt=prompt,
            height=512,
            width=512,
            num_inference_steps=20,
            guidance_scale=4.5,
            generator=generator,
        )
    return result.images[0]


def check_image_quality(image: Image.Image):
    """Check if image is black/broken."""
    pixels = list(image.getdata())
    mean_val = sum(sum(p) for p in pixels) / (len(pixels) * 3)
    return mean_val


def main():
    print("="*60)
    print("QUICK QUANTIZATION DIAGNOSTIC")
    print("="*60)
    
    prompt = DEFAULT_VISUAL_PROMPTS[0]
    print(f"\nTest prompt: {prompt[:50]}...")
    
    output_dir = Path("quick_diagnostic")
    output_dir.mkdir(exist_ok=True)
    
    # Test 1: FP16 Baseline
    print("\n[1/4] FP16 Baseline...")
    pipe = load_sd35_pipeline(device="cuda", dtype=torch.float16)
    img = generate_image(pipe, prompt)
    img.save(output_dir / "1_fp16_baseline.png")
    quality = check_image_quality(img)
    print(f"  Mean pixel value: {quality:.1f}")
    del pipe
    torch.cuda.empty_cache()
    
    # Test 2: Simple RTN (ALL layers)
    print("\n[2/4] Simple RTN - ALL layers...")
    pipe = load_sd35_pipeline(device="cuda", dtype=torch.float16)
    quantize_transformer(pipe.transformer, skip_patterns=[], bits=4)
    img = generate_image(pipe, prompt)
    img.save(output_dir / "2_rtn_all_layers.png")
    quality = check_image_quality(img)
    print(f"  Mean pixel value: {quality:.1f}")
    del pipe
    torch.cuda.empty_cache()
    
    # Test 3: Simple RTN (skip K/V)
    print("\n[3/4] Simple RTN - Skip K/V...")
    pipe = load_sd35_pipeline(device="cuda", dtype=torch.float16)
    skip = ['to_k', 'to_v', 'add_k_proj', 'add_v_proj']
    quantize_transformer(pipe.transformer, skip_patterns=skip, bits=4)
    img = generate_image(pipe, prompt)
    img.save(output_dir / "3_rtn_skip_kv.png")
    quality = check_image_quality(img)
    print(f"  Mean pixel value: {quality:.1f}")
    del pipe
    torch.cuda.empty_cache()
    
    # Test 4: Simple RTN (skip K/V + embeddings + norms)
    print("\n[4/4] Simple RTN - Skip K/V + embeddings + norms...")
    pipe = load_sd35_pipeline(device="cuda", dtype=torch.float16)
    skip = [
        'to_k', 'to_v', 'add_k_proj', 'add_v_proj',  # K/V
        'time_embed', 'time_text_embed', 'context_embedder',  # Embeddings
        'norm', 'proj_out'  # Norms and output
    ]
    quantize_transformer(pipe.transformer, skip_patterns=skip, bits=4)
    img = generate_image(pipe, prompt)
    img.save(output_dir / "4_rtn_skip_sensitive.png")
    quality = check_image_quality(img)
    print(f"  Mean pixel value: {quality:.1f}")
    del pipe
    torch.cuda.empty_cache()
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Images saved to: {output_dir.absolute()}")
    print("\nCheck the images manually to see which approach works!")
    print("If ALL images are black, there's a fundamental issue.")
    print("If some work, we know which layers to skip.")


if __name__ == "__main__":
    main()