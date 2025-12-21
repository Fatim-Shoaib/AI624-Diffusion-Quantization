#!/usr/bin/env python3
"""
=============================================================================
HYPOTHESIS 5: Which Layer Groups Cause Issues?
=============================================================================

Test quantizing different groups of layers separately to identify
which specific layer types cause the most damage.

Layer groups:
1. Attention Q layers only
2. Attention K layers only  
3. Attention V layers only
4. Attention output layers only
5. FFN up-projection only
6. FFN down-projection only
7. Norm layers only
8. Embedding layers only

This will help us build a precise skip list.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import json
from PIL import Image
from tqdm import tqdm

from config import DEFAULT_VISUAL_PROMPTS
from models.sd35_loader import load_sd35_pipeline


def quantize_matching_layers(transformer, include_patterns, bits=4):
    """Quantize only layers matching the include patterns."""
    quantized = 0
    
    for name, module in transformer.named_modules():
        if isinstance(module, nn.Linear):
            # Check if this layer matches any include pattern
            matches = any(p in name for p in include_patterns)
            if not matches:
                continue
            
            weight = module.weight.data.float()
            max_val = weight.abs().amax(dim=-1, keepdim=True)
            scale = max_val / (2 ** (bits - 1) - 1)
            scale = torch.clamp(scale, min=1e-8)
            
            qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
            q = torch.clamp(torch.round(weight / scale), qmin, qmax)
            module.weight.data = (q * scale).to(module.weight.dtype)
            quantized += 1
    
    return quantized


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
    output_dir = Path("diagnosis_layer_groups")
    output_dir.mkdir(exist_ok=True)
    
    prompt = DEFAULT_VISUAL_PROMPTS[0]
    
    # Define layer groups to test
    layer_groups = {
        "attn_q": ["attn.to_q", "attn2.to_q", "add_q_proj"],
        "attn_k": ["attn.to_k", "attn2.to_k", "add_k_proj"],
        "attn_v": ["attn.to_v", "attn2.to_v", "add_v_proj"],
        "attn_out": ["attn.to_out", "attn2.to_out", "to_add_out"],
        "ffn_up": ["ff.net.0.proj", "ff_context.net.0.proj"],
        "ffn_down": ["ff.net.2", "ff_context.net.2"],
        "norm": ["norm1.linear", "norm1_context.linear"],
        "embed": ["time_text_embed", "context_embedder"],
    }
    
    results = {}
    
    print("="*60)
    print("LAYER GROUP ANALYSIS")
    print("="*60)
    
    # FP16 Baseline
    print("\n[0] FP16 Baseline...")
    pipe = load_sd35_pipeline(device="cuda", dtype=torch.float16)
    img = generate_image(pipe, prompt)
    img.save(output_dir / "00_fp16_baseline.png")
    results['fp16'] = check_image_quality(img)
    print(f"  Image mean: {results['fp16']:.1f}")
    del pipe
    torch.cuda.empty_cache()
    
    # Test each group
    for i, (group_name, patterns) in enumerate(layer_groups.items(), 1):
        print(f"\n[{i}] Testing {group_name} layers...")
        print(f"    Patterns: {patterns}")
        
        pipe = load_sd35_pipeline(device="cuda", dtype=torch.float16)
        n_quantized = quantize_matching_layers(pipe.transformer, patterns)
        print(f"    Quantized {n_quantized} layers")
        
        img = generate_image(pipe, prompt)
        img.save(output_dir / f"{i:02d}_{group_name}.png")
        
        quality = check_image_quality(img)
        results[group_name] = {
            "quality": quality,
            "n_quantized": n_quantized,
            "is_black": quality < 10,
        }
        
        status = "❌ BLACK" if quality < 10 else "✅ OK"
        print(f"    Image mean: {quality:.1f} {status}")
        
        del pipe
        torch.cuda.empty_cache()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Impact of Each Layer Group")
    print("="*60)
    print(f"{'Group':<15} {'Layers':<10} {'Quality':<10} {'Status':<10}")
    print("-"*45)
    
    problematic = []
    for group_name, data in results.items():
        if group_name == 'fp16':
            continue
        status = "BLACK" if data['is_black'] else "OK"
        print(f"{group_name:<15} {data['n_quantized']:<10} {data['quality']:<10.1f} {status:<10}")
        if data['is_black']:
            problematic.append(group_name)
    
    if problematic:
        print(f"\n⚠️ PROBLEMATIC GROUPS: {', '.join(problematic)}")
        print("   These layer groups should be skipped!")
    else:
        print("\n✅ No single layer group causes black images alone")
        print("   The issue is from combining multiple groups")
    
    with open(output_dir / "layer_group_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
