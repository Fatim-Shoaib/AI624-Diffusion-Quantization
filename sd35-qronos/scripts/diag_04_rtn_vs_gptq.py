#!/usr/bin/env python3
"""
=============================================================================
HYPOTHESIS 4: Our GPTQ Implementation vs Simple RTN
=============================================================================

Compare our GPTQ-style quantization (with error diffusion) against simple RTN.

If RTN works but our GPTQ doesn't, the bug is in our error diffusion logic.
If both fail similarly, the issue is elsewhere (e.g., layer selection, timesteps).

This test will:
1. Apply simple RTN to all layers → generate image
2. Apply our GPTQ to all layers → generate image
3. Compare the results
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import json
from PIL import Image

from config import DEFAULT_VISUAL_PROMPTS
from models.sd35_loader import load_sd35_pipeline


def apply_simple_rtn(transformer, skip_patterns=None, bits=4):
    """Apply simple RTN quantization (no error diffusion)."""
    skip_patterns = skip_patterns or []
    quantized = 0
    
    for name, module in transformer.named_modules():
        if isinstance(module, nn.Linear):
            if any(p in name for p in skip_patterns):
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


def apply_gptq_style(transformer, calibration_dir, skip_patterns=None, bits=4):
    """Apply GPTQ-style quantization with error diffusion using saved H matrices."""
    skip_patterns = skip_patterns or []
    calibration_dir = Path(calibration_dir)
    
    if not calibration_dir.exists():
        print("No calibration data found, falling back to RTN")
        return apply_simple_rtn(transformer, skip_patterns, bits)
    
    with open(calibration_dir / "metadata.json") as f:
        metadata = json.load(f)
    
    quantized = 0
    failed = 0
    
    for name, module in transformer.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        
        if any(p in name for p in skip_patterns):
            continue
        
        # Load H matrix
        layer_key = name
        if layer_key not in metadata['layers']:
            # Try alternative naming
            layer_key = name.replace('.', '_')
            if layer_key not in metadata['layers']:
                print(f"No H matrix for {name}, using RTN")
                # Apply simple RTN
                weight = module.weight.data.float()
                max_val = weight.abs().amax(dim=-1, keepdim=True)
                scale = max_val / 7
                scale = torch.clamp(scale, min=1e-8)
                q = torch.clamp(torch.round(weight / scale), -8, 7)
                module.weight.data = (q * scale).to(module.weight.dtype)
                quantized += 1
                continue
        
        try:
            H_path = calibration_dir / metadata['layers'][layer_key]['H_file']
            H = torch.load(H_path, map_location='cpu', weights_only=True)
            
            while H.dim() > 2:
                H = H.squeeze(-1)
            
            # Apply GPTQ-style quantization
            weight = module.weight.data.clone().float()
            d_out, d_in = weight.shape
            
            H = H.float()
            damp = 1e-5 * H.diag().mean()
            H_damped = H + damp * torch.eye(d_in)
            
            try:
                L = torch.linalg.cholesky(H_damped)
                H_inv = torch.cholesky_inverse(L)
            except:
                # Fall back to RTN
                max_val = weight.abs().amax(dim=-1, keepdim=True)
                scale = max_val / 7
                scale = torch.clamp(scale, min=1e-8)
                q = torch.clamp(torch.round(weight / scale), -8, 7)
                module.weight.data = (q * scale).to(module.weight.dtype)
                failed += 1
                continue
            
            # Column-by-column with error diffusion
            for col in range(d_in):
                w_col = weight[:, col].clone()
                
                # Quantize
                max_val = w_col.abs().max()
                if max_val > 0:
                    scale = max_val / 7
                    q_col = torch.clamp(torch.round(w_col / scale), -8, 7) * scale
                else:
                    q_col = w_col
                
                # Error diffusion
                error = (w_col - q_col) / (H_inv[col, col] + 1e-8)
                weight[:, col] = q_col
                
                if col < d_in - 1:
                    weight[:, col+1:] -= error.unsqueeze(1) * H_inv[col, col+1:].unsqueeze(0)
            
            module.weight.data = weight.to(module.weight.dtype)
            quantized += 1
            
        except Exception as e:
            print(f"Error on {name}: {e}")
            failed += 1
    
    print(f"Quantized: {quantized}, Failed: {failed}")
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
    output_dir = Path("diagnosis_rtn_vs_gptq")
    output_dir.mkdir(exist_ok=True)
    
    prompt = DEFAULT_VISUAL_PROMPTS[0]
    skip_patterns = ['to_k', 'to_v', 'add_k_proj', 'add_v_proj',
                     'time_text_embed', 'context_embedder', 'norm', 'proj_out']
    
    results = {}
    
    print("="*60)
    print("RTN vs GPTQ COMPARISON")
    print("="*60)
    
    # Test 1: FP16 Baseline
    print("\n[1/3] FP16 Baseline...")
    pipe = load_sd35_pipeline(device="cuda", dtype=torch.float16)
    img = generate_image(pipe, prompt)
    img.save(output_dir / "1_fp16_baseline.png")
    results['fp16'] = check_image_quality(img)
    print(f"  Image mean: {results['fp16']:.1f}")
    del pipe
    torch.cuda.empty_cache()
    
    # Test 2: Simple RTN
    print("\n[2/3] Simple RTN...")
    pipe = load_sd35_pipeline(device="cuda", dtype=torch.float16)
    n_quantized = apply_simple_rtn(pipe.transformer, skip_patterns)
    print(f"  Quantized {n_quantized} layers")
    img = generate_image(pipe, prompt)
    img.save(output_dir / "2_simple_rtn.png")
    results['rtn'] = check_image_quality(img)
    print(f"  Image mean: {results['rtn']:.1f}")
    del pipe
    torch.cuda.empty_cache()
    
    # Test 3: GPTQ-style with error diffusion
    print("\n[3/3] GPTQ-style with error diffusion...")
    pipe = load_sd35_pipeline(device="cuda", dtype=torch.float16)
    n_quantized = apply_gptq_style(pipe.transformer, "calibration_data", skip_patterns)
    print(f"  Quantized {n_quantized} layers")
    img = generate_image(pipe, prompt)
    img.save(output_dir / "3_gptq_style.png")
    results['gptq'] = check_image_quality(img)
    print(f"  Image mean: {results['gptq']:.1f}")
    del pipe
    torch.cuda.empty_cache()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"FP16 baseline:  {results['fp16']:.1f}")
    print(f"Simple RTN:     {results['rtn']:.1f} {'(BLACK)' if results['rtn'] < 10 else ''}")
    print(f"GPTQ-style:     {results['gptq']:.1f} {'(BLACK)' if results['gptq'] < 10 else ''}")
    
    if results['rtn'] > 10 and results['gptq'] < 10:
        print("\n⚠️ RTN works but GPTQ fails → Bug in error diffusion!")
    elif results['rtn'] < 10 and results['gptq'] < 10:
        print("\n⚠️ Both fail → Issue is not in error diffusion")
    elif results['rtn'] > 10 and results['gptq'] > 10:
        print("\n✅ Both work → Current implementation is fine")
    
    with open(output_dir / "comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
