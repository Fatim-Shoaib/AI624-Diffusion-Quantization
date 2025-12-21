#!/usr/bin/env python3
"""
=============================================================================
HYPOTHESIS 7: Save/Load Corruption Test
=============================================================================

Test if the issue occurs during save/load of quantized weights.

Scenarios:
1. Quantize in memory → generate image (works from earlier tests)
2. Quantize → save → load → generate image (might be corrupted)

If #1 works but #2 fails, there's a bug in save/load logic.
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


def simple_quantize_all(transformer, skip_patterns=None):
    """Quantize all linear layers."""
    skip_patterns = skip_patterns or []
    
    for name, module in transformer.named_modules():
        if isinstance(module, nn.Linear):
            if any(p in name for p in skip_patterns):
                continue
            
            weight = module.weight.data.float()
            max_val = weight.abs().amax(dim=-1, keepdim=True)
            scale = max_val / 7
            scale = torch.clamp(scale, min=1e-8)
            
            q = torch.clamp(torch.round(weight / scale), -8, 7)
            module.weight.data = (q * scale).to(module.weight.dtype)


def generate_image(pipe, prompt, seed=42):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    with torch.no_grad():
        result = pipe(prompt=prompt, height=512, width=512,
                     num_inference_steps=20, guidance_scale=4.5, generator=generator)
    return result.images[0]


def check_image_quality(image):
    pixels = list(image.getdata())
    return sum(sum(p) for p in pixels) / (len(pixels) * 3)


def main():
    output_dir = Path("diagnosis_save_load")
    output_dir.mkdir(exist_ok=True)
    temp_model_path = output_dir / "temp_quantized_state.pt"
    
    prompt = DEFAULT_VISUAL_PROMPTS[0]
    skip_patterns = ['to_k', 'to_v', 'add_k_proj', 'add_v_proj',
                     'time_text_embed', 'context_embedder', 'norm', 'proj_out']
    
    print("="*60)
    print("SAVE/LOAD CORRUPTION TEST")
    print("="*60)
    
    results = {}
    
    # Test 1: FP16 Baseline
    print("\n[1/4] FP16 Baseline...")
    pipe = load_sd35_pipeline(device="cuda", dtype=torch.float16)
    img = generate_image(pipe, prompt)
    img.save(output_dir / "1_fp16_baseline.png")
    results['fp16'] = check_image_quality(img)
    print(f"  Quality: {results['fp16']:.1f}")
    del pipe
    torch.cuda.empty_cache()
    
    # Test 2: Quantize in memory, generate immediately
    print("\n[2/4] Quantize in memory → generate...")
    pipe = load_sd35_pipeline(device="cuda", dtype=torch.float16)
    simple_quantize_all(pipe.transformer, skip_patterns)
    img = generate_image(pipe, prompt)
    img.save(output_dir / "2_quantized_in_memory.png")
    results['in_memory'] = check_image_quality(img)
    print(f"  Quality: {results['in_memory']:.1f}")
    
    # Save the quantized state dict
    print("\n[3/4] Saving quantized weights...")
    state_dict = pipe.transformer.state_dict()
    torch.save(state_dict, temp_model_path)
    print(f"  Saved to {temp_model_path}")
    
    # Check file integrity
    file_size = temp_model_path.stat().st_size / (1024**3)
    print(f"  File size: {file_size:.2f} GB")
    
    del pipe
    torch.cuda.empty_cache()
    
    # Test 3: Load saved weights into fresh model
    print("\n[4/4] Load saved weights → generate...")
    pipe = load_sd35_pipeline(device="cuda", dtype=torch.float16)
    
    # Load saved state dict
    saved_state = torch.load(temp_model_path, map_location="cuda")
    pipe.transformer.load_state_dict(saved_state)
    
    img = generate_image(pipe, prompt)
    img.save(output_dir / "3_loaded_from_file.png")
    results['from_file'] = check_image_quality(img)
    print(f"  Quality: {results['from_file']:.1f}")
    
    # Compare
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"FP16 baseline:      {results['fp16']:.1f}")
    print(f"Quantized (memory): {results['in_memory']:.1f}")
    print(f"Quantized (loaded): {results['from_file']:.1f}")
    
    # Verify weights match
    print("\n" + "-"*60)
    print("WEIGHT VERIFICATION")
    print("-"*60)
    
    # Reload both and compare
    pipe1 = load_sd35_pipeline(device="cuda", dtype=torch.float16)
    simple_quantize_all(pipe1.transformer, skip_patterns)
    
    pipe2 = load_sd35_pipeline(device="cuda", dtype=torch.float16)
    saved_state = torch.load(temp_model_path, map_location="cuda")
    pipe2.transformer.load_state_dict(saved_state)
    
    # Compare weights
    mismatch_count = 0
    for (name1, p1), (name2, p2) in zip(pipe1.transformer.named_parameters(),
                                         pipe2.transformer.named_parameters()):
        if not torch.allclose(p1, p2, atol=1e-6):
            mismatch_count += 1
            diff = (p1 - p2).abs().max().item()
            if mismatch_count <= 5:
                print(f"  Mismatch: {name1}, max diff: {diff:.6f}")
    
    if mismatch_count == 0:
        print("  ✅ All weights match perfectly")
    else:
        print(f"  ❌ {mismatch_count} parameters have mismatches")
    
    # Interpretation
    print("\n" + "-"*60)
    print("INTERPRETATION")
    print("-"*60)
    
    if results['in_memory'] > 10 and results['from_file'] < 10:
        print("⚠️ Save/load is corrupting the weights!")
    elif results['in_memory'] < 10 and results['from_file'] < 10:
        print("⚠️ Both methods produce black images - not a save/load issue")
    elif results['in_memory'] > 10 and results['from_file'] > 10:
        print("✅ Both work - save/load is fine")
    
    # Cleanup
    temp_model_path.unlink()
    
    with open(output_dir / "save_load_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
