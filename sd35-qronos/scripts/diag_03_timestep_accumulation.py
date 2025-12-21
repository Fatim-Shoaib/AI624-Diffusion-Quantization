#!/usr/bin/env python3
"""
=============================================================================
HYPOTHESIS 3: Cross-Timestep Error Accumulation
=============================================================================

Test if quantization errors compound across timesteps.

The diffusion model runs 28 timesteps. If we quantize:
- Timestep 0: Small error ε₀
- Timestep 1: Input is already wrong, produces error ε₁ + f(ε₀)
- Timestep 2: Error ε₂ + f(ε₁ + f(ε₀))
- ...compounds exponentially!

This test will:
1. Run FP16 inference and save intermediate latents at each timestep
2. Run quantized inference and save intermediate latents
3. Compare the divergence at each timestep

If divergence grows exponentially, cross-timestep accumulation is the issue.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
import json
from tqdm import tqdm

from config import DEFAULT_VISUAL_PROMPTS
from models.sd35_loader import load_sd35_pipeline


def simple_quantize_all_layers(transformer, skip_patterns=None):
    """Quantize all linear layers except those matching skip patterns."""
    skip_patterns = skip_patterns or []
    
    for name, module in transformer.named_modules():
        if isinstance(module, nn.Linear):
            should_skip = any(p in name for p in skip_patterns)
            if should_skip:
                continue
            
            weight = module.weight.data.float()
            max_val = weight.abs().amax(dim=-1, keepdim=True)
            scale = max_val / 7  # 4-bit
            scale = torch.clamp(scale, min=1e-8)
            
            q = torch.clamp(torch.round(weight / scale), -8, 7)
            module.weight.data = (q * scale).to(module.weight.dtype)


def run_with_intermediate_capture(pipe, prompt, num_steps=28, seed=42):
    """Run inference and capture intermediate latents at each timestep."""
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # Encode prompt
    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = (
        pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            prompt_3=None,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
        )
    )
    
    # Prepare latents
    latents = pipe.prepare_latents(
        1, pipe.transformer.config.in_channels,
        512, 512, prompt_embeds.dtype, device, generator
    )
    
    # Setup scheduler
    pipe.scheduler.set_timesteps(num_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    
    # Store intermediate latents
    intermediate_latents = []
    intermediate_latents.append(latents.clone().cpu())
    
    # Denoising loop
    for i, t in enumerate(tqdm(timesteps, desc="Denoising", leave=False)):
        latent_model_input = torch.cat([latents] * 2)
        
        timestep = t.expand(latent_model_input.shape[0])
        
        with torch.no_grad():
            noise_pred = pipe.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                return_dict=False,
            )[0]
        
        # CFG
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + 4.5 * (noise_pred_text - noise_pred_uncond)
        
        # Scheduler step
        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        
        # Store
        intermediate_latents.append(latents.clone().cpu())
    
    return intermediate_latents


def main():
    output_dir = Path("diagnosis_timesteps")
    output_dir.mkdir(exist_ok=True)
    
    prompt = DEFAULT_VISUAL_PROMPTS[0]
    num_steps = 28
    
    print("="*60)
    print("CROSS-TIMESTEP ERROR ACCUMULATION TEST")
    print("="*60)
    
    # Load pipeline
    print("\n[1/4] Loading FP16 pipeline...")
    pipe_fp16 = load_sd35_pipeline(device="cuda", dtype=torch.float16)
    
    # Run FP16 inference
    print("\n[2/4] Running FP16 inference...")
    latents_fp16 = run_with_intermediate_capture(pipe_fp16, prompt, num_steps)
    
    # Quantize and run again
    print("\n[3/4] Quantizing model...")
    skip_patterns = ['to_k', 'to_v', 'add_k_proj', 'add_v_proj', 
                     'time_text_embed', 'context_embedder', 'norm', 'proj_out']
    simple_quantize_all_layers(pipe_fp16.transformer, skip_patterns)
    
    print("\n[4/4] Running quantized inference...")
    latents_quant = run_with_intermediate_capture(pipe_fp16, prompt, num_steps)
    
    # Compare divergence at each timestep
    print("\n" + "="*60)
    print("DIVERGENCE ANALYSIS")
    print("="*60)
    
    results = []
    
    print(f"\n{'Timestep':<10} {'L2 Diff':<15} {'Rel Diff':<15} {'Max Diff':<15}")
    print("-"*55)
    
    for t in range(len(latents_fp16)):
        l_fp16 = latents_fp16[t].float()
        l_quant = latents_quant[t].float()
        
        l2_diff = (l_fp16 - l_quant).pow(2).sum().sqrt().item()
        rel_diff = l2_diff / (l_fp16.pow(2).sum().sqrt().item() + 1e-8)
        max_diff = (l_fp16 - l_quant).abs().max().item()
        
        results.append({
            "timestep": t,
            "l2_diff": l2_diff,
            "rel_diff": rel_diff,
            "max_diff": max_diff,
        })
        
        print(f"{t:<10} {l2_diff:<15.6f} {rel_diff:<15.6f} {max_diff:<15.6f}")
    
    # Analyze growth pattern
    l2_diffs = [r['l2_diff'] for r in results]
    
    # Check if exponential growth
    if len(l2_diffs) > 5:
        early_avg = np.mean(l2_diffs[1:5])
        late_avg = np.mean(l2_diffs[-5:])
        growth_ratio = late_avg / (early_avg + 1e-8)
        
        print(f"\nEarly timestep avg diff: {early_avg:.6f}")
        print(f"Late timestep avg diff: {late_avg:.6f}")
        print(f"Growth ratio: {growth_ratio:.2f}x")
        
        if growth_ratio > 10:
            print("\n⚠️ EXPONENTIAL ERROR GROWTH DETECTED!")
            print("   This confirms cross-timestep error accumulation.")
        elif growth_ratio > 2:
            print("\n⚠️ Moderate error growth detected.")
        else:
            print("\n✅ Error growth is contained.")
    
    # Save results
    with open(output_dir / "timestep_divergence.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()