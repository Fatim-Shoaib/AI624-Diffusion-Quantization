#!/usr/bin/env python3
"""
=============================================================================
HYPOTHESIS 8: Inspect Actual Quantized Model
=============================================================================

Load our actual quantized model from quantized_model/ and inspect:
1. Are there NaN/Inf values in any weights?
2. Are weight distributions reasonable?
3. Compare to FP16 baseline weights

This will tell us if the saved quantized model has corrupted weights.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import json
import numpy as np
from tqdm import tqdm

from models.sd35_loader import load_sd35_pipeline


def main():
    output_dir = Path("diagnosis_inspect_model")
    output_dir.mkdir(exist_ok=True)
    
    quantized_dir = Path("quantized_model")
    
    print("="*60)
    print("INSPECT ACTUAL QUANTIZED MODEL")
    print("="*60)
    
    # Load FP16 baseline
    print("\n[1/3] Loading FP16 baseline...")
    pipe_fp16 = load_sd35_pipeline(device="cpu", dtype=torch.float16)
    
    # Load quantized model
    print("\n[2/3] Loading quantized model...")
    if not quantized_dir.exists():
        print("❌ No quantized model found at", quantized_dir)
        return
    
    pipe_quant = load_sd35_pipeline(device="cpu", dtype=torch.float16)
    state_dict_path = quantized_dir / "transformer_state_dict.pt"
    
    if not state_dict_path.exists():
        print("❌ No state dict found at", state_dict_path)
        return
    
    state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=False)
    pipe_quant.transformer.load_state_dict(state_dict)
    
    # Compare weights
    print("\n[3/3] Comparing weights...")
    
    results = []
    nan_layers = []
    inf_layers = []
    zero_layers = []
    high_diff_layers = []
    
    for (name_fp16, param_fp16), (name_quant, param_quant) in zip(
        pipe_fp16.transformer.named_parameters(),
        pipe_quant.transformer.named_parameters()
    ):
        if "weight" not in name_fp16:
            continue
        
        w_fp16 = param_fp16.data.float()
        w_quant = param_quant.data.float()
        
        # Check for issues
        has_nan = torch.isnan(w_quant).any().item()
        has_inf = torch.isinf(w_quant).any().item()
        is_zero = (w_quant == 0).all().item()
        
        # Compute difference
        diff = (w_fp16 - w_quant).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        rel_diff = mean_diff / (w_fp16.abs().mean().item() + 1e-8)
        
        result = {
            "name": name_fp16,
            "shape": list(w_fp16.shape),
            "has_nan": has_nan,
            "has_inf": has_inf,
            "is_zero": is_zero,
            "fp16_mean": w_fp16.abs().mean().item(),
            "quant_mean": w_quant.abs().mean().item(),
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "rel_diff": rel_diff,
        }
        results.append(result)
        
        if has_nan:
            nan_layers.append(name_fp16)
        if has_inf:
            inf_layers.append(name_fp16)
        if is_zero:
            zero_layers.append(name_fp16)
        if rel_diff > 0.5:  # More than 50% relative change
            high_diff_layers.append((name_fp16, rel_diff))
    
    # Report findings
    print("\n" + "="*60)
    print("FINDINGS")
    print("="*60)
    
    print(f"\nTotal layers checked: {len(results)}")
    
    if nan_layers:
        print(f"\n❌ LAYERS WITH NaN: {len(nan_layers)}")
        for name in nan_layers[:10]:
            print(f"   - {name}")
        if len(nan_layers) > 10:
            print(f"   ... and {len(nan_layers) - 10} more")
    else:
        print("\n✅ No NaN values found")
    
    if inf_layers:
        print(f"\n❌ LAYERS WITH Inf: {len(inf_layers)}")
        for name in inf_layers[:10]:
            print(f"   - {name}")
    else:
        print("✅ No Inf values found")
    
    if zero_layers:
        print(f"\n⚠️ LAYERS THAT ARE ALL ZEROS: {len(zero_layers)}")
        for name in zero_layers[:10]:
            print(f"   - {name}")
    else:
        print("✅ No all-zero layers")
    
    if high_diff_layers:
        print(f"\n⚠️ LAYERS WITH >50% CHANGE: {len(high_diff_layers)}")
        sorted_by_diff = sorted(high_diff_layers, key=lambda x: x[1], reverse=True)
        for name, diff in sorted_by_diff[:15]:
            print(f"   - {name}: {diff*100:.1f}% change")
    
    # Summary statistics
    print("\n" + "-"*60)
    print("OVERALL STATISTICS")
    print("-"*60)
    
    all_rel_diffs = [r['rel_diff'] for r in results]
    print(f"Mean relative change: {np.mean(all_rel_diffs)*100:.2f}%")
    print(f"Max relative change: {np.max(all_rel_diffs)*100:.2f}%")
    print(f"Median relative change: {np.median(all_rel_diffs)*100:.2f}%")
    
    # Check if quantization actually happened
    unchanged = [r for r in results if r['max_diff'] < 1e-6]
    print(f"\nUnchanged layers (not quantized): {len(unchanged)}")
    
    # Save results
    with open(output_dir / "inspection_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nFull results saved to {output_dir}")
    
    # Final verdict
    print("\n" + "="*60)
    print("VERDICT")
    print("="*60)
    
    if nan_layers or inf_layers:
        print("❌ CRITICAL: Quantized model contains NaN/Inf values!")
        print("   This is definitely causing black images.")
    elif zero_layers and len(zero_layers) > 10:
        print("⚠️ Many layers are all zeros - possible corruption")
    elif len(high_diff_layers) > len(results) * 0.5:
        print("⚠️ Most layers have >50% change - aggressive quantization")
    else:
        print("✅ Weight statistics look reasonable")
        print("   Issue may be elsewhere (e.g., layer interactions)")


if __name__ == "__main__":
    main()
