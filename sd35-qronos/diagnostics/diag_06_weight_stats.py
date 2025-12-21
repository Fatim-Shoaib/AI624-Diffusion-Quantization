#!/usr/bin/env python3
"""
=============================================================================
HYPOTHESIS 6: Weight Statistics Analysis
=============================================================================

Analyze weight distributions before and after quantization to check:
1. Are there outlier weights that break quantization?
2. Does quantization introduce NaN/Inf?
3. Is the quantization error abnormally large for some layers?
4. Are there layers with extreme weight ranges?

This helps identify if certain layers have weight distributions
that are fundamentally incompatible with 4-bit quantization.
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


def analyze_weight_stats(weight, name):
    """Compute comprehensive statistics for a weight tensor."""
    w = weight.float()
    
    stats = {
        "name": name,
        "shape": list(weight.shape),
        "dtype": str(weight.dtype),
        "min": w.min().item(),
        "max": w.max().item(),
        "mean": w.mean().item(),
        "std": w.std().item(),
        "abs_mean": w.abs().mean().item(),
        "abs_max": w.abs().max().item(),
        "sparsity": (w == 0).float().mean().item(),
        "has_nan": torch.isnan(w).any().item(),
        "has_inf": torch.isinf(w).any().item(),
    }
    
    # Outlier analysis
    percentiles = [1, 5, 25, 50, 75, 95, 99]
    flat_w = w.flatten()
    for p in percentiles:
        stats[f"p{p}"] = torch.quantile(flat_w, p/100).item()
    
    # Dynamic range (important for quantization)
    stats["dynamic_range"] = stats["abs_max"] / (w.abs().mean().item() + 1e-8)
    
    return stats


def quantize_and_compare(weight, bits=4):
    """Quantize weight and compute error statistics."""
    w = weight.float()
    
    # Simple per-channel quantization
    max_val = w.abs().amax(dim=-1, keepdim=True)
    scale = max_val / (2 ** (bits - 1) - 1)
    scale = torch.clamp(scale, min=1e-8)
    
    qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    q = torch.clamp(torch.round(w / scale), qmin, qmax)
    w_quant = q * scale
    
    # Error analysis
    error = (w - w_quant).abs()
    rel_error = error / (w.abs() + 1e-8)
    
    return {
        "mse": error.pow(2).mean().item(),
        "mae": error.mean().item(),
        "max_error": error.max().item(),
        "mean_rel_error": rel_error.mean().item(),
        "max_rel_error": rel_error.max().item(),
        "quant_has_nan": torch.isnan(w_quant).any().item(),
        "quant_has_inf": torch.isinf(w_quant).any().item(),
    }


def main():
    output_dir = Path("diagnosis_weight_stats")
    output_dir.mkdir(exist_ok=True)
    
    print("="*60)
    print("WEIGHT STATISTICS ANALYSIS")
    print("="*60)
    
    # Load model
    print("\nLoading model...")
    pipe = load_sd35_pipeline(device="cuda", dtype=torch.float16)
    
    # Analyze all linear layers
    all_stats = []
    
    print("\nAnalyzing layers...")
    for name, module in tqdm(list(pipe.transformer.named_modules())):
        if not isinstance(module, nn.Linear):
            continue
        
        weight = module.weight.data.cpu()
        
        # Get basic stats
        stats = analyze_weight_stats(weight, name)
        
        # Get quantization error
        quant_stats = quantize_and_compare(weight)
        stats.update(quant_stats)
        
        all_stats.append(stats)
    
    # Find problematic layers
    print("\n" + "="*60)
    print("ANALYSIS RESULTS")
    print("="*60)
    
    # Check for NaN/Inf
    nan_layers = [s for s in all_stats if s['has_nan'] or s['quant_has_nan']]
    inf_layers = [s for s in all_stats if s['has_inf'] or s['quant_has_inf']]
    
    if nan_layers:
        print(f"\n❌ Layers with NaN: {len(nan_layers)}")
        for s in nan_layers[:5]:
            print(f"   - {s['name']}")
    
    if inf_layers:
        print(f"\n❌ Layers with Inf: {len(inf_layers)}")
        for s in inf_layers[:5]:
            print(f"   - {s['name']}")
    
    # High dynamic range (problematic for quantization)
    high_dynamic = [s for s in all_stats if s['dynamic_range'] > 100]
    if high_dynamic:
        print(f"\n⚠️ Layers with high dynamic range (>100): {len(high_dynamic)}")
        sorted_by_dr = sorted(high_dynamic, key=lambda x: x['dynamic_range'], reverse=True)
        for s in sorted_by_dr[:10]:
            print(f"   - {s['name']}: {s['dynamic_range']:.1f}")
    
    # High quantization error
    high_error = [s for s in all_stats if s['mean_rel_error'] > 0.5]
    if high_error:
        print(f"\n⚠️ Layers with high quantization error (>50%): {len(high_error)}")
        sorted_by_err = sorted(high_error, key=lambda x: x['mean_rel_error'], reverse=True)
        for s in sorted_by_err[:10]:
            print(f"   - {s['name']}: {s['mean_rel_error']*100:.1f}%")
    
    # Summary statistics
    print("\n" + "-"*60)
    print("OVERALL STATISTICS")
    print("-"*60)
    
    all_mse = [s['mse'] for s in all_stats]
    all_dr = [s['dynamic_range'] for s in all_stats]
    
    print(f"Total layers analyzed: {len(all_stats)}")
    print(f"MSE - Mean: {np.mean(all_mse):.6f}, Max: {np.max(all_mse):.6f}")
    print(f"Dynamic range - Mean: {np.mean(all_dr):.1f}, Max: {np.max(all_dr):.1f}")
    
    # Identify layer types with highest errors
    layer_type_errors = {}
    for s in all_stats:
        # Extract layer type
        parts = s['name'].split('.')
        layer_type = '.'.join(parts[-2:]) if len(parts) >= 2 else s['name']
        
        if layer_type not in layer_type_errors:
            layer_type_errors[layer_type] = []
        layer_type_errors[layer_type].append(s['mse'])
    
    print("\n" + "-"*60)
    print("AVERAGE MSE BY LAYER TYPE")
    print("-"*60)
    
    type_avg = {k: np.mean(v) for k, v in layer_type_errors.items()}
    sorted_types = sorted(type_avg.items(), key=lambda x: x[1], reverse=True)
    
    for layer_type, avg_mse in sorted_types[:15]:
        print(f"  {layer_type:<30}: {avg_mse:.6f}")
    
    # Save full results
    with open(output_dir / "weight_stats.json", "w") as f:
        json.dump(all_stats, f, indent=2)
    
    print(f"\nFull results saved to {output_dir}/weight_stats.json")


if __name__ == "__main__":
    main()
