#!/usr/bin/env python3
"""
=============================================================================
Apply Quantization with H Matrix Validation
=============================================================================

This script applies quantization with proper validation:
1. Check if H matrix is valid (symmetric, positive definite)
2. If valid: Use GPTQ-style error diffusion
3. If invalid: Fall back to simple RTN

This ensures we never use corrupted H matrices.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import json
from tqdm import tqdm

from config import DEFAULT_VISUAL_PROMPTS
from models.sd35_loader import load_sd35_pipeline, save_quantized_transformer


def validate_h_matrix(H: torch.Tensor, layer_name: str, damp: float = 1e-5) -> tuple:
    """
    Validate H matrix and return (is_valid, H_inv or None, reason).
    """
    # Ensure float32 and 2D
    H = H.float()
    while H.dim() > 2:
        H = H.squeeze(-1)
    
    n = H.shape[0]
    
    # Check 1: Is H symmetric?
    sym_error = (H - H.t()).abs().max().item()
    if sym_error > 1e-3:
        return False, None, f"Not symmetric (error={sym_error:.2e})"
    
    # Force symmetry
    H = (H + H.t()) / 2
    
    # Check 2: Any NaN/Inf?
    if torch.isnan(H).any() or torch.isinf(H).any():
        return False, None, "Contains NaN/Inf"
    
    # Check 3: Add dampening and try Cholesky
    diag_mean = H.diag().abs().mean()
    if diag_mean == 0:
        return False, None, "Zero diagonal"
    
    H_damped = H + damp * diag_mean * torch.eye(n, device=H.device, dtype=H.dtype)
    
    try:
        L = torch.linalg.cholesky(H_damped)
        H_inv = torch.cholesky_inverse(L)
        
        # Check 4: Is H_inv reasonable?
        if torch.isnan(H_inv).any() or torch.isinf(H_inv).any():
            return False, None, "H_inv contains NaN/Inf"
        
        if H_inv.abs().max() > 1e10:
            return False, None, f"H_inv too large (max={H_inv.abs().max():.2e})"
        
        return True, H_inv, "OK"
        
    except Exception as e:
        return False, None, f"Cholesky failed: {str(e)[:50]}"


def apply_simple_rtn(layer: nn.Linear, bits: int = 4):
    """Apply simple RTN quantization."""
    weight = layer.weight.data.float()
    out_features, in_features = weight.shape
    
    # Per-channel quantization
    max_val = weight.abs().amax(dim=-1, keepdim=True)
    scale = max_val / (2 ** (bits - 1) - 1)
    scale = torch.clamp(scale, min=1e-8)
    
    qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    q = torch.clamp(torch.round(weight / scale), qmin, qmax)
    
    layer.weight.data = (q * scale).to(layer.weight.dtype)


def apply_gptq_with_h(layer: nn.Linear, H_inv: torch.Tensor, bits: int = 4):
    """Apply GPTQ-style quantization with error diffusion."""
    weight = layer.weight.data.clone().float()
    out_features, in_features = weight.shape
    device = weight.device
    dtype = layer.weight.dtype
    
    # Ensure H_inv is on same device
    H_inv = H_inv.to(device).float()
    
    # Column-by-column quantization with error diffusion
    for col in range(in_features):
        w_col = weight[:, col].clone()
        
        # Quantize
        max_val = w_col.abs().max()
        if max_val > 0:
            scale = max_val / (2 ** (bits - 1) - 1)
            qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
            q_col = torch.clamp(torch.round(w_col / scale), qmin, qmax) * scale
        else:
            q_col = w_col
        
        # Compute error
        h_diag = H_inv[col, col]
        if h_diag.abs() < 1e-10:
            h_diag = 1e-10  # Prevent division by zero
        
        error = (w_col - q_col) / h_diag
        
        # Update this column
        weight[:, col] = q_col
        
        # Diffuse error to remaining columns
        if col < in_features - 1:
            weight[:, col+1:] -= error.unsqueeze(1) * H_inv[col, col+1:].unsqueeze(0)
    
    layer.weight.data = weight.to(dtype)


def main():
    output_dir = Path("quantized_model_validated")
    output_dir.mkdir(exist_ok=True)
    calibration_dir = Path("calibration_data")
    
    # Skip patterns for sensitive layers
    skip_patterns = [
        'to_k', 'to_v', 'add_k_proj', 'add_v_proj',  # K/V layers
        'time_text_embed', 'context_embedder',  # Embeddings
        'norm', 'proj_out'  # Norms and output
    ]
    
    print("="*60)
    print("QUANTIZATION WITH H MATRIX VALIDATION")
    print("="*60)
    
    # Load model
    print("\nLoading model...")
    pipe = load_sd35_pipeline(device="cuda", dtype=torch.float16)
    
    # Load calibration metadata
    if not calibration_dir.exists():
        print("No calibration data found! Using simple RTN for all layers.")
        use_calibration = False
    else:
        with open(calibration_dir / "metadata.json") as f:
            metadata = json.load(f)
        use_calibration = True
        print(f"Loaded calibration metadata for {len(metadata['layers'])} layers")
    
    # Collect all linear layers
    linear_layers = []
    for name, module in pipe.transformer.named_modules():
        if isinstance(module, nn.Linear):
            linear_layers.append((name, module))
    
    print(f"Found {len(linear_layers)} linear layers")
    
    # Statistics
    stats = {
        "skipped": 0,
        "gptq_success": 0,
        "rtn_fallback": 0,
        "h_invalid": 0,
        "no_calibration": 0,
    }
    
    # Process each layer
    for name, layer in tqdm(linear_layers, desc="Quantizing"):
        # Check skip patterns
        if any(p in name for p in skip_patterns):
            stats["skipped"] += 1
            continue
        
        # Try to use GPTQ with calibration
        used_gptq = False
        
        if use_calibration and name in metadata['layers']:
            layer_info = metadata['layers'][name]
            H_file = calibration_dir / layer_info['H_file']
            
            if H_file.exists():
                H = torch.load(H_file, map_location='cpu', weights_only=True)
                
                # Validate H matrix
                is_valid, H_inv, reason = validate_h_matrix(H, name)
                
                if is_valid:
                    try:
                        apply_gptq_with_h(layer, H_inv, bits=4)
                        stats["gptq_success"] += 1
                        used_gptq = True
                    except Exception as e:
                        tqdm.write(f"  GPTQ failed for {name}: {e}")
                        stats["h_invalid"] += 1
                else:
                    stats["h_invalid"] += 1
                    # Uncomment to see which layers have invalid H:
                    # tqdm.write(f"  Invalid H for {name}: {reason}")
            else:
                stats["no_calibration"] += 1
        else:
            stats["no_calibration"] += 1
        
        # Fall back to RTN if GPTQ wasn't used
        if not used_gptq:
            apply_simple_rtn(layer, bits=4)
            stats["rtn_fallback"] += 1
    
    # Save model
    print("\nSaving quantized model...")
    save_quantized_transformer(pipe.transformer, output_dir)
    
    # Summary
    print("\n" + "="*60)
    print("QUANTIZATION SUMMARY")
    print("="*60)
    print(f"Total layers:      {len(linear_layers)}")
    print(f"Skipped:           {stats['skipped']}")
    print(f"GPTQ success:      {stats['gptq_success']}")
    print(f"RTN fallback:      {stats['rtn_fallback']}")
    print(f"  - Invalid H:     {stats['h_invalid']}")
    print(f"  - No calibration:{stats['no_calibration']}")
    print(f"\nModel saved to {output_dir}")
    
    # Quick test
    print("\n" + "="*60)
    print("QUICK QUALITY TEST")
    print("="*60)
    
    prompt = DEFAULT_VISUAL_PROMPTS[0]
    generator = torch.Generator(device="cuda").manual_seed(42)
    
    with torch.no_grad():
        result = pipe(
            prompt=prompt,
            height=512, width=512,
            num_inference_steps=20,
            guidance_scale=4.5,
            generator=generator,
        )
    
    img = result.images[0]
    pixels = list(img.getdata())
    mean_val = sum(sum(p) for p in pixels) / (len(pixels) * 3)
    
    print(f"Test image mean pixel value: {mean_val:.1f}")
    
    if mean_val < 10:
        print("❌ BLACK IMAGE - Something is still wrong!")
    else:
        print("✅ VISIBLE IMAGE - Quantization successful!")
    
    img.save(output_dir / "test_image.png")
    print(f"Test image saved to {output_dir / 'test_image.png'}")


if __name__ == "__main__":
    main()
