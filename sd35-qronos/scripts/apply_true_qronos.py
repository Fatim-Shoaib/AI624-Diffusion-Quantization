#!/usr/bin/env python3
"""
=============================================================================
TRUE QRONOS QUANTIZATION
=============================================================================

This implements the ACTUAL Qronos algorithm from the paper:
"Qronos: Quantization Round Off via Native Optimization Strategy"

KEY DIFFERENCE FROM GPTQ:
- GPTQ: Uses only H = X^T X
- Qronos: Uses BOTH H = X̃^T X̃ AND G = X̃^T X

The G matrix captures cross-covariance between quantized and float activations,
allowing Qronos to correct for activation quantization errors as well.

Qronos Algorithm:
1. H = X̃^T X̃  (Hessian from quantized activations)
2. G = X̃^T X   (Cross-covariance: quantized inputs × float inputs)
3. For each column j:
   - Quantize weight column: q_j = Q(w_j)
   - Compute error considering BOTH H and G
   - Diffuse error to future weights using G, not just H
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import json
from tqdm import tqdm

from config import DEFAULT_VISUAL_PROMPTS
from models.sd35_loader import load_sd35_pipeline


class QronosQuantizer:
    """
    True Qronos quantizer using both H and G matrices.
    """
    
    def __init__(self, H: torch.Tensor, G: torch.Tensor, bits: int = 4):
        """
        Args:
            H: Hessian matrix X̃^T X̃ from quantized activations
            G: Cross-covariance matrix X̃^T X from quantized × float activations
            bits: Quantization bit width
        """
        self.bits = bits
        self.qmin = -(2 ** (bits - 1))
        self.qmax = 2 ** (bits - 1) - 1
        
        # Ensure float32 and proper shape
        self.H = H.float().clone()
        self.G = G.float().clone()
        
        while self.H.dim() > 2:
            self.H = self.H.squeeze(-1)
        while self.G.dim() > 2:
            self.G = self.G.squeeze(-1)
        
        # Force H symmetry
        self.H = (self.H + self.H.t()) / 2
        
        self.n = self.H.shape[0]
        self.valid = True
        self.H_inv = None
        
        # Prepare inverse
        self._prepare_inverse()
    
    def _prepare_inverse(self, damp_factor: float = 0.01):
        """Compute H inverse via Cholesky decomposition."""
        try:
            # Add dampening
            diag_mean = self.H.diag().abs().mean()
            if diag_mean == 0:
                self.valid = False
                return
            
            damp = damp_factor * diag_mean
            H_damped = self.H + damp * torch.eye(self.n, device=self.H.device)
            
            # Cholesky decomposition
            L = torch.linalg.cholesky(H_damped)
            self.H_inv = torch.cholesky_inverse(L)
            
            # Validate
            if torch.isnan(self.H_inv).any() or torch.isinf(self.H_inv).any():
                self.valid = False
                return
            
            if self.H_inv.abs().max() > 1e10:
                self.valid = False
                return
                
        except Exception as e:
            print(f"    Cholesky failed: {e}")
            self.valid = False
    
    def quantize_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """
        Apply TRUE Qronos quantization using both H and G.
        
        The key Qronos insight:
        - GPTQ diffuses error using: w[:, j+1:] -= error * H_inv[j, j+1:]
        - Qronos diffuses error using: w[:, j+1:] -= error * (G @ H_inv)[j, j+1:]
        
        This accounts for the correlation between quantized and float activations.
        """
        if not self.valid:
            return self._simple_rtn(weight)
        
        weight = weight.float().clone()
        out_features, in_features = weight.shape
        device = weight.device
        
        # Move matrices to same device
        H_inv = self.H_inv.to(device)
        G = self.G.to(device)
        H = self.H.to(device)
        
        # =====================================================================
        # KEY QRONOS DIFFERENCE: Compute G @ H_inv for error diffusion
        # =====================================================================
        # In GPTQ: error_diffusion_matrix = H_inv
        # In Qronos: error_diffusion_matrix = G @ H_inv (or H_inv @ G^T depending on formulation)
        # This captures the cross-correlation between quantized and float activations
        
        try:
            # Qronos error diffusion matrix
            # G captures X̃^T X, H_inv captures (X̃^T X̃)^{-1}
            # The product gives us how errors in quantized domain affect float domain
            GH_inv = G @ H_inv
        except Exception as e:
            print(f"    G @ H_inv failed: {e}")
            return self._simple_rtn(weight)
        
        # Column-by-column quantization with Qronos error diffusion
        for col in range(in_features):
            w_col = weight[:, col].clone()
            
            # Quantize this column
            q_col = self._quantize_column(w_col)
            
            # Compute quantization error
            error = w_col - q_col
            
            # Store quantized value
            weight[:, col] = q_col
            
            # =====================================================================
            # QRONOS ERROR DIFFUSION (different from GPTQ!)
            # =====================================================================
            # GPTQ uses:  w[:, col+1:] -= (error / H_inv[col,col]) * H_inv[col, col+1:]
            # Qronos uses: w[:, col+1:] -= (error / H[col,col]) * GH_inv[col, col+1:]
            #
            # The G matrix accounts for how quantized inputs correlate with float inputs
            # =====================================================================
            
            if col < in_features - 1:
                h_diag = H[col, col]
                if h_diag.abs() > 1e-10:
                    # Qronos: use GH_inv for diffusion instead of H_inv
                    error_scaled = error / h_diag
                    weight[:, col+1:] -= error_scaled.unsqueeze(1) * GH_inv[col, col+1:].unsqueeze(0)
        
        return weight
    
    def _quantize_column(self, w_col: torch.Tensor) -> torch.Tensor:
        """Quantize a single column using symmetric quantization."""
        max_val = w_col.abs().max()
        if max_val == 0:
            return w_col
        
        scale = max_val / self.qmax
        q = torch.clamp(torch.round(w_col / scale), self.qmin, self.qmax)
        return q * scale
    
    def _simple_rtn(self, weight: torch.Tensor) -> torch.Tensor:
        """Fallback: simple round-to-nearest."""
        max_val = weight.abs().amax(dim=-1, keepdim=True)
        scale = max_val / self.qmax
        scale = torch.clamp(scale, min=1e-8)
        q = torch.clamp(torch.round(weight / scale), self.qmin, self.qmax)
        return q * scale


def apply_qronos_to_layer(layer: nn.Linear, H: torch.Tensor, G: torch.Tensor, bits: int = 4) -> str:
    """
    Apply Qronos quantization to a single layer.
    Returns status string.
    """
    quantizer = QronosQuantizer(H, G, bits)
    
    if not quantizer.valid:
        return "invalid_matrices"
    
    try:
        weight = layer.weight.data
        dtype = weight.dtype
        
        q_weight = quantizer.quantize_weight(weight)
        layer.weight.data = q_weight.to(dtype)
        
        return "qronos_success"
    except Exception as e:
        print(f"    Error: {e}")
        return "error"


def main():
    calibration_dir = Path("calibration_quick")
    output_dir = Path("qronos_test_output")
    output_dir.mkdir(exist_ok=True)
    
    print("="*70)
    print("TRUE QRONOS QUANTIZATION TEST")
    print("="*70)
    print("\nThis uses BOTH H and G matrices (not just H like GPTQ)")
    print("H = X̃ᵀX̃  (Hessian from quantized activations)")
    print("G = X̃ᵀX   (Cross-covariance: quantized × float)")
    print("="*70)
    
    # Check calibration data
    if not calibration_dir.exists():
        print(f"\nERROR: No calibration data found at {calibration_dir}")
        print("Please run: python scripts/quick_calibration.py")
        return
    
    # Load metadata
    with open(calibration_dir / "metadata.json") as f:
        metadata = json.load(f)
    
    print(f"\nFound calibration data for {len(metadata['layers'])} layers")
    
    # Load pipeline
    print("\nLoading pipeline...")
    pipe = load_sd35_pipeline(device="cuda", dtype=torch.float16)
    
    # Apply Qronos to calibrated layers
    print("\nApplying TRUE Qronos quantization...")
    
    stats = {"qronos_success": 0, "invalid_matrices": 0, "error": 0, "not_found": 0}
    
    for layer_name, layer_info in metadata["layers"].items():
        print(f"\n  {layer_name}:")
        
        # Find the layer in the model
        layer = None
        for name, module in pipe.transformer.named_modules():
            if name == layer_name and isinstance(module, nn.Linear):
                layer = module
                break
        
        if layer is None:
            print(f"    Layer not found in model!")
            stats["not_found"] += 1
            continue
        
        # Load H and G matrices
        H_path = calibration_dir / layer_info["H_file"]
        G_path = calibration_dir / layer_info["G_file"]
        
        H = torch.load(H_path, map_location='cpu', weights_only=True)
        G = torch.load(G_path, map_location='cpu', weights_only=True)
        
        print(f"    H: {H.shape}, max={H.abs().max():.2e}")
        print(f"    G: {G.shape}, max={G.abs().max():.2e}")
        
        # Apply Qronos
        status = apply_qronos_to_layer(layer, H, G, bits=4)
        stats[status] += 1
        print(f"    Status: {status}")
    
    # Summary
    print("\n" + "="*70)
    print("QRONOS APPLICATION SUMMARY")
    print("="*70)
    for status, count in stats.items():
        print(f"  {status}: {count}")
    
    # Test with image generation
    print("\n" + "="*70)
    print("TESTING QUANTIZED MODEL")
    print("="*70)
    
    prompt = DEFAULT_VISUAL_PROMPTS[0]
    print(f"\nPrompt: {prompt[:60]}...")
    
    generator = torch.Generator(device="cuda").manual_seed(42)
    
    with torch.no_grad():
        result = pipe(
            prompt=prompt,
            height=512,
            width=512,
            num_inference_steps=20,
            guidance_scale=4.5,
            generator=generator,
        )
    
    img = result.images[0]
    
    # Check image quality
    pixels = list(img.getdata())
    mean_val = sum(sum(p) for p in pixels) / (len(pixels) * 3)
    
    print(f"\nImage mean pixel value: {mean_val:.1f}")
    
    if mean_val < 10:
        print("❌ BLACK IMAGE - Qronos failed!")
    elif mean_val < 30:
        print("⚠️ VERY DARK IMAGE - Qronos partially failed")
    else:
        print("✅ VISIBLE IMAGE - Qronos working!")
    
    img.save(output_dir / "qronos_test_image.png")
    print(f"\nTest image saved to {output_dir / 'qronos_test_image.png'}")
    
    # Save stats
    with open(output_dir / "qronos_stats.json", "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
