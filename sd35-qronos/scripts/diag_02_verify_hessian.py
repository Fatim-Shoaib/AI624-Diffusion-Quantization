#!/usr/bin/env python3
"""
=============================================================================
HYPOTHESIS 2: Verify Hessian Matrix Computation
=============================================================================

Test if our incremental H computation is mathematically correct.
Compare incremental vs batch computation.

This will tell us:
- If the calibration data is being computed correctly
- If there's a bug in the incremental averaging
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from config import DEFAULT_VISUAL_PROMPTS
from models.sd35_loader import load_sd35_pipeline


def test_incremental_vs_batch():
    """Test that incremental H computation equals batch computation."""
    print("="*60)
    print("TEST: Incremental vs Batch H Computation")
    print("="*60)
    
    # Create random test data
    torch.manual_seed(42)
    n_samples = 100
    n_features = 256
    
    # Generate random activations
    activations = [torch.randn(32, n_features) for _ in range(n_samples)]
    
    # Method 1: Batch computation
    all_acts = torch.cat(activations, dim=0)  # [n_samples*32, n_features]
    H_batch = (all_acts.t() @ all_acts) / all_acts.shape[0]
    
    # Method 2: Incremental computation (our method)
    H_incremental = torch.zeros(n_features, n_features)
    n_total = 0
    
    for act in activations:
        batch_size = act.shape[0]
        n_total += batch_size
        
        act_t = act.t()
        H_update = act_t @ act_t.t()
        
        H_incremental *= (n_total - batch_size) / n_total
        H_incremental += H_update / n_total
    
    # Compare
    diff = (H_batch - H_incremental).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"Max difference: {max_diff:.2e}")
    print(f"Mean difference: {mean_diff:.2e}")
    print(f"Relative error: {max_diff / H_batch.abs().max().item():.2e}")
    
    if max_diff < 1e-5:
        print("✅ PASS: Incremental computation is correct")
        return True
    else:
        print("❌ FAIL: Significant difference detected!")
        return False


def test_h_matrix_properties():
    """Test properties of our saved H matrices."""
    print("\n" + "="*60)
    print("TEST: H Matrix Properties")
    print("="*60)
    
    calibration_dir = Path("calibration_data")
    if not calibration_dir.exists():
        print("❌ No calibration data found!")
        return False
    
    import json
    with open(calibration_dir / "metadata.json") as f:
        metadata = json.load(f)
    
    issues = []
    
    for name, info in tqdm(list(metadata['layers'].items())[:20], desc="Checking H matrices"):
        H_path = calibration_dir / info['H_file']
        H = torch.load(H_path, map_location='cpu', weights_only=True)
        
        # Squeeze if needed
        while H.dim() > 2:
            H = H.squeeze(-1)
        
        # Check 1: Is H symmetric?
        symmetry_error = (H - H.t()).abs().max().item()
        if symmetry_error > 1e-5:
            issues.append(f"{name}: Not symmetric (error={symmetry_error:.2e})")
        
        # Check 2: Is H positive semi-definite? (all eigenvalues >= 0)
        try:
            eigenvalues = torch.linalg.eigvalsh(H)
            min_eigenvalue = eigenvalues.min().item()
            if min_eigenvalue < -1e-5:
                issues.append(f"{name}: Negative eigenvalue ({min_eigenvalue:.2e})")
        except:
            issues.append(f"{name}: Eigenvalue computation failed")
        
        # Check 3: Any NaN or Inf?
        if torch.isnan(H).any():
            issues.append(f"{name}: Contains NaN")
        if torch.isinf(H).any():
            issues.append(f"{name}: Contains Inf")
        
        # Check 4: Is diagonal reasonable? (not all zeros, not all same)
        diag = H.diag()
        if (diag == 0).all():
            issues.append(f"{name}: All-zero diagonal")
        if (diag == diag[0]).all() and diag.numel() > 1:
            issues.append(f"{name}: Constant diagonal (suspicious)")
    
    if issues:
        print(f"❌ Found {len(issues)} issues:")
        for issue in issues[:10]:
            print(f"  - {issue}")
        return False
    else:
        print("✅ PASS: All H matrices look healthy")
        return True


def test_h_invertibility():
    """Test if H matrices can be inverted (required for GPTQ)."""
    print("\n" + "="*60)
    print("TEST: H Matrix Invertibility")
    print("="*60)
    
    calibration_dir = Path("calibration_data")
    if not calibration_dir.exists():
        print("❌ No calibration data found!")
        return False
    
    import json
    with open(calibration_dir / "metadata.json") as f:
        metadata = json.load(f)
    
    invertible = 0
    not_invertible = 0
    
    for name, info in tqdm(list(metadata['layers'].items()), desc="Testing invertibility"):
        H_path = calibration_dir / info['H_file']
        H = torch.load(H_path, map_location='cpu', weights_only=True)
        
        while H.dim() > 2:
            H = H.squeeze(-1)
        
        H = H.float()
        
        # Add dampening (like we do in quantization)
        damp = 1e-5 * H.diag().mean()
        H_damped = H + damp * torch.eye(H.shape[0])
        
        try:
            L = torch.linalg.cholesky(H_damped)
            H_inv = torch.cholesky_inverse(L)
            invertible += 1
        except:
            not_invertible += 1
    
    print(f"Invertible: {invertible}/{invertible + not_invertible}")
    print(f"Not invertible: {not_invertible}/{invertible + not_invertible}")
    
    if not_invertible > invertible * 0.1:  # More than 10% fail
        print("⚠️ WARNING: Many H matrices are not invertible")
        return False
    else:
        print("✅ PASS: Most H matrices are invertible")
        return True


def main():
    print("="*60)
    print("HESSIAN MATRIX DIAGNOSTICS")
    print("="*60)
    
    results = {}
    
    # Test 1: Incremental vs Batch
    results['incremental_correct'] = test_incremental_vs_batch()
    
    # Test 2: H matrix properties
    results['h_properties_ok'] = test_h_matrix_properties()
    
    # Test 3: Invertibility
    results['h_invertible'] = test_h_invertibility()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test}: {status}")
    
    return all(results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)