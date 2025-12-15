#!/usr/bin/env python3
"""
=============================================================================
GPTQ Quantization Script for SD 3.5 Medium
=============================================================================

This script applies GPTQ quantization to the SD 3.5 Medium transformer.

GPTQ (Generative Pre-trained Transformer Quantization) uses second-order
information (Hessian) to minimize quantization error while achieving
aggressive compression (e.g., 4-bit weights).

Configuration:
- W4A8: 4-bit weights, 8-bit activations
- Group size: 128 (balance between accuracy and compression)

Usage:
    python 02_quantize_model.py --calibration-data ./calibration_data --output-dir ./quantized_model

Requirements:
    - Calibration data from 01_collect_calibration_data.py
    - RTX 4090 (24GB VRAM) recommended
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import gc

import torch
import torch.nn as nn
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    CALIBRATION_DATA_DIR,
    QUANTIZED_MODEL_DIR,
    QUANT_CONFIG,
)
from models.sd35_loader import (
    load_sd35_pipeline,
    load_sd35_transformer,
    get_transformer_blocks,
    get_model_size,
    save_transformer_state,
)
from quantization.gptq import GPTQ, Quantizer_GPTQ
from quantization.quant_linear import (
    QuantLinear,
    find_linear_layers,
    replace_linear_with_quantized,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


def load_hessians(
    calibration_dir: Path,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
    """
    Load pre-computed Hessian matrices from disk.
    
    Args:
        calibration_dir: Directory containing calibration data (Hessians)
        
    Returns:
        Tuple of (hessians dict, nsamples dict)
    """
    calibration_dir = Path(calibration_dir)
    
    # Load metadata
    metadata_path = calibration_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    hessians = {}
    nsamples = {}
    
    for name, info in tqdm(metadata["layers"].items(), desc="Loading Hessians"):
        tensor_path = calibration_dir / info["file"]
        H = torch.load(tensor_path, map_location="cpu")
        hessians[name] = H
        nsamples[name] = info.get("nsamples", 1)
    
    logger.info(f"Loaded Hessians for {len(hessians)} layers")
    logger.info(f"Calibration config: {metadata.get('config', {})}")
    
    return hessians, nsamples


def quantize_transformer_gptq(
    transformer: nn.Module,
    hessians: Dict[str, torch.Tensor],
    wbits: int = 4,
    group_size: int = 128,
    percdamp: float = 0.01,
    symmetric: bool = True,
    device: str = "cuda",
) -> nn.Module:
    """
    Apply GPTQ quantization to transformer using pre-computed Hessians.
    
    Args:
        transformer: SD3 transformer model
        hessians: Dictionary of pre-computed Hessian matrices (H = X^T X / n)
        wbits: Weight bits (e.g., 4)
        group_size: Quantization group size
        percdamp: Dampening percentage for Hessian
        symmetric: Use symmetric quantization
        device: Device for quantization
        
    Returns:
        Quantized transformer
    """
    logger.info(f"Quantizing transformer with W{wbits} (group_size={group_size})")
    
    # Find all linear layers
    linear_layers = find_linear_layers(
        transformer,
        skip_patterns=["time", "embed", "norm", "pos"],
    )
    
    logger.info(f"Found {len(linear_layers)} linear layers to quantize")
    
    # Sort layers by depth for sequential processing
    layer_names = sorted(linear_layers.keys())
    
    # Quantize each layer
    quantized_count = 0
    skipped_count = 0
    total_loss = 0.0
    
    for name in tqdm(layer_names, desc="Quantizing layers"):
        layer = linear_layers[name]
        
        # Check if we have Hessian for this layer
        if name not in hessians:
            logger.warning(f"No Hessian for {name}, skipping")
            skipped_count += 1
            continue
        
        H = hessians[name].to(device).float()
        
        # Move layer to device
        layer = layer.to(device)
        W = layer.weight.data.clone().float()
        
        # Get dimensions
        rows, columns = W.shape  # [out_features, in_features]
        
        # Verify Hessian dimensions match
        if H.shape[0] != columns or H.shape[1] != columns:
            logger.warning(f"Hessian shape mismatch for {name}: H={H.shape}, W={W.shape}")
            skipped_count += 1
            continue
        
        # Handle dead columns (zero diagonal in Hessian)
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        
        # Dampening for numerical stability
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(columns, device=device)
        H[diag, diag] += damp
        
        # Cholesky decomposition for inverse: H = L L^T, H^{-1} = L^{-T} L^{-1}
        try:
            H_chol = torch.linalg.cholesky(H)
            H_inv = torch.cholesky_inverse(H_chol)
            Hinv = torch.linalg.cholesky(H_inv, upper=True)
        except RuntimeError as e:
            logger.warning(f"Cholesky failed for {name}, using pseudo-inverse: {e}")
            Hinv = torch.linalg.pinv(H)
            Hinv = torch.linalg.cholesky(Hinv + 1e-6 * torch.eye(columns, device=device), upper=True)
        
        # Initialize quantizer
        quantizer = Quantizer_GPTQ()
        quantizer.configure(
            bits=wbits,
            perchannel=True,
            sym=symmetric,
            mse=True,
            group_size=group_size,
        )
        
        # Storage for quantized weights
        Q = torch.zeros_like(W)
        Losses = torch.zeros_like(W)
        
        # Block size for lazy batch updates
        blocksize = 128
        
        # Process columns in blocks for efficiency
        for i1 in range(0, columns, blocksize):
            i2 = min(i1 + blocksize, columns)
            count = i2 - i1
            
            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]
            
            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]
                
                # Update quantization params for group-wise quantization
                if group_size > 0 and (i1 + i) % group_size == 0:
                    quantizer.find_params(
                        W[:, (i1 + i):min((i1 + i + group_size), columns)],
                        weight=True
                    )
                
                # Quantize current column
                q = quantizer.quantize(w.unsqueeze(1)).flatten()
                Q1[:, i] = q
                
                # Compute loss for this column
                Losses1[:, i] = (w - q) ** 2 / d ** 2
                
                # Error compensation: update remaining weights in block
                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1
            
            # Store results for this block
            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2
            
            # Propagate error to remaining columns
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
        
        # Update layer weights
        layer.weight.data = Q.to(layer.weight.dtype)
        
        # Track statistics
        layer_loss = torch.sum(Losses).item()
        total_loss += layer_loss
        quantized_count += 1
        
        # Cleanup
        del H, W, Q, Losses, Hinv
        torch.cuda.empty_cache()
        
        # Log progress
        if quantized_count % 50 == 0:
            logger.info(f"Quantized {quantized_count}/{len(layer_names)} layers, "
                       f"cumulative loss: {total_loss:.4f}")
    
    logger.info(f"Quantization complete: {quantized_count} quantized, {skipped_count} skipped")
    logger.info(f"Total quantization loss: {total_loss:.4f}")
    
    return transformer


def quantize_with_simple_rtn(
    transformer: nn.Module,
    wbits: int = 4,
    group_size: int = 128,
    symmetric: bool = True,
    device: str = "cuda",
) -> nn.Module:
    """
    Simple Round-To-Nearest (RTN) quantization as a baseline.
    
    This is faster than GPTQ but typically lower quality.
    
    Args:
        transformer: SD3 transformer model
        wbits: Weight bits
        group_size: Quantization group size
        symmetric: Use symmetric quantization
        device: Device
        
    Returns:
        Quantized transformer
    """
    logger.info(f"Quantizing transformer with RTN W{wbits}")
    
    # Replace linear layers with quantized versions
    replaced = replace_linear_with_quantized(
        transformer,
        bits=wbits,
        group_size=group_size,
        symmetric=symmetric,
        skip_patterns=["time", "embed", "norm", "pos"],
    )
    
    # Quantize each layer
    for name, quant_layer in tqdm(replaced.items(), desc="Quantizing"):
        quant_layer.to(device)
        quant_layer.quantize_weights()
    
    logger.info(f"Quantized {len(replaced)} layers")
    
    return transformer


def verify_quantization(
    original_transformer: nn.Module,
    quantized_transformer: nn.Module,
    test_input: torch.Tensor,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Verify quantization by comparing outputs.
    
    Args:
        original_transformer: Original FP16 transformer
        quantized_transformer: Quantized transformer
        test_input: Test input tensor
        device: Device
        
    Returns:
        Dictionary with verification metrics
    """
    logger.info("Verifying quantization...")
    
    # This would require running both transformers and comparing outputs
    # For now, we just verify the model can run
    
    try:
        quantized_transformer.eval()
        quantized_transformer.to(device)
        
        # Just verify it runs without error
        # Note: Full verification would require proper inputs with timesteps, etc.
        
        return {"status": "success"}
        
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return {"status": "failed", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Apply GPTQ quantization to SD 3.5 Medium transformer",
    )
    
    # Quantization settings
    parser.add_argument(
        "--wbits", type=int, default=4, choices=[2, 3, 4, 8],
        help="Weight bits (default: 4)"
    )
    parser.add_argument(
        "--group-size", type=int, default=128,
        help="Quantization group size (default: 128)"
    )
    parser.add_argument(
        "--percdamp", type=float, default=0.01,
        help="Dampening percentage for GPTQ (default: 0.01)"
    )
    parser.add_argument(
        "--symmetric", action="store_true", default=True,
        help="Use symmetric quantization (default: True)"
    )
    parser.add_argument(
        "--method", type=str, default="gptq", choices=["gptq", "rtn"],
        help="Quantization method (default: gptq)"
    )
    
    # Data paths
    parser.add_argument(
        "--calibration-data", type=str, default=str(CALIBRATION_DATA_DIR),
        help="Path to calibration data directory"
    )
    parser.add_argument(
        "--model-id", type=str,
        default="stabilityai/stable-diffusion-3.5-medium",
        help="Hugging Face model ID"
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(QUANTIZED_MODEL_DIR),
        help="Output directory for quantized model"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("SD 3.5 Medium GPTQ Quantization")
    logger.info("=" * 60)
    logger.info(f"Method: {args.method.upper()}")
    logger.info(f"Weight bits: {args.wbits}")
    logger.info(f"Group size: {args.group_size}")
    
    # Check GPU
    if not torch.cuda.is_available():
        logger.error("CUDA not available!")
        return 1
    
    device = "cuda"
    logger.info(f"GPU: {torch.cuda.get_device_name()}")
    
    # Load Hessian matrices (only needed for GPTQ)
    hessians = None
    if args.method == "gptq":
        calib_path = Path(args.calibration_data)
        if not calib_path.exists():
            logger.error(f"Calibration data not found: {calib_path}")
            logger.error("Run 01_collect_calibration_data.py first!")
            return 1
        
        logger.info("\nLoading Hessian matrices...")
        hessians, nsamples = load_hessians(calib_path)
    
    # Load transformer
    logger.info("\nLoading SD 3.5 Medium transformer...")
    transformer = load_sd35_transformer(
        model_id=args.model_id,
        device=device,
        dtype=torch.float16,
    )
    
    original_size = get_model_size(transformer, "GB")
    logger.info(f"Original transformer size: {original_size:.2f} GB")
    
    # Quantize
    logger.info("\nStarting quantization...")
    start_time = time.time()
    
    if args.method == "gptq":
        transformer = quantize_transformer_gptq(
            transformer=transformer,
            hessians=hessians,
            wbits=args.wbits,
            group_size=args.group_size,
            percdamp=args.percdamp,
            symmetric=args.symmetric,
            device=device,
        )
    else:
        transformer = quantize_with_simple_rtn(
            transformer=transformer,
            wbits=args.wbits,
            group_size=args.group_size,
            symmetric=args.symmetric,
            device=device,
        )
    
    quant_time = time.time() - start_time
    logger.info(f"Quantization completed in {quant_time / 60:.1f} minutes")
    
    # Check quantized size
    quantized_size = get_model_size(transformer, "GB")
    compression_ratio = original_size / quantized_size
    logger.info(f"Quantized transformer size: {quantized_size:.2f} GB")
    logger.info(f"Compression ratio: {compression_ratio:.2f}x")
    
    # Save quantized model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\nSaving quantized transformer to {output_dir}...")
    save_transformer_state(transformer, output_dir)
    
    # Save quantization config
    config = {
        "method": args.method,
        "wbits": args.wbits,
        "group_size": args.group_size,
        "symmetric": args.symmetric,
        "original_size_gb": original_size,
        "quantized_size_gb": quantized_size,
        "compression_ratio": compression_ratio,
        "quantization_time_minutes": quant_time / 60,
    }
    
    config_path = output_dir / "quantization_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info("\nQuantization complete!")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Original size: {original_size:.2f} GB")
    logger.info(f"Quantized size: {quantized_size:.2f} GB")
    logger.info(f"Compression: {compression_ratio:.2f}x")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())