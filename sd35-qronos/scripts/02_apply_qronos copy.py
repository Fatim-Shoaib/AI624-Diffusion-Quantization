#!/usr/bin/env python3
"""
=============================================================================
Step 2: Apply Qronos Quantization to SD 3.5 Medium
=============================================================================

This script applies the Qronos quantization algorithm to the SD 3.5 Medium
transformer using the calibration data collected in Step 1.

Qronos Key Features:
- Uses both H (Hessian from quantized inputs) and G (cross-covariance) matrices
- Two-phase quantization: special first column handling + Cholesky updates
- Better error correction than GPTQ for same bit-width

Usage:
    python 02_apply_qronos.py --calibration-data ./calibration_data

Requirements:
    - Calibration data from 01_collect_calibration.py
    - RTX 4090 (24GB VRAM) recommended
    - ~30-60 minutes for full quantization
"""

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    MODEL_CONFIG,
    QRONOS_CONFIG,
    CALIBRATION_DIR,
    QUANTIZED_MODEL_DIR,
)
from models.sd35_loader import (
    load_sd35_transformer,
    get_model_size,
    get_transformer_linear_layers,
    save_quantized_transformer,
)
from quantization.qronos_core import QronosQuantizer
from quantization.calibration import CalibrationDataCollector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


def apply_qronos_to_transformer(
    transformer: nn.Module,
    calibration_data: dict,
    weight_bits: int = 4,
    group_size: int = 128,
    symmetric: bool = True,
    percdamp: float = 1e-5,
    num_blocks: int = 100,
    act_order: bool = False,
    skip_layers: list = None,
    output_dir: Path = None,
) -> nn.Module:
    """
    Apply Qronos quantization to all linear layers in the transformer.
    
    Args:
        transformer: SD3Transformer2DModel to quantize
        calibration_data: Dict mapping layer names to (H, G) tuples
        weight_bits: Number of bits for weight quantization
        group_size: Group size for per-group quantization
        symmetric: Use symmetric quantization
        percdamp: Dampening factor (uses spectral norm)
        num_blocks: Number of sub-blocks for Cholesky updates
        act_order: Whether to use activation ordering
        skip_layers: Layer name patterns to skip
        output_dir: Directory for checkpoints
        
    Returns:
        Quantized transformer
    """
    skip_layers = skip_layers or ['time_embed', 'label_embed', 'proj_out', 'pos_embed']
    
    device = next(transformer.parameters()).device
    
    # Get all linear layers
    linear_layers = get_transformer_linear_layers(transformer, skip_layers)
    logger.info(f"Found {len(linear_layers)} linear layers to quantize")
    
    # Check which layers have calibration data
    layers_with_data = set(calibration_data.keys())
    layers_to_quantize = set(linear_layers.keys())
    
    missing_data = layers_to_quantize - layers_with_data
    if missing_data:
        logger.warning(f"Missing calibration data for {len(missing_data)} layers:")
        for name in list(missing_data)[:5]:
            logger.warning(f"  - {name}")
        if len(missing_data) > 5:
            logger.warning(f"  ... and {len(missing_data) - 5} more")
    
    # Apply Qronos to each layer
    quantized_count = 0
    failed_count = 0
    skipped_count = 0
    
    # Checkpointing setup
    checkpoint_dir = Path(output_dir) if output_dir else None
    checkpoint_file = checkpoint_dir / "quantization_checkpoint.json" if checkpoint_dir else None
    completed_layers = set()
    
    if checkpoint_file and checkpoint_file.exists():
        with open(checkpoint_file) as f:
            checkpoint_data = json.load(f)
            completed_layers = set(checkpoint_data.get('completed_layers', []))
            logger.info(f"Resuming from checkpoint: {len(completed_layers)} layers already done")
    
    layer_names = list(linear_layers.keys())
    for idx, name in enumerate(tqdm(layer_names, desc="Applying Qronos")):
        layer = linear_layers[name]
        
        # Skip already completed layers
        if name in completed_layers:
            continue
        
        if name not in calibration_data:
            logger.debug(f"Skipping {name}: no calibration data")
            skipped_count += 1
            continue
        
        H, G = calibration_data[name]
        
        # Print progress every 20 layers
        if idx % 20 == 0:
            print(f"\n[Layer {idx+1}/{len(layer_names)}] {name}")
            print(f"  H shape: {H.shape}, G shape: {G.shape}")
        
        # Create quantizer
        quantizer = QronosQuantizer(
            layer=layer,
            layer_name=name,
            weight_bits=weight_bits,
            group_size=group_size,
            symmetric=symmetric,
            percdamp=percdamp,
            num_blocks=num_blocks,
            act_order=act_order,
        )
        
        # Set the covariance matrices with dimension check
        H_loaded = H.to(device)
        G_loaded = G.to(device)
        
        # Ensure matrices are 2D (squeeze any extra dimensions)
        if H_loaded.dim() > 2:
            H_loaded = H_loaded.squeeze()
        if G_loaded.dim() > 2:
            G_loaded = G_loaded.squeeze()
        
        # Verify dimensions match layer
        expected_size = layer.in_features
        if H_loaded.shape[0] != expected_size or H_loaded.shape[1] != expected_size:
            logger.warning(f"Skipping {name}: H shape {H_loaded.shape} doesn't match in_features {expected_size}")
            continue
        if G_loaded.shape[0] != expected_size or G_loaded.shape[1] != expected_size:
            logger.warning(f"Skipping {name}: G shape {G_loaded.shape} doesn't match in_features {expected_size}")
            continue
            
        quantizer.H = H_loaded
        quantizer.G = G_loaded
        quantizer.nsamples = 1  # Already aggregated
        
        try:
            # Debug: print tensor shapes before quantization
            print(f"  Attempting quantization...")
            print(f"    quantizer.H shape: {quantizer.H.shape}")
            print(f"    quantizer.G shape: {quantizer.G.shape}")
            print(f"    layer weight shape: {layer.weight.shape}")
            
            quantizer.apply()
            quantized_count += 1
            completed_layers.add(name)
            
            # Save checkpoint every 50 layers
            if checkpoint_file and quantized_count % 50 == 0:
                with open(checkpoint_file, 'w') as f:
                    json.dump({'completed_layers': list(completed_layers)}, f)
                print(f"\n  Checkpoint saved: {quantized_count} layers quantized")
                
        except Exception as e:
            logger.warning(f"Failed to quantize {name}: {e}")
            failed_count += 1
        
        # Clear GPU memory
        if hasattr(quantizer, 'H'):
            del quantizer.H
        if hasattr(quantizer, 'G'):
            del quantizer.G
        torch.cuda.empty_cache()
    
    logger.info(f"Quantized {quantized_count} layers, {failed_count} failed, {skipped_count} skipped")
    
    # Clean up checkpoint file on success
    if checkpoint_file and checkpoint_file.exists() and failed_count == 0:
        checkpoint_file.unlink()
    
    return transformer


def main():
    parser = argparse.ArgumentParser(description="Apply Qronos quantization")
    parser.add_argument(
        "--calibration-data", type=str, default=str(CALIBRATION_DIR),
        help="Path to calibration data directory"
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(QUANTIZED_MODEL_DIR),
        help="Output directory for quantized model"
    )
    parser.add_argument(
        "--model-id", type=str, default=MODEL_CONFIG.model_id,
        help="Hugging Face model ID"
    )
    parser.add_argument(
        "--wbits", type=int, default=QRONOS_CONFIG.weight_bits,
        help=f"Weight bit-width (default: {QRONOS_CONFIG.weight_bits})"
    )
    parser.add_argument(
        "--group-size", type=int, default=QRONOS_CONFIG.group_size,
        help=f"Quantization group size (default: {QRONOS_CONFIG.group_size})"
    )
    parser.add_argument(
        "--percdamp", type=float, default=QRONOS_CONFIG.percdamp,
        help=f"Dampening factor (default: {QRONOS_CONFIG.percdamp})"
    )
    parser.add_argument(
        "--num-blocks", type=int, default=QRONOS_CONFIG.num_blocks,
        help=f"Number of sub-blocks (default: {QRONOS_CONFIG.num_blocks})"
    )
    parser.add_argument(
        "--symmetric", action="store_true", default=QRONOS_CONFIG.weight_symmetric,
        help="Use symmetric quantization"
    )
    parser.add_argument(
        "--act-order", action="store_true", default=QRONOS_CONFIG.act_order,
        help="Use activation ordering"
    )
    
    args = parser.parse_args()
    
    # Check CUDA
    if not torch.cuda.is_available():
        logger.error("CUDA not available!")
        return 1
    
    device = "cuda"
    logger.info(f"GPU: {torch.cuda.get_device_name()}")
    
    # Check calibration data exists
    calib_path = Path(args.calibration_data)
    if not calib_path.exists():
        logger.error(f"Calibration data not found: {calib_path}")
        logger.error("Run 01_collect_calibration.py first!")
        return 1
    
    # Load calibration data
    logger.info("\nLoading calibration data...")
    calibration_data = CalibrationDataCollector.load_calibration_data(calib_path)
    logger.info(f"Loaded calibration data for {len(calibration_data)} layers")
    
    # Load transformer
    logger.info("\nLoading SD 3.5 Medium transformer...")
    transformer = load_sd35_transformer(
        model_id=args.model_id,
        device=device,
        dtype=torch.float16,
    )
    
    original_size = get_model_size(transformer, "GB")
    logger.info(f"Original transformer size: {original_size:.2f} GB")
    
    # Apply Qronos quantization
    logger.info("\nApplying Qronos quantization...")
    logger.info(f"  Weight bits: {args.wbits}")
    logger.info(f"  Group size: {args.group_size}")
    logger.info(f"  Dampening: {args.percdamp}")
    logger.info(f"  Symmetric: {args.symmetric}")
    
    start_time = time.time()
    
    transformer = apply_qronos_to_transformer(
        transformer=transformer,
        calibration_data=calibration_data,
        weight_bits=args.wbits,
        group_size=args.group_size,
        symmetric=args.symmetric,
        percdamp=args.percdamp,
        num_blocks=args.num_blocks,
        act_order=args.act_order,
        output_dir=args.output_dir,
    )
    
    quant_time = time.time() - start_time
    logger.info(f"\nQuantization completed in {quant_time / 60:.1f} minutes")
    
    # Check quantized size
    quantized_size = get_model_size(transformer, "GB")
    compression_ratio = original_size / quantized_size if quantized_size > 0 else 0
    logger.info(f"Quantized transformer size: {quantized_size:.2f} GB")
    logger.info(f"Compression ratio: {compression_ratio:.2f}x")
    
    # Save quantized model
    output_dir = Path(args.output_dir)
    logger.info(f"\nSaving quantized transformer to {output_dir}...")
    
    quantization_config = {
        "method": "qronos",
        "weight_bits": args.wbits,
        "group_size": args.group_size,
        "symmetric": args.symmetric,
        "percdamp": args.percdamp,
        "num_blocks": args.num_blocks,
        "act_order": args.act_order,
        "original_size_gb": original_size,
        "quantized_size_gb": quantized_size,
        "compression_ratio": compression_ratio,
        "quantization_time_minutes": quant_time / 60,
    }
    
    save_quantized_transformer(
        transformer=transformer,
        output_dir=output_dir,
        quantization_config=quantization_config,
    )
    
    logger.info("\nQronos quantization complete!")
    logger.info(f"Quantized model saved to: {output_dir}")
    logger.info("\nNext step: Run 03_benchmark.py to evaluate the quantized model")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())