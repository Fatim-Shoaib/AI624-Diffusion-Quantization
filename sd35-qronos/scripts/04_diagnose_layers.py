#!/usr/bin/env python3
"""
=============================================================================
Diagnostic Script: Find Which Layers Break Quantization
=============================================================================

This script quantizes ONE layer at a time and generates test images to identify
which specific layers cause the black image problem.

Strategy:
1. Load FP16 model
2. For each layer:
   a. Quantize ONLY that layer
   b. Generate 1 test image
   c. Compute CLIP score
   d. Reset to FP16
3. Report which layers cause the biggest quality drops

Usage:
    python scripts/04_diagnose_layers.py --num-layers 50
"""

import argparse
import gc
import json
import logging
import sys
import copy
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MODEL_CONFIG, DEFAULT_VISUAL_PROMPTS
from models.sd35_loader import load_sd35_pipeline, get_model_size

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


def simple_quantize_layer(layer: nn.Linear, bits: int = 4, group_size: int = 128):
    """
    Apply simple RTN quantization to a single layer.
    Returns the original weights so we can restore them.
    """
    original_weight = layer.weight.data.clone()
    
    weight = layer.weight.data.float()
    out_features, in_features = weight.shape
    
    if group_size > 0 and in_features % group_size == 0:
        # Per-group quantization
        num_groups = in_features // group_size
        weight_grouped = weight.view(out_features, num_groups, group_size)
        
        # Compute scale per group
        max_val = weight_grouped.abs().amax(dim=-1, keepdim=True)
        scale = max_val / (2 ** (bits - 1) - 1)
        scale = torch.clamp(scale, min=1e-8)
        
        # Quantize and dequantize
        qmin = -(2 ** (bits - 1))
        qmax = 2 ** (bits - 1) - 1
        q = torch.clamp(torch.round(weight_grouped / scale), qmin, qmax)
        weight_dequant = (q * scale).view(out_features, in_features)
    else:
        # Per-channel quantization
        max_val = weight.abs().amax(dim=-1, keepdim=True)
        scale = max_val / (2 ** (bits - 1) - 1)
        scale = torch.clamp(scale, min=1e-8)
        
        qmin = -(2 ** (bits - 1))
        qmax = 2 ** (bits - 1) - 1
        q = torch.clamp(torch.round(weight / scale), qmin, qmax)
        weight_dequant = q * scale
    
    layer.weight.data = weight_dequant.to(original_weight.dtype)
    
    return original_weight


def restore_layer(layer: nn.Linear, original_weight: torch.Tensor):
    """Restore layer to original weights."""
    layer.weight.data = original_weight


def generate_test_image(pipe, prompt: str, seed: int = 42):
    """Generate a single test image."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    
    with torch.no_grad():
        result = pipe(
            prompt=prompt,
            height=512,  # Smaller for faster testing
            width=512,
            num_inference_steps=20,  # Fewer steps for faster testing
            guidance_scale=4.5,
            generator=generator,
        )
    
    return result.images[0]


def compute_clip_score(image: Image.Image, prompt: str, clip_model, clip_preprocess, tokenizer, device):
    """Compute CLIP score for a single image."""
    import torch
    
    # Process image
    image_tensor = clip_preprocess(image).unsqueeze(0).to(device)
    
    # Process text
    text_tokens = tokenizer([prompt]).to(device)
    
    with torch.no_grad():
        image_features = clip_model.encode_image(image_tensor)
        text_features = clip_model.encode_text(text_tokens)
        
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        similarity = (image_features @ text_features.T).item()
    
    return similarity


def check_for_nan_inf(tensor: torch.Tensor, name: str) -> bool:
    """Check if tensor contains NaN or Inf values."""
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    if has_nan or has_inf:
        logger.warning(f"  {name}: NaN={has_nan}, Inf={has_inf}")
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Diagnose layer-by-layer quantization")
    parser.add_argument("--num-layers", type=int, default=50, help="Number of layers to test")
    parser.add_argument("--skip-clip", action="store_true", help="Skip CLIP score computation")
    parser.add_argument("--bits", type=int, default=4, help="Quantization bits")
    parser.add_argument("--group-size", type=int, default=128, help="Group size")
    parser.add_argument("--output-dir", type=str, default="diagnostic_results", help="Output directory")
    
    args = parser.parse_args()
    
    device = "cuda"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load CLIP model if needed
    clip_model = None
    if not args.skip_clip:
        try:
            import open_clip
            clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k"
            )
            clip_model = clip_model.to(device).eval()
            tokenizer = open_clip.get_tokenizer("ViT-B-32")
            logger.info("CLIP model loaded")
        except Exception as e:
            logger.warning(f"Could not load CLIP: {e}")
            args.skip_clip = True
    
    # Load pipeline
    logger.info("Loading SD 3.5 Medium pipeline...")
    pipe = load_sd35_pipeline(device=device, dtype=torch.float16)
    
    # Get test prompt
    test_prompt = DEFAULT_VISUAL_PROMPTS[0]
    logger.info(f"Test prompt: {test_prompt[:50]}...")
    
    # Generate baseline image
    logger.info("\n=== Generating FP16 Baseline ===")
    baseline_image = generate_test_image(pipe, test_prompt)
    baseline_image.save(output_dir / "baseline_fp16.png")
    
    baseline_clip = 0.0
    if not args.skip_clip:
        baseline_clip = compute_clip_score(
            baseline_image, test_prompt, clip_model, clip_preprocess, tokenizer, device
        )
        logger.info(f"Baseline CLIP score: {baseline_clip:.4f}")
    
    # Get all linear layers
    linear_layers = {}
    for name, module in pipe.transformer.named_modules():
        if isinstance(module, nn.Linear):
            linear_layers[name] = module
    
    logger.info(f"\nFound {len(linear_layers)} linear layers")
    
    # Test each layer
    results = []
    layers_to_test = list(linear_layers.items())[:args.num_layers]
    
    logger.info(f"\n=== Testing {len(layers_to_test)} layers ===\n")
    
    for idx, (name, layer) in enumerate(tqdm(layers_to_test, desc="Testing layers")):
        result = {
            "layer_name": name,
            "in_features": layer.in_features,
            "out_features": layer.out_features,
        }
        
        try:
            # Check original weights
            orig_weight = layer.weight.data
            result["orig_has_nan"] = torch.isnan(orig_weight).any().item()
            result["orig_has_inf"] = torch.isinf(orig_weight).any().item()
            result["orig_max"] = orig_weight.abs().max().item()
            result["orig_mean"] = orig_weight.abs().mean().item()
            
            # Quantize this layer
            original_weight = simple_quantize_layer(layer, args.bits, args.group_size)
            
            # Check quantized weights
            quant_weight = layer.weight.data
            result["quant_has_nan"] = torch.isnan(quant_weight).any().item()
            result["quant_has_inf"] = torch.isinf(quant_weight).any().item()
            result["quant_max"] = quant_weight.abs().max().item()
            result["quant_mean"] = quant_weight.abs().mean().item()
            
            # Generate image with this layer quantized
            test_image = generate_test_image(pipe, test_prompt)
            
            # Check if image is black
            img_array = torch.tensor(list(test_image.getdata())).float()
            result["image_mean"] = img_array.mean().item()
            result["image_std"] = img_array.std().item()
            result["is_black"] = result["image_mean"] < 10  # Nearly black
            
            # Compute CLIP score
            if not args.skip_clip:
                clip_score = compute_clip_score(
                    test_image, test_prompt, clip_model, clip_preprocess, tokenizer, device
                )
                result["clip_score"] = clip_score
                result["clip_drop"] = baseline_clip - clip_score
            
            # Save image if it's problematic
            if result["is_black"] or result.get("clip_drop", 0) > 0.1:
                test_image.save(output_dir / f"layer_{idx:03d}_{name.replace('.', '_')}.png")
                logger.info(f"\n  ⚠️ PROBLEMATIC: {name}")
                logger.info(f"     Image mean: {result['image_mean']:.1f}, CLIP: {result.get('clip_score', 'N/A')}")
            
            # Restore original weights
            restore_layer(layer, original_weight)
            
            result["status"] = "ok"
            
        except Exception as e:
            result["status"] = f"error: {str(e)}"
            logger.error(f"\n  ❌ ERROR on {name}: {e}")
            # Make sure to restore
            try:
                restore_layer(layer, original_weight)
            except:
                pass
        
        results.append(result)
        
        # Clear cache
        torch.cuda.empty_cache()
    
    # Save results
    results_file = output_dir / "layer_diagnosis.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("DIAGNOSIS SUMMARY")
    logger.info("="*60)
    
    problematic = [r for r in results if r.get("is_black", False)]
    nan_layers = [r for r in results if r.get("quant_has_nan", False)]
    inf_layers = [r for r in results if r.get("quant_has_inf", False)]
    
    logger.info(f"Layers tested: {len(results)}")
    logger.info(f"Layers causing black images: {len(problematic)}")
    logger.info(f"Layers with NaN after quantization: {len(nan_layers)}")
    logger.info(f"Layers with Inf after quantization: {len(inf_layers)}")
    
    if problematic:
        logger.info("\nProblematic layers (cause black images):")
        for r in problematic[:10]:
            logger.info(f"  - {r['layer_name']}")
    
    if nan_layers:
        logger.info("\nLayers with NaN:")
        for r in nan_layers[:10]:
            logger.info(f"  - {r['layer_name']}")
    
    # Find layers with biggest CLIP drops
    if not args.skip_clip:
        sorted_by_clip = sorted(results, key=lambda x: x.get("clip_drop", 0), reverse=True)
        logger.info("\nTop 10 layers by CLIP score drop:")
        for r in sorted_by_clip[:10]:
            logger.info(f"  - {r['layer_name']}: drop={r.get('clip_drop', 0):.4f}")
    
    logger.info(f"\nFull results saved to: {results_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())