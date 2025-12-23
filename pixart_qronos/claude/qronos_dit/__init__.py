"""
Qronos-DiT: Qronos Quantization Algorithm for Diffusion Transformers

This package implements the Qronos post-training quantization algorithm
adapted for Diffusion Transformer (DiT) models, specifically PixArt-α.

The Qronos algorithm is from:
"Qronos: Correcting the Past by Shaping the Future in Post-Training Quantization"

Key features:
- Explicit error correction from both original and quantized activations
- Efficient Cholesky-based implementation
- Checkpoint support for long-running quantization
- Support for skipping specific layers (K/V projections)

Usage:
    from qronos_dit import PixArtQuantizer
    
    quantizer = PixArtQuantizer(
        model_id="PixArt-alpha/PixArt-XL-2-512x512",
        bits=8,
        skip_layers=['to_k', 'to_v'],
    )
    quantizer.load_model()
    quantizer.quantize_full_model(prompts)
    quantizer.save_quantized_model("./quantized_model")
"""

from .qronos import QronosDiT, QronosDiTSimple
from .quant_utils import Quantizer, quantize_tensor, quantize_gptq
from .qlinear import QLinearLayer, find_linear_layers, replace_linear_with_qlinear
from .pixart_quantizer import PixArtQuantizer
from .evaluation import (
    Evaluator,
    CLIPScorer,
    ImageGenerator,
    measure_peak_vram,
    reset_vram_stats,
    load_coco_captions,
    get_default_eval_prompts,
)

__version__ = "0.1.0"
__all__ = [
    # Core Qronos
    "QronosDiT",
    "QronosDiTSimple",
    # Quantization utilities
    "Quantizer",
    "quantize_tensor",
    "quantize_gptq",
    # Linear layer handling
    "QLinearLayer",
    "find_linear_layers",
    "replace_linear_with_qlinear",
    # PixArt quantizer
    "PixArtQuantizer",
    # Evaluation
    "Evaluator",
    "CLIPScorer",
    "ImageGenerator",
    "measure_peak_vram",
    "reset_vram_stats",
    "load_coco_captions",
    "get_default_eval_prompts",
]
