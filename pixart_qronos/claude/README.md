# Qronos-DiT: Qronos Quantization for Diffusion Transformers

This project implements the **Qronos** post-training quantization algorithm adapted for **Diffusion Transformer (DiT)** models, specifically **PixArt-α 512**.

## Overview

Qronos is an advanced post-training quantization method that improves upon GPTQ by:
1. **Explicit error correction**: Uses both original (X) and quantized (X̃) activations
2. **Better first iteration**: Computes G = X̃ᵀX for cross-correlation correction
3. **Efficient implementation**: Uses Cholesky decomposition for fast computation

This adaptation follows the pattern established by Q-DiT (GPTQ for DiT) and extends it with Qronos improvements.

## Key Features

- ✅ **8-bit weight quantization** with float64 calculations for numerical stability
- ✅ **Checkpointing support** for long-running quantization (resumable)
- ✅ **K/V projection exclusion**: Option to skip quantizing attention K and V projections
- ✅ **CLIP score evaluation** for quality assessment
- ✅ **Visual comparison** with FP16 baseline
- ✅ **Peak VRAM tracking**

## Installation

```bash
cd qronos_dit
pip install -r requirements.txt
```

## Usage

### Quick Start

```python
from qronos_dit import PixArtQuantizer

# Initialize quantizer
quantizer = PixArtQuantizer(
    model_id="PixArt-alpha/PixArt-XL-2-512x512",
    bits=8,
    skip_layers=['to_k', 'to_v'],  # Don't quantize K/V projections
)

# Load model
quantizer.load_model()

# Calibration prompts (from COCO or custom)
prompts = ["A photo of a cat", "A beautiful sunset", ...]  # 256 prompts recommended

# Quantize with checkpoints every 4 blocks
stats = quantizer.quantize_full_model(
    prompts=prompts,
    checkpoint_interval=4,
)

# Save quantized model
quantizer.save_quantized_model("./quantized_pixart")
```

### Full Pipeline Script

```bash
# Run full quantization with evaluation
python scripts/run_quantization.py \
    --model_id PixArt-alpha/PixArt-XL-2-512x512 \
    --bits 8 \
    --num_calibration_samples 256 \
    --output_dir ./output \
    --evaluate

# Resume from checkpoint
python scripts/run_quantization.py \
    --resume_from ./checkpoints/checkpoint_block_12 \
    --output_dir ./output
```

### Using COCO Captions

```bash
# Download COCO captions first
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip

# Run with COCO captions
python scripts/run_quantization.py \
    --coco_captions_path ./annotations/captions_val2017.json \
    --num_calibration_samples 256
```

## Algorithm Details

### Qronos vs GPTQ

**GPTQ** (used in Q-DiT):
- Collects Hessian H = X̃ᵀX̃ from quantized model activations
- Error diffusion: w_{t+1} -= (w_t - q_t) / d * H_inv[t, t+1:]

**Qronos** (this implementation):
- Collects both G = X̃ᵀX and H = X̃ᵀX̃
- First iteration: q_1 = Q((G[1,:] @ w - H[1,2:] @ w[2:]) / H[1,1])
- Then updates: w[2:] = H[2:,2:]⁻¹ @ (G[2:,:] @ w_orig - H[2:,1] @ q_1)
- Subsequent iterations: Same as GPTQ (RTN + error diffusion)

### Key Equations

For the first weight column:
```
q₁ = Q((G₁,≥₁ · w - H₁,≥₂ · w₍≥₂₎) / H₁,₁)
w₍≥₂₎ = H₍≥₂,≥₂₎⁻¹ · (G₍≥₂,≥₁₎ · w_orig - H₍≥₂,₁₎ · q₁)
```

For subsequent columns (t ≥ 2):
```
qₜ = Q(wₜ⁽ᵗ⁻¹⁾)
w₍≥ₜ₊₁₎⁽ᵗ⁾ = w₍≥ₜ₊₁₎⁽ᵗ⁻¹⁾ - L₍≥ₜ₊₁,ₜ₎ · (wₜ⁽ᵗ⁻¹⁾ - qₜ) / Lₜ,ₜ
```

Where L is the Cholesky factor of H⁻¹.

## Project Structure

```
qronos_dit/
├── qronos_dit/
│   ├── __init__.py          # Package exports
│   ├── qronos.py             # Core Qronos algorithm
│   ├── quant_utils.py        # Quantization utilities
│   ├── qlinear.py            # Quantized linear layer wrapper
│   ├── pixart_quantizer.py   # PixArt model handler
│   └── evaluation.py         # CLIP score & visualization
├── scripts/
│   └── run_quantization.py   # Main CLI script
├── configs/                  # Configuration files
├── checkpoints/              # Checkpoint storage (created at runtime)
├── requirements.txt
└── README.md
```

## Checkpointing

The quantization process automatically saves checkpoints:

- **Automatic**: Every N blocks (configurable via `--checkpoint_interval`)
- **Manual resume**: Use `--resume_from ./checkpoints/checkpoint_block_X`
- **Final checkpoint**: Saved as `./checkpoints/final`

Checkpoint contents:
- `transformer.pt`: Model state dict
- `metadata.json`: Progress and configuration

## Evaluation Metrics

### CLIP Score
Measures image-text alignment using CLIP embeddings.
- Higher is better (typically 25-35 for good generations)
- Comparison: FP16 baseline vs Quantized

### Peak VRAM
Tracks maximum GPU memory usage during inference.
- Helps verify memory reduction from quantization

### Visual Comparison
Side-by-side images generated from identical prompts and seeds.

## Hardware Requirements

- **GPU**: RTX 4090 (24GB VRAM) or similar
- **RAM**: 64GB recommended
- **Storage**: ~10GB for model and checkpoints

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--bits` | 8 | Quantization bit-width |
| `--sym` | True | Symmetric quantization |
| `--group_size` | -1 | Group size (-1 = per-channel) |
| `--blocksize` | 128 | GPTQ block size |
| `--percdamp` | 0.01 | Hessian damping percentage |
| `--alpha` | 1e-6 | Qronos damping factor |
| `--skip_layers` | [to_k, to_v] | Layers to skip |
| `--checkpoint_interval` | 4 | Blocks between checkpoints |

## References

- **GPTQ Paper**: [Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)
- **Qronos Paper**: [Qronos: Correcting the Past by Shaping the Future in Post-Training Quantization](https://arxiv.org/abs/2505.11695)
- **Q-DiT**: [Q-DiT: Accurate Post-Training Quantization for Diffusion Transformers](https://github.com/Juanerx/Q-DiT)
- **PixArt-α**: [PixArt-α: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis](https://arxiv.org/abs/2310.00426)

## License

This project is for research purposes. See individual component licenses:
- GPTQ: Apache 2.0
- Qronos/Brevitas: BSD-3-Clause
- PixArt-α: Apache 2.0

## Troubleshooting

### Out of Memory
- Reduce `--num_calibration_samples`
- Use checkpointing with smaller `--checkpoint_interval`
- Ensure float64 calculations are on CPU (default)

### Cholesky Decomposition Failed
- Increase `--percdamp` (try 0.05 or 0.1)
- Increase `--alpha` (try 1e-5 or 1e-4)
- Check for zero activations in calibration data

### Poor Quality Results
- Use more calibration samples (256+)
- Use diverse prompts (COCO captions recommended)
- Check that K/V projections are not quantized
