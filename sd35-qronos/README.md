# Qronos Quantization for Stable Diffusion 3.5 Medium

## Project Overview

This project adapts **Qronos**, a state-of-the-art post-training quantization algorithm originally designed for LLMs, to work with **Stable Diffusion 3.5 Medium's MMDiT architecture**.

### What is Qronos?

Qronos is a sequential rounding algorithm that improves upon GPTQ and GPFQ by:
1. **Error Correction**: Explicitly corrects errors from both weight AND activation quantization
2. **Error Diffusion**: Propagates quantization error into future unquantized weights
3. **Previous Layer Awareness**: Accounts for errors introduced by quantizing previous layers

| Method | Weight Error Correction | Activation Error Correction | Error Diffusion | Previous Layer Error |
|--------|------------------------|----------------------------|-----------------|---------------------|
| **GPTQ** | ✅ | ❌ | ✅ | ❌ |
| **GPFQ** | ✅ | ✅ | ❌ | ✅ |
| **Qronos** | ✅ | ✅ | ✅ | ✅ |

### Reference
- [Qronos Paper](https://arxiv.org/abs/2505.11695): "Correcting the Past by Shaping the Future... in Post-Training Quantization"
- [Brevitas Implementation](https://github.com/Xilinx/brevitas)

---

## Configuration

### Quantization Settings (Same as GPTQ Baseline)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Weight bits | 4 | W4 |
| Activation bits | 8 | A8 |
| Group size | 128 | Per-group quantization |
| Calibration samples | 256 | Diverse prompts |
| Target | Transformer only | Text encoders & VAE stay FP16 |

### Qronos-Specific Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `percdamp` | 1e-5 | Regularization (uses spectral norm) |
| `num_blocks` | 100 | Sub-blocks for Cholesky updates |

---

## Project Structure

```
sd35-qronos/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── config.py                           # Configuration (shared with baseline)
│
├── models/
│   ├── __init__.py
│   └── sd35_loader.py                  # SD 3.5 loading utilities
│
├── quantization/
│   ├── __init__.py
│   ├── qronos_core.py                  # Core Qronos algorithm
│   ├── qronos_linear.py                # Quantized linear layer
│   └── calibration.py                  # Calibration data collection
│
├── scripts/
│   ├── 01_collect_calibration.py       # Collect H and G matrices
│   ├── 02_apply_qronos.py              # Apply Qronos quantization
│   └── 03_benchmark.py                 # Benchmark (reuse metrics)
│
├── evaluation/                         # Reuse from GPTQ baseline
│   ├── fid_score.py
│   ├── clip_score.py
│   └── metrics.py
│
└── prompts/
    └── visual_inspection.txt           # 50 diverse prompts
```

---

## Usage

### Step 1: Collect Calibration Data
```bash
python scripts/01_collect_calibration.py --num-samples 256
```

### Step 2: Apply Qronos Quantization
```bash
python scripts/02_apply_qronos.py --wbits 4 --group-size 128
```

### Step 3: Benchmark
```bash
python scripts/03_benchmark.py --num-images 5000
```

---

## Algorithm Details

### Phase 1: First Column (Special Handling)

The first column requires computing the optimal quantized weight using both
the original float activations (X) and quantized activations (X̃):

```
q₁ = Q((Gw - Uhv) / H₁₁)
```

Where:
- `G = X̃ᵀX` (cross-covariance)
- `H = X̃ᵀX̃` (Hessian from quantized inputs)
- `Uh = triu(H, 1)` (upper triangular)

### Phase 2: Subsequent Columns (Cholesky Updates)

For columns t ≥ 2, Qronos simplifies to round-to-nearest followed by
error diffusion using the Cholesky decomposition:

```python
for t in range(1, columns):
    q[t] = Q(w[t])  # Round-to-nearest
    error = (w[t] - q[t]) / L[t,t]
    w[t+1:] -= error * L[t+1:, t]  # Diffuse error
```

Where `H⁻¹ = LLᵀ` is the Cholesky decomposition.

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 16 GB | 24 GB (RTX 4090) |
| RAM | 32 GB | 64 GB |
| Storage | 30 GB | 50 GB |

---

## Expected Results

| Metric | FP16 | GPTQ W4A8 | Qronos W4A8 |
|--------|------|-----------|-------------|
| FID | ~25-30 | ~30-40 | ~28-35 (expected) |
| CLIP Score | ~0.30 | ~0.28-0.30 | ~0.29-0.30 (expected) |
| Model Size | ~5 GB | ~1.5 GB | ~1.5 GB |
| Peak VRAM | ~14 GB | ~7-9 GB | ~7-9 GB |

Qronos is expected to achieve better quality (lower FID, higher CLIP) than GPTQ
at the same bit-width due to its improved error correction mechanism.

---

## License

Research purposes only.
