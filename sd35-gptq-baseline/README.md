# SD 3.5 Medium GPTQ Quantization Baseline

## Project Overview

This project establishes a **GPTQ quantization baseline** for Stable Diffusion 3.5 Medium. 
The baseline will be used to compare against the **Qronos quantization adaptation** for diffusion models.

### Why SD 3.5 Medium?
- Uses **MMDiT (Multimodal Diffusion Transformer)** architecture
- Pure transformer-based (no UNet), making it ideal for adapting LLM quantization techniques
- ~2.5B parameters in the transformer, similar scale to mid-sized LLMs
- State-of-the-art image quality

### Quantization Configuration
- **Weights**: 4-bit (W4)
- **Activations**: 8-bit (A8)
- **Method**: GPTQ (post-training quantization using Hessian information)

---

## Project Structure

```
sd35-gptq-baseline/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── config.py                           # Configuration and constants
│
├── models/
│   ├── __init__.py
│   └── sd35_loader.py                  # SD 3.5 Medium loading utilities
│
├── quantization/
│   ├── __init__.py
│   ├── gptq.py                         # Core GPTQ algorithm
│   ├── quant_linear.py                 # Quantized linear layer
│   └── quantizer.py                    # Quantization utilities
│
├── evaluation/
│   ├── __init__.py
│   ├── fid_score.py                    # FID calculation
│   ├── clip_score.py                   # CLIP score calculation
│   └── metrics.py                      # Combined metrics utilities
│
├── scripts/
│   ├── 01_collect_calibration_data.py  # Step 1: Collect calibration data
│   ├── 02_quantize_model.py            # Step 2: Apply GPTQ quantization
│   ├── 03_benchmark.py                 # Step 3: Run full benchmark
│   └── generate_samples.py             # Utility: Generate sample images
│
├── prompts/
│   ├── visual_inspection.txt           # 50 curated prompts for visual comparison
│   └── coco_captions.txt               # COCO captions for FID (downloaded automatically)
│
├── outputs/                            # Generated images (created automatically)
├── calibration_data/                   # Calibration tensors (created automatically)
└── results/                            # Benchmark results (created automatically)
```

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 16 GB | 24 GB (RTX 4090) |
| RAM | 32 GB | 64 GB |
| Storage | 30 GB | 50 GB |

---

## Installation

### Step 1: Create Environment

```bash
conda create -n sd35-baseline python=3.10
conda activate sd35-baseline
```

### Step 2: Install PyTorch with CUDA

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Step 3: Install Dependencies

```bash
cd sd35-gptq-baseline
pip install -r requirements.txt
```

### Step 4: Login to Hugging Face (required for SD 3.5)

```bash
huggingface-cli login
```

You need to accept the [SD 3.5 license](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) on Hugging Face.

---

## Usage

### Quick Start (Full Pipeline)

```bash
# Run all steps sequentially
python scripts/01_collect_calibration_data.py
python scripts/02_quantize_model.py
python scripts/03_benchmark.py
```

### Step-by-Step

#### Step 1: Collect Calibration Data

Generates intermediate activations needed for GPTQ calibration.

```bash
python scripts/01_collect_calibration_data.py \
    --num_samples 256 \
    --batch_size 4 \
    --output_dir ./calibration_data
```

#### Step 2: Quantize Model

Applies GPTQ to the transformer blocks.

```bash
python scripts/02_quantize_model.py \
    --calibration_data ./calibration_data \
    --wbits 4 \
    --abits 8 \
    --output_dir ./quantized_model
```

#### Step 3: Benchmark

Runs full evaluation comparing FP16 vs quantized model.

```bash
python scripts/03_benchmark.py \
    --num_images 5000 \
    --batch_size 4 \
    --output_dir ./results
```

### Generate Visual Samples Only

```bash
python scripts/generate_samples.py \
    --prompts_file ./prompts/visual_inspection.txt \
    --output_dir ./outputs/visual_samples \
    --compare  # Generates both FP16 and quantized side-by-side
```

---

## Metrics Explained

### FID (Fréchet Inception Distance)
- **What**: Statistical distance between generated and real image distributions
- **Range**: 0 to ∞ (lower is better)
- **Target**: < 30 for good quality
- **Reference**: MS-COCO 2017 validation images

### CLIP Score
- **What**: Cosine similarity between image and text CLIP embeddings
- **Range**: 0 to 1 (higher is better)
- **Target**: > 0.28 for good alignment
- **Note**: Measures how well image matches the prompt

### Model Size
- **FP16**: ~5 GB for transformer alone
- **W4A8**: ~1.5-2 GB (expected ~3x compression)

### Peak VRAM
- **FP16**: ~12-15 GB during inference
- **W4A8**: ~6-8 GB (expected)

---

## Expected Results

| Metric | FP16 Baseline | GPTQ W4A8 | Notes |
|--------|---------------|-----------|-------|
| FID | ~25-30 | ~28-35 | Slight degradation expected |
| CLIP Score | ~0.30 | ~0.29 | Minimal change expected |
| Model Size | ~5 GB | ~1.5 GB | ~3x compression |
| Peak VRAM | ~14 GB | ~7 GB | ~2x reduction |
| Inference Time | 1x | 0.8-1.2x | Depends on kernel support |

---

## Development Log

### Phase 1: Baseline Establishment (Current)
- [x] Project structure setup
- [ ] SD 3.5 Medium loading utilities
- [ ] Calibration data collection
- [ ] GPTQ implementation for MMDiT
- [ ] Benchmarking pipeline
- [ ] Run baseline experiments

### Phase 2: Qronos Adaptation (Next)
- [ ] Study Qronos algorithm
- [ ] Adapt for MMDiT architecture
- [ ] Implement and benchmark
- [ ] Compare against GPTQ baseline

---

## References

- [GPTQ Paper](https://arxiv.org/abs/2210.17323)
- [Q-DiT: Quantizing Diffusion Transformers](https://arxiv.org/abs/2406.17343)
- [Stable Diffusion 3.5](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium)
- [MMDiT Architecture](https://arxiv.org/abs/2403.03206)

---

## License

Research purposes only. Respect the licenses of underlying models and libraries.
