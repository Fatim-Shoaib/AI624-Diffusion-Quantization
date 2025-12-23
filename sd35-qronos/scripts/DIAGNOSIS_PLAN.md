# Comprehensive Quantization Diagnosis Plan

## Overview

Our Qronos quantization produces **black images** while simple RTN produces **visible (blurry) images**. This document outlines a systematic diagnosis plan to identify the root cause.

## Key Findings So Far

From `layer_diagnosis.json`:
- **NO single layer causes black images** when quantized alone
- All 50 tested layers produce visible images with CLIP ~0.40
- No NaN or Inf in quantized weights
- **Conclusion: The issue is from COMBINING multiple quantized layers**

---

## Hypotheses and Diagnostic Scripts

### Hypothesis 1: Progressive Layer Accumulation
**Script:** `diag_01_progressive_layers.py`

**Question:** At what point does quantizing N layers together cause black images?

**Method:**
1. Quantize 1 layer → generate image
2. Quantize 5 layers → generate image
3. Quantize 10, 20, 50, 100... layers
4. Find the "tipping point"

**Expected Outcome:**
- If black appears at N=50, we know the threshold
- Helps identify if specific layer combinations are problematic

**Run:**
```bash
python scripts/diag_01_progressive_layers.py
```

---

### Hypothesis 2: Hessian Computation Bug
**Script:** `diag_02_verify_hessian.py`

**Question:** Is our incremental H matrix computation mathematically correct?

**Method:**
1. Compare incremental vs batch H computation on synthetic data
2. Check H matrix properties (symmetry, positive semi-definiteness)
3. Verify H matrices can be inverted

**Expected Outcome:**
- Confirms if calibration data is correct
- Identifies numerical issues in H computation

**Run:**
```bash
python scripts/diag_02_verify_hessian.py
```

---

### Hypothesis 3: Cross-Timestep Error Accumulation
**Script:** `diag_03_timestep_accumulation.py`

**Question:** Do quantization errors compound exponentially across the 28 denoising timesteps?

**Method:**
1. Run FP16 inference, save latents at each timestep
2. Run quantized inference, save latents at each timestep
3. Measure divergence at each step
4. Check if error grows exponentially

**Expected Outcome:**
- If divergence grows 10x+ from early to late timesteps → confirms timestep accumulation
- This would explain why single-layer tests pass but full model fails

**Run:**
```bash
python scripts/diag_03_timestep_accumulation.py
```

---

### Hypothesis 4: Our GPTQ Implementation Bug
**Script:** `diag_04_rtn_vs_gptq.py`

**Question:** Does simple RTN work while our GPTQ-style implementation fails?

**Method:**
1. Apply simple RTN (no H matrix, no error diffusion) → generate image
2. Apply our GPTQ-style (with H matrix and error diffusion) → generate image
3. Compare results

**Expected Outcome:**
- If RTN works but GPTQ fails → bug in error diffusion logic
- If both work → issue is elsewhere

**Run:**
```bash
python scripts/diag_04_rtn_vs_gptq.py
```

---

### Hypothesis 5: Specific Layer Groups Are Problematic
**Script:** `diag_05_layer_groups.py`

**Question:** Which layer TYPES cause the most damage when quantized together?

**Method:**
Test each group in isolation:
1. All attention Q layers
2. All attention K layers
3. All attention V layers
4. All attention output layers
5. All FFN up-projections
6. All FFN down-projections
7. All norm layers
8. All embedding layers

**Expected Outcome:**
- Identifies which layer types should be skipped
- May reveal that certain combinations are toxic

**Run:**
```bash
python scripts/diag_05_layer_groups.py
```

---

### Hypothesis 6: Weight Distribution Issues
**Script:** `diag_06_weight_stats.py`

**Question:** Do some layers have weight distributions incompatible with 4-bit?

**Method:**
1. Compute statistics: min, max, mean, std, percentiles
2. Compute dynamic range (max/mean)
3. Compute quantization error per layer
4. Identify outliers

**Expected Outcome:**
- Layers with high dynamic range (>100) may need special handling
- Layers with high quantization error (>50%) should be skipped

**Run:**
```bash
python scripts/diag_06_weight_stats.py
```

---

### Hypothesis 7: Save/Load Corruption
**Script:** `diag_07_save_load.py`

**Question:** Does saving and loading quantized weights corrupt them?

**Method:**
1. Quantize in memory → generate image (control)
2. Quantize → save → load → generate image (test)
3. Compare weights bit-for-bit

**Expected Outcome:**
- If in-memory works but loaded fails → save/load bug
- If both work/fail equally → not a save/load issue

**Run:**
```bash
python scripts/diag_07_save_load.py
```

---

### Hypothesis 8: Actual Quantized Model Inspection
**Script:** `diag_08_inspect_model.py`

**Question:** What's actually in our saved quantized model?

**Method:**
1. Load our actual quantized model from `quantized_model/`
2. Check for NaN/Inf in all weights
3. Compare to FP16 baseline
4. Identify abnormal layers

**Expected Outcome:**
- Direct evidence of corruption if present
- May reveal layers that weren't quantized correctly

**Run:**
```bash
python scripts/diag_08_inspect_model.py
```

---

## Recommended Execution Order

### Quick Diagnostics (run these first):
```bash
python scripts/diag_02_verify_hessian.py    # 1 min - verify calibration
python scripts/diag_08_inspect_model.py     # 1 min - inspect saved model
python scripts/diag_07_save_load.py         # 3 min - test save/load
python scripts/diag_04_rtn_vs_gptq.py       # 5 min - RTN vs GPTQ
```

### Medium Diagnostics:
```bash
python scripts/diag_05_layer_groups.py      # 10 min - layer groups
python scripts/diag_06_weight_stats.py      # 2 min - weight stats
```

### Slow Diagnostics:
```bash
python scripts/diag_03_timestep_accumulation.py  # 5 min - timestep analysis
python scripts/diag_01_progressive_layers.py     # 30+ min - progressive test
```

### Or Run Everything:
```bash
python scripts/run_all_diagnostics.py       # Full suite
python scripts/run_all_diagnostics.py --quick  # Skip slow ones
```

---

## Interpretation Guide

### If `diag_02_verify_hessian.py` fails:
- Our calibration is buggy
- Need to fix incremental H computation

### If `diag_08_inspect_model.py` finds NaN/Inf:
- Our quantization is producing invalid values
- Check the apply() method for division by zero

### If `diag_04_rtn_vs_gptq.py` shows RTN works but GPTQ fails:
- Bug in error diffusion logic
- Check H_inv computation and error propagation

### If `diag_03_timestep_accumulation.py` shows exponential growth:
- Cross-timestep error is the problem
- May need timestep-aware calibration or reduced quantization

### If `diag_05_layer_groups.py` shows specific groups fail:
- Need to skip those layer types
- Update skip_patterns list

### If `diag_01_progressive_layers.py` shows tipping point:
- Can only quantize N layers before failure
- May need to prioritize which layers to quantize

---

## Output Directories

After running diagnostics, check:

| Directory | Contents |
|-----------|----------|
| `diagnosis_progressive/` | Images at each layer count |
| `diagnosis_timesteps/` | Divergence data per timestep |
| `diagnosis_rtn_vs_gptq/` | RTN vs GPTQ images |
| `diagnosis_layer_groups/` | Per-group test images |
| `diagnosis_weight_stats/` | Weight statistics JSON |
| `diagnosis_save_load/` | Save/load test images |
| `diagnosis_inspect_model/` | Inspection results |

---

## Questions to Answer

After running all diagnostics, we should be able to answer:

1. ✅ Is our H matrix computation correct?
2. ✅ Are there NaN/Inf in our quantized model?
3. ✅ Does RTN work while GPTQ fails?
4. ✅ Which layer groups are most sensitive?
5. ✅ How many layers can we quantize before failure?
6. ✅ Do errors grow across timesteps?
7. ✅ Is save/load corrupting weights?

The answers will guide us to the root cause and solution.
