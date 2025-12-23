import torch
import torch.nn as nn
import os
import gc
import sys
import numpy as np

# CORRECTED IMPORT: Use Transformer2DModel instead of PixArtAlpha2DModel
from diffusers import Transformer2DModel, PixArtAlphaPipeline
from qronos import QronosQuantizer

# --- Configuration ---
MODEL_ID = "PixArt-alpha/PixArt-XL-2-1024-MS"
DEVICE = "cuda"
DTYPE = torch.float16
QUANT_BITS = 4
GROUP_SIZE = 128
CALIB_SAMPLES = 128
SEQ_LEN = 1024
CHECKPOINT_DIR = "./qronos_checkpoints"


# --- Quantizer Helper ---
class QuantizerParams:
    def __init__(self):
        self.maxq = 0
        self.scale = 0
        self.zero = 0

    def configure(self, bits):
        self.maxq = torch.tensor(2**bits - 1)

    def find_params(self, x, weight=False):
        dev = x.device
        self.maxq = self.maxq.to(dev)
        x = x.flatten(1)
        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1
        self.scale = (xmax - xmin) / self.maxq
        self.zero = torch.round(-xmin / self.scale)
        self.scale = self.scale.unsqueeze(1)
        self.zero = self.zero.unsqueeze(1)

    def ready(self):
        return torch.is_tensor(self.scale)


# --- Utils ---
def get_pixart_layers(model):
    layers = []
    skipped = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Constraint: DO NOT quantize Key and Value matrices
            # Specific naming for PixArt/DiT attention
            lower_name = name.lower()
            if (
                "to_k" in lower_name
                or "to_v" in lower_name
                or ".k." in lower_name
                or ".v." in lower_name
            ):
                skipped.append(name)
                continue
            layers.append((name, module))

    print(
        f"Skipped {len(skipped)} layers (Key/Value matrices). Examples: {skipped[:3]}"
    )
    return layers


def get_dummy_calibration_data(nsamples=16):
    data = []
    print("Generating dummy calibration data (Noise + Timesteps + Embeddings)...")
    for _ in range(nsamples):
        # PixArt-XL-2-1024 dimensions
        # Hidden states: [Batch, SeqLen, Channels] -> [1, 1024, 1152]
        hidden_states = torch.randn(1, SEQ_LEN, 1152).to(DEVICE, dtype=DTYPE)

        # Timestep: [Batch]
        timestep = torch.randint(0, 1000, (1,)).to(DEVICE)

        # Encoder states (T5 embeddings): [Batch, 120, 4096]
        encoder_hidden_states = torch.randn(1, 120, 4096).to(DEVICE, dtype=DTYPE)

        data.append((hidden_states, timestep, encoder_hidden_states))
    return data


# --- Evaluation Functions ---
def evaluate_model(model_path):
    print("\n--- Starting Evaluation ---")

    # 1. Load full pipeline
    try:
        pipe = PixArtAlphaPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(
            DEVICE
        )
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        return

    # 2. Inject Quantized Weights
    print(f"Loading quantized weights from {model_path}...")
    state_dict = torch.load(model_path, map_location=DEVICE)

    # Load into pipe.transformer
    miss, unexp = pipe.transformer.load_state_dict(state_dict, strict=False)
    if len(miss) > 0:
        print(
            f"Warning: Missing keys during load: {len(miss)} (Likely unquantized params)"
        )

    # 3. Peak VRAM
    torch.cuda.reset_peak_memory_stats()

    # 4. Generate Image
    prompt = (
        "A cinematic shot of an astronaut riding a horse on mars, highly detailed, 8k"
    )
    print(f"Generating image for prompt: '{prompt}'")

    with torch.no_grad():
        image = pipe(prompt, num_inference_steps=20).images[0]

    vram_after = torch.cuda.max_memory_allocated() / 1024**3
    print(f"Peak VRAM during inference: {vram_after:.2f} GB")

    # 5. Save Image
    os.makedirs("results", exist_ok=True)
    img_path = "results/pixart_qronos_output.png"
    image.save(img_path)
    print(f"Image saved to {img_path}")

    # 6. CLIP Score
    try:
        from torchmetrics.multimodal.clip_score import CLIPScore

        metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(DEVICE)
        img_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).to(DEVICE)
        score = metric(img_tensor, prompt)
        print(f"CLIP Score: {score.item():.4f}")
    except ImportError:
        print("CLIP Score skipped (torchmetrics not installed).")
    except Exception as e:
        print(f"CLIP Score failed: {e}")


def main():
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    # --- Load Model ---
    print(f"Loading Transformer from: {MODEL_ID}")
    # CORRECTED CLASS: Transformer2DModel
    model = Transformer2DModel.from_pretrained(
        MODEL_ID, subfolder="transformer", torch_dtype=DTYPE
    ).to(DEVICE)
    model.eval()

    layers = get_pixart_layers(model)
    print(f"Found {len(layers)} linear layers to quantize.")

    # --- Setup Qronos ---
    qronos_handlers = {}
    for name, layer in layers:
        qronos_handlers[name] = QronosQuantizer(layer)

    def add_batch_hook(name):
        def hook(module, inp, out):
            # Input is tuple (hidden_states, ...)
            qronos_handlers[name].add_batch(inp[0].data)

        return hook

    handles = []
    for name, layer in layers:
        handles.append(layer.register_forward_hook(add_batch_hook(name)))

    # --- Calibration ---
    calib_data = get_dummy_calibration_data(CALIB_SAMPLES)
    print(f"Running Calibration on {CALIB_SAMPLES} samples...")

    with torch.no_grad():
        for i, batch in enumerate(calib_data):
            hs, ts, ehs = batch
            model(hs, timestep=ts, encoder_hidden_states=ehs)
            if (i + 1) % 20 == 0:
                print(f"Calibrated {i+1}/{CALIB_SAMPLES}")

    for h in handles:
        h.remove()

    print("Calibration Complete. Starting Qronos Quantization...")

    # --- Quantization Loop ---
    for i, (name, layer) in enumerate(layers):
        print(f"[{i+1}/{len(layers)}] Quantizing {name}...")

        handler = qronos_handlers[name]

        quantizer = QuantizerParams()
        quantizer.configure(bits=QUANT_BITS)

        # Qronos Step (Float64 enforced inside)
        handler.fasterquant(
            quantizer,
            blocksize=128,
            groupsize=GROUP_SIZE,
            beta=1e4,  # Stability constant
        )

        handler.free()

        # Checkpoint
        if (i + 1) % 10 == 0:
            print(f"Saving Checkpoint...")
            torch.save(
                model.state_dict(),
                os.path.join(CHECKPOINT_DIR, f"pixart_qronos_step_{i+1}.pt"),
            )

        gc.collect()
        torch.cuda.empty_cache()

    # Final Save
    final_path = os.path.join(CHECKPOINT_DIR, "pixart_qronos_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"Model saved to {final_path}")

    # --- Run Evaluation ---
    del model
    gc.collect()
    torch.cuda.empty_cache()

    evaluate_model(final_path)


if __name__ == "__main__":
    main()
