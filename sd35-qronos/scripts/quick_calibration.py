#!/usr/bin/env python3
"""
=============================================================================
Quick Calibration for Testing Qronos
=============================================================================

Collects calibration data for ONLY the first N layers.
This is for quick testing of the Qronos algorithm.

Time estimate: ~5-10 minutes instead of 1 day
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import json
from tqdm import tqdm

from config import DEFAULT_VISUAL_PROMPTS as CALIBRATION_PROMPTS
from models.sd35_loader import load_sd35_pipeline


class QuickCalibrationCollector:
    """Collects H and G matrices for Qronos."""
    
    def __init__(self, layer: nn.Linear, layer_name: str):
        self.layer = layer
        self.layer_name = layer_name
        self.in_features = layer.in_features
        
        # Use float64 to prevent overflow
        self.H = torch.zeros((self.in_features, self.in_features), dtype=torch.float64, device='cpu')
        self.G = torch.zeros((self.in_features, self.in_features), dtype=torch.float64, device='cpu')
        
        self.nsamples = 0
        self.float_input_cache = None  # Cache float input for G computation
        self.hook = None
        self.collecting_H = True  # Toggle between H and G collection
    
    def _hook_fn(self, module, inp, output):
        inp_tensor = inp[0]
        
        # Flatten to 2D
        if inp_tensor.dim() > 2:
            inp_tensor = inp_tensor.reshape(-1, inp_tensor.shape[-1])
        
        # Move to CPU, use float64
        inp_cpu = inp_tensor.detach().cpu().to(torch.float64)
        batch_size = inp_cpu.shape[0]
        
        # Clamp to prevent overflow
        inp_cpu = torch.clamp(inp_cpu, -1e4, 1e4)
        
        inp_t = inp_cpu.t()  # [features, batch]
        
        if self.collecting_H:
            # First pass: H = X̃ᵀX̃ (from "quantized" inputs)
            # For now, we use the same inputs - the key is the G computation
            self.nsamples += batch_size
            
            H_update = inp_t @ inp_t.t()
            alpha = (self.nsamples - batch_size) / max(self.nsamples, 1)
            self.H = alpha * self.H + H_update / max(self.nsamples, 1)
            
            # Cache for G computation
            self.float_input_cache = inp_t.clone()
        else:
            # Second pass: G = X̃ᵀX (cross-covariance)
            if self.float_input_cache is not None:
                G_update = self.float_input_cache @ inp_t.t()
                alpha = (self.nsamples - batch_size) / max(self.nsamples, 1)
                self.G = alpha * self.G + G_update / max(self.nsamples, 1)
                self.float_input_cache = None
    
    def register(self):
        self.hook = self.layer.register_forward_hook(self._hook_fn)
        return self
    
    def remove(self):
        if self.hook:
            self.hook.remove()
            self.hook = None
    
    def set_mode(self, collecting_H: bool):
        self.collecting_H = collecting_H
    
    def get_matrices(self):
        # Force symmetry for H
        H = (self.H + self.H.t()) / 2
        G = self.G.clone()
        
        # Convert to float32
        H = H.to(torch.float32)
        G = G.to(torch.float32)
        
        return H, G


def main():
    output_dir = Path("calibration_quick")
    output_dir.mkdir(exist_ok=True)
    
    # Configuration
    num_layers = 10  # Only first 10 layers
    num_prompts = 20  # Only 20 prompts (instead of 256)
    num_timesteps = 3  # Only 3 timesteps (instead of 5)
    
    print("="*60)
    print("QUICK CALIBRATION FOR QRONOS TESTING")
    print("="*60)
    print(f"Layers to calibrate: {num_layers}")
    print(f"Prompts: {num_prompts}")
    print(f"Timesteps: {num_timesteps}")
    
    # Load pipeline
    print("\nLoading pipeline...")
    pipe = load_sd35_pipeline(device="cuda", dtype=torch.float16)
    device = "cuda"
    
    # Get first N linear layers
    linear_layers = []
    for name, module in pipe.transformer.named_modules():
        if isinstance(module, nn.Linear):
            linear_layers.append((name, module))
            if len(linear_layers) >= num_layers:
                break
    
    print(f"\nCollecting for {len(linear_layers)} layers:")
    for name, _ in linear_layers:
        print(f"  - {name}")
    
    # Create collectors
    collectors = {}
    for name, module in linear_layers:
        collector = QuickCalibrationCollector(module, name)
        collector.register()
        collectors[name] = collector
    
    # Timestep indices
    timestep_indices = [0, 14, 27][:num_timesteps]
    
    # Prompts
    prompts = CALIBRATION_PROMPTS[:num_prompts]
    
    print(f"\nRunning calibration...")
    
    generator = torch.Generator(device=device).manual_seed(42)
    
    for prompt_idx, prompt in enumerate(tqdm(prompts, desc="Prompts")):
        # Encode prompt
        (prompt_embeds, negative_prompt_embeds,
         pooled_prompt_embeds, negative_pooled_prompt_embeds) = pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            prompt_3=None,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
        )
        
        # Prepare latents
        latents = pipe.prepare_latents(
            1, pipe.transformer.config.in_channels,
            512, 512,  # Smaller resolution for speed
            prompt_embeds.dtype, device, generator
        )
        
        # Setup scheduler
        pipe.scheduler.set_timesteps(28, device=device)
        timesteps = pipe.scheduler.timesteps
        
        # Run selected timesteps
        for t_idx in timestep_indices:
            t = timesteps[t_idx]
            
            latent_model_input = torch.cat([latents] * 2)
            timestep = t.expand(latent_model_input.shape[0])
            
            # Pass 1: Collect for H
            for collector in collectors.values():
                collector.set_mode(True)
            
            with torch.no_grad():
                _ = pipe.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    return_dict=False,
                )[0]
            
            # Pass 2: Collect for G
            for collector in collectors.values():
                collector.set_mode(False)
            
            with torch.no_grad():
                _ = pipe.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    return_dict=False,
                )[0]
    
    # Remove hooks
    for collector in collectors.values():
        collector.remove()
    
    # Save calibration data
    print("\nSaving calibration data...")
    
    metadata = {"layers": {}}
    
    for name, collector in collectors.items():
        H, G = collector.get_matrices()
        
        safe_name = name.replace('.', '_')
        H_file = f"{safe_name}_H.pt"
        G_file = f"{safe_name}_G.pt"
        
        torch.save(H, output_dir / H_file)
        torch.save(G, output_dir / G_file)
        
        metadata["layers"][name] = {
            "H_file": H_file,
            "G_file": G_file,
            "in_features": collector.in_features,
            "nsamples": collector.nsamples,
        }
        
        # Print matrix stats
        print(f"\n{name}:")
        print(f"  H shape: {H.shape}, max: {H.abs().max():.2e}, symmetric: {(H - H.t()).abs().max() < 1e-5}")
        print(f"  G shape: {G.shape}, max: {G.abs().max():.2e}")
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Calibration complete! Saved to {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()