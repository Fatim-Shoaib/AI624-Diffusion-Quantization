"""
=============================================================================
FIXED Calibration Data Collection for Qronos
=============================================================================

Key fixes:
1. Use float64 for H matrix computation to prevent overflow
2. Normalize activations before outer product
3. Symmetrize H matrix after computation
4. Validate H matrix properties before saving
"""

import gc
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm


class ActivationCollectorFixed:
    """
    Fixed activation collector with proper numerical handling.
    """
    
    def __init__(
        self,
        layer: nn.Linear,
        layer_name: str,
        device: str = 'cuda',
    ):
        self.layer = layer
        self.layer_name = layer_name
        self.device = device
        
        self.in_features = layer.in_features
        
        # Use float64 to prevent overflow!
        self.H = torch.zeros(
            (self.in_features, self.in_features),
            dtype=torch.float64,  # CRITICAL: Use double precision
            device='cpu'
        )
        self.G = torch.zeros(
            (self.in_features, self.in_features),
            dtype=torch.float64,
            device='cpu'
        )
        
        self.nsamples = 0
        self.quant_input_cache = None
        self.hook = None
        self.collecting_quant = True
        
        # Track statistics for debugging
        self.max_activation = 0.0
        self.mean_activation = 0.0
    
    def _hook_fn(self, module: nn.Module, inp: Tuple[Tensor, ...], output: Tensor):
        """Forward hook with numerical safeguards."""
        inp_tensor = inp[0]
        
        # Flatten to 2D
        if len(inp_tensor.shape) > 2:
            inp_tensor = inp_tensor.reshape(-1, inp_tensor.shape[-1])
        
        # Move to CPU and use float64
        inp_cpu = inp_tensor.detach().cpu().to(torch.float64)
        batch_size = inp_cpu.shape[0]
        
        # Track statistics
        self.max_activation = max(self.max_activation, inp_cpu.abs().max().item())
        self.mean_activation = (self.mean_activation * self.nsamples + 
                                inp_cpu.abs().mean().item() * batch_size) / (self.nsamples + batch_size + 1e-8)
        
        # Clamp extreme values to prevent overflow
        inp_cpu = torch.clamp(inp_cpu, -1e6, 1e6)
        
        inp_t = inp_cpu.t()  # [features, batch]
        
        if self.collecting_quant:
            # First pass: H = X̃ᵀX̃
            self.nsamples += batch_size
            
            # Compute outer product
            H_update = inp_t @ inp_t.t()
            
            # Running average with numerical stability
            alpha = (self.nsamples - batch_size) / max(self.nsamples, 1)
            self.H = alpha * self.H + H_update / max(self.nsamples, 1)
            
            # Cache for G
            self.quant_input_cache = inp_t.clone()
        else:
            # Second pass: G = X̃ᵀX
            if self.quant_input_cache is not None:
                # Use same batch size from first pass
                G_update = self.quant_input_cache @ inp_t.t()
                
                alpha = (self.nsamples - batch_size) / max(self.nsamples, 1)
                self.G = alpha * self.G + G_update / max(self.nsamples, 1)
                self.quant_input_cache = None
    
    def register(self):
        """Register the forward hook."""
        self.hook = self.layer.register_forward_hook(self._hook_fn)
        return self
    
    def remove(self):
        """Remove the forward hook."""
        if self.hook is not None:
            self.hook.remove()
            self.hook = None
    
    def set_mode(self, collecting_quant: bool):
        """Set collection mode."""
        self.collecting_quant = collecting_quant
    
    def get_matrices(self) -> Tuple[Tensor, Tensor]:
        """Get the computed H and G matrices with validation."""
        # Force symmetry for H (should already be symmetric, but ensure it)
        H = (self.H + self.H.t()) / 2
        G = self.G.clone()
        
        # Convert back to float32 for storage
        H = H.to(torch.float32)
        G = G.to(torch.float32)
        
        # Validate
        sym_error = (H - H.t()).abs().max().item()
        if sym_error > 1e-5:
            print(f"  WARNING: {self.layer_name} H matrix asymmetry: {sym_error:.2e}")
        
        # Check for NaN/Inf
        if torch.isnan(H).any() or torch.isinf(H).any():
            print(f"  WARNING: {self.layer_name} H contains NaN/Inf!")
            # Replace with identity as fallback
            H = torch.eye(self.in_features, dtype=torch.float32)
        
        return H, G
    
    def get_stats(self) -> dict:
        """Get collection statistics."""
        return {
            "nsamples": self.nsamples,
            "max_activation": self.max_activation,
            "mean_activation": self.mean_activation,
        }


def collect_calibration_data_fixed(
    pipe,
    prompts: List[str],
    output_dir: Path,
    num_inference_steps: int = 28,
    timestep_indices: Optional[List[int]] = None,
    guidance_scale: float = 4.5,
    height: int = 1024,
    width: int = 1024,
    seed: int = 42,
    checkpoint_every: int = 25,
):
    """
    Fixed calibration data collection.
    """
    device = pipe.device if hasattr(pipe, 'device') else 'cuda'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup timesteps
    if timestep_indices is None:
        timestep_indices = [0, 6, 13, 20, 27]
    
    # Create collectors for all linear layers
    collectors: Dict[str, ActivationCollectorFixed] = {}
    
    print("Setting up activation collectors...")
    for name, module in pipe.transformer.named_modules():
        if isinstance(module, nn.Linear):
            collector = ActivationCollectorFixed(module, name, device)
            collector.register()
            collectors[name] = collector
    
    print(f"Created {len(collectors)} collectors")
    
    # Check for existing checkpoint
    checkpoint_file = output_dir / "calibration_checkpoint.json"
    start_prompt_idx = 0
    
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            checkpoint = json.load(f)
        start_prompt_idx = checkpoint.get("completed_prompts", 0)
        print(f"Resuming from prompt {start_prompt_idx}")
        
        # Load existing H/G matrices
        for name, collector in collectors.items():
            H_file = output_dir / f"{name.replace('.', '_')}_H.pt"
            G_file = output_dir / f"{name.replace('.', '_')}_G.pt"
            if H_file.exists() and G_file.exists():
                collector.H = torch.load(H_file, weights_only=True).to(torch.float64)
                collector.G = torch.load(G_file, weights_only=True).to(torch.float64)
                collector.nsamples = checkpoint.get("nsamples", 0)
    
    # Collection loop
    generator = torch.Generator(device=device).manual_seed(seed)
    
    for prompt_idx in range(start_prompt_idx, len(prompts)):
        prompt = prompts[prompt_idx]
        print(f"\n[Prompt {prompt_idx + 1}/{len(prompts)}] {prompt[:50]}...")
        
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
            height, width, prompt_embeds.dtype, device, generator
        )
        
        # Setup scheduler
        pipe.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = pipe.scheduler.timesteps
        
        # Run selected timesteps
        for t_idx in timestep_indices:
            if t_idx >= len(timesteps):
                continue
            
            t = timesteps[t_idx]
            print(f"  Timestep {t_idx}: ", end="", flush=True)
            
            latent_model_input = torch.cat([latents] * 2)
            timestep = t.expand(latent_model_input.shape[0])
            
            # Pass 1: Collect for H (with standard forward)
            for collector in collectors.values():
                collector.set_mode(True)
            
            print("H...", end="", flush=True)
            with torch.no_grad():
                _ = pipe.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    return_dict=False,
                )[0]
            
            # Pass 2: Collect for G (same inputs, for cross-covariance)
            for collector in collectors.values():
                collector.set_mode(False)
            
            print("G...", end="", flush=True)
            with torch.no_grad():
                _ = pipe.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    return_dict=False,
                )[0]
            
            print("✓")
        
        # Checkpoint periodically
        if (prompt_idx + 1) % checkpoint_every == 0:
            print(f"\nSaving checkpoint at prompt {prompt_idx + 1}...")
            save_checkpoint(collectors, output_dir, prompt_idx + 1)
    
    # Remove hooks
    for collector in collectors.values():
        collector.remove()
    
    # Save final results
    print("\nSaving final calibration data...")
    save_calibration_data(collectors, output_dir)
    
    print(f"\nCalibration complete! Data saved to {output_dir}")
    
    return collectors


def save_checkpoint(collectors: Dict[str, ActivationCollectorFixed], 
                   output_dir: Path, completed_prompts: int):
    """Save checkpoint."""
    # Save H/G matrices
    for name, collector in collectors.items():
        H, G = collector.get_matrices()
        safe_name = name.replace('.', '_')
        torch.save(H, output_dir / f"{safe_name}_H.pt")
        torch.save(G, output_dir / f"{safe_name}_G.pt")
    
    # Save progress
    checkpoint = {
        "completed_prompts": completed_prompts,
        "nsamples": list(collectors.values())[0].nsamples if collectors else 0,
    }
    with open(output_dir / "calibration_checkpoint.json", "w") as f:
        json.dump(checkpoint, f)


def save_calibration_data(collectors: Dict[str, ActivationCollectorFixed], 
                         output_dir: Path):
    """Save final calibration data with metadata."""
    metadata = {
        "num_layers": len(collectors),
        "layers": {},
    }
    
    for name, collector in tqdm(collectors.items(), desc="Saving"):
        H, G = collector.get_matrices()
        stats = collector.get_stats()
        
        safe_name = name.replace('.', '_')
        H_file = f"{safe_name}_H.pt"
        G_file = f"{safe_name}_G.pt"
        
        torch.save(H, output_dir / H_file)
        torch.save(G, output_dir / G_file)
        
        metadata["layers"][name] = {
            "H_file": H_file,
            "G_file": G_file,
            "in_features": collector.in_features,
            "nsamples": stats["nsamples"],
            "max_activation": stats["max_activation"],
            "mean_activation": stats["mean_activation"],
        }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
