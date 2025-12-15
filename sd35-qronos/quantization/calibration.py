"""
=============================================================================
Calibration Data Collection for Qronos
=============================================================================

This module handles collecting calibration data for Qronos quantization.

Key Difference from GPTQ:
- GPTQ: Only collects H = XᵀX (Hessian from float inputs)
- Qronos: Collects BOTH H = X̃ᵀX̃ (from quantized inputs) AND G = X̃ᵀX (cross-covariance)

For diffusion models, we also sample across multiple timesteps to capture
the varying activation distributions throughout the denoising process.
"""

import gc
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm


class ActivationCollector:
    """
    Collects input activations for linear layers during forward passes.
    
    This implements the two-pass collection required for Qronos:
    1. First pass: Collect quantized activations → update H
    2. Second pass: Collect float activations → update G
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
        
        # Dimensions
        self.in_features = layer.in_features
        
        # Covariance matrices (on CPU for memory efficiency)
        self.H = torch.zeros(
            (self.in_features, self.in_features),
            dtype=torch.float32,
            device='cpu',
            pin_memory=torch.cuda.is_available()
        )
        self.G = torch.zeros(
            (self.in_features, self.in_features),
            dtype=torch.float32,
            device='cpu',
            pin_memory=torch.cuda.is_available()
        )
        self.B = torch.zeros(
            (self.in_features, self.in_features),
            dtype=torch.float32,
            device='cpu',
            pin_memory=torch.cuda.is_available()
        )
        
        self.nsamples = 0
        self.quant_input_cache = None
        
        # Hook handle
        self.hook = None
        self.collecting_quant = True  # Toggle between quant and float collection
    
    def _hook_fn(self, module: nn.Module, inp: Tuple[Tensor, ...], output: Tensor):
        """Forward hook to collect activations."""
        inp_tensor = inp[0]
        
        # Flatten to 2D: [batch * seq, features]
        if len(inp_tensor.shape) > 2:
            inp_tensor = inp_tensor.reshape(-1, inp_tensor.shape[-1])
        
        inp_tensor = inp_tensor.to(torch.float32)
        batch_size = inp_tensor.shape[0]
        inp_t = inp_tensor.t()  # [features, batch]
        
        if self.collecting_quant:
            # First pass: collecting quantized activations for H
            self.nsamples += batch_size
            self.B.copy_(inp_t.cpu() @ inp_t.t().cpu())
            self.H *= (self.nsamples - batch_size) / self.nsamples
            self.H += self.B / self.nsamples
            
            # Cache for G computation
            self.quant_input_cache = inp_t.cpu()
        else:
            # Second pass: collecting float activations for G
            if self.quant_input_cache is not None:
                self.B.copy_(self.quant_input_cache @ inp_t.t().cpu())
                self.G *= (self.nsamples - batch_size) / self.nsamples
                self.G += self.B / self.nsamples
                self.quant_input_cache = None
    
    def register_hook(self):
        """Register the forward hook."""
        self.hook = self.layer.register_forward_hook(self._hook_fn)
    
    def remove_hook(self):
        """Remove the forward hook."""
        if self.hook is not None:
            self.hook.remove()
            self.hook = None
    
    def set_collecting_quant(self, quant: bool):
        """Set whether to collect quantized or float activations."""
        self.collecting_quant = quant
    
    def get_covariance_matrices(self) -> Tuple[Tensor, Tensor]:
        """Return the collected H and G matrices."""
        return self.H, self.G
    
    def get_num_samples(self) -> int:
        """Return number of samples collected."""
        return self.nsamples


class CalibrationDataCollector:
    """
    Manages calibration data collection for multiple layers.
    
    Handles the complexity of:
    1. Collecting activations across multiple timesteps
    2. Two-pass collection (quantized then float)
    3. Memory-efficient storage
    """
    
    def __init__(
        self,
        transformer: nn.Module,
        skip_layers: Optional[List[str]] = None,
        device: str = 'cuda',
    ):
        """
        Initialize the calibration collector.
        
        Args:
            transformer: The SD3Transformer2DModel to collect from
            skip_layers: List of layer name patterns to skip
            device: Device for computation
        """
        self.transformer = transformer
        self.skip_layers = skip_layers or ['time_embed', 'label_embed', 'proj_out', 'pos_embed']
        self.device = device
        
        # Create collectors for each linear layer
        self.collectors: Dict[str, ActivationCollector] = {}
        
        for name, module in transformer.named_modules():
            if isinstance(module, nn.Linear):
                # Skip specified layers
                if any(skip in name for skip in self.skip_layers):
                    continue
                
                self.collectors[name] = ActivationCollector(
                    layer=module,
                    layer_name=name,
                    device=device,
                )
        
        print(f"Created collectors for {len(self.collectors)} layers")
    
    def register_hooks(self):
        """Register hooks on all collectors."""
        for collector in self.collectors.values():
            collector.register_hook()
    
    def remove_hooks(self):
        """Remove hooks from all collectors."""
        for collector in self.collectors.values():
            collector.remove_hook()
    
    def set_collecting_mode(self, quant: bool):
        """Set collection mode for all collectors."""
        for collector in self.collectors.values():
            collector.set_collecting_quant(quant)
    
    def collect_from_forward(
        self,
        forward_fn: Callable,
        forward_kwargs: dict,
        enable_input_quant: bool = True,
    ):
        """
        Collect activations from a forward pass.
        
        For Qronos, we need two forward passes:
        1. With input quantization enabled → H matrix
        2. With input quantization disabled → G matrix
        
        Args:
            forward_fn: Function to call for forward pass
            forward_kwargs: Keyword arguments for forward function
            enable_input_quant: Whether this is the quantized or float pass
        """
        self.set_collecting_mode(enable_input_quant)
        
        with torch.no_grad():
            forward_fn(**forward_kwargs)
    
    def get_all_covariance_matrices(self) -> Dict[str, Tuple[Tensor, Tensor]]:
        """
        Get H and G matrices for all layers.
        
        Returns:
            Dict mapping layer names to (H, G) tuples
        """
        result = {}
        for name, collector in self.collectors.items():
            result[name] = collector.get_covariance_matrices()
        return result
    
    def save_calibration_data(self, output_dir: Path):
        """
        Save collected calibration data to disk.
        
        Args:
            output_dir: Directory to save to
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        metadata = {
            'num_layers': len(self.collectors),
            'layers': {}
        }
        
        for name, collector in tqdm(self.collectors.items(), desc="Saving calibration data"):
            H, G = collector.get_covariance_matrices()
            
            # Create safe filename
            safe_name = name.replace('.', '_').replace('/', '_')
            h_path = output_dir / f"{safe_name}_H.pt"
            g_path = output_dir / f"{safe_name}_G.pt"
            
            torch.save(H, h_path)
            torch.save(G, g_path)
            
            metadata['layers'][name] = {
                'H_file': h_path.name,
                'G_file': g_path.name,
                'in_features': collector.in_features,
                'nsamples': collector.nsamples,
            }
        
        # Save metadata
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved calibration data for {len(self.collectors)} layers to {output_dir}")
    
    @staticmethod
    def load_calibration_data(calibration_dir: Path) -> Dict[str, Tuple[Tensor, Tensor]]:
        """
        Load calibration data from disk.
        
        Args:
            calibration_dir: Directory containing calibration data
            
        Returns:
            Dict mapping layer names to (H, G) tuples
        """
        calibration_dir = Path(calibration_dir)
        
        with open(calibration_dir / 'metadata.json') as f:
            metadata = json.load(f)
        
        result = {}
        for name, info in tqdm(metadata['layers'].items(), desc="Loading calibration data"):
            H = torch.load(calibration_dir / info['H_file'], map_location='cpu')
            G = torch.load(calibration_dir / info['G_file'], map_location='cpu')
            result[name] = (H, G)
        
        print(f"Loaded calibration data for {len(result)} layers")
        return result


def collect_calibration_for_diffusion(
    pipe,
    prompts: List[str],
    num_timesteps_per_sample: int = 5,
    timestep_strategy: str = "uniform",
    num_inference_steps: int = 28,
    guidance_scale: float = 4.5,
    height: int = 1024,
    width: int = 1024,
    seed: int = 42,
    device: str = "cuda",
) -> CalibrationDataCollector:
    """
    Collect calibration data for diffusion model quantization.
    
    This function handles the diffusion-specific aspects of calibration:
    1. Sampling across multiple timesteps
    2. Two-pass collection for Qronos (quantized then float)
    
    Args:
        pipe: StableDiffusion3Pipeline
        prompts: List of calibration prompts
        num_timesteps_per_sample: Number of timesteps to sample per prompt
        timestep_strategy: How to select timesteps ("uniform", "linear", "quadratic")
        num_inference_steps: Total inference steps
        guidance_scale: Classifier-free guidance scale
        height: Image height
        width: Image width
        seed: Random seed
        device: Device for computation
        
    Returns:
        CalibrationDataCollector with collected H and G matrices
    """
    transformer = pipe.transformer
    
    # Create collector
    collector = CalibrationDataCollector(
        transformer=transformer,
        device=device,
    )
    
    # Register hooks
    collector.register_hooks()
    
    # Select timesteps to sample
    if timestep_strategy == "uniform":
        timestep_indices = torch.linspace(0, num_inference_steps - 1, num_timesteps_per_sample).long()
    elif timestep_strategy == "linear":
        timestep_indices = torch.arange(0, num_inference_steps, num_inference_steps // num_timesteps_per_sample)[:num_timesteps_per_sample]
    else:  # quadratic - more samples at high noise
        t = torch.linspace(0, 1, num_timesteps_per_sample)
        timestep_indices = (t ** 2 * (num_inference_steps - 1)).long()
    
    print(f"Collecting calibration at timestep indices: {timestep_indices.tolist()}")
    
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # Process each prompt
    for prompt_idx, prompt in enumerate(tqdm(prompts, desc="Collecting calibration data")):
        # Encode prompt
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = (
            pipe.encode_prompt(
                prompt=prompt,
                prompt_2=None,
                prompt_3=None,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=guidance_scale > 1.0,
            )
        )
        
        # Prepare latents
        num_channels_latents = pipe.transformer.config.in_channels
        latents = pipe.prepare_latents(
            batch_size=1,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
        )
        
        # Set up scheduler
        pipe.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = pipe.scheduler.timesteps
        
        # Sample at selected timesteps
        for t_idx in timestep_indices:
            if t_idx >= len(timesteps):
                continue
                
            t = timesteps[t_idx]
            
            # Expand latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
            
            # Prepare timestep
            timestep = t.expand(latent_model_input.shape[0])
            
            # =========================================================
            # QRONOS: Two-pass collection
            # =========================================================
            
            # Pass 1: Collect with quantized activations (for H matrix)
            # Note: In full implementation, input quantization would be enabled here
            # For now, we collect the same activations for both H and G
            collector.set_collecting_mode(quant=True)
            
            with torch.no_grad():
                _ = pipe.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    return_dict=False,
                )[0]
            
            # Pass 2: Collect with float activations (for G matrix)
            collector.set_collecting_mode(quant=False)
            
            with torch.no_grad():
                _ = pipe.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    return_dict=False,
                )[0]
            
            # Take one denoising step to get new latents for next timestep
            if t_idx < len(timesteps) - 1:
                with torch.no_grad():
                    noise_pred = pipe.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        return_dict=False,
                    )[0]
                    
                    # Perform guidance
                    if guidance_scale > 1.0:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    
                    # Compute previous latents
                    latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        
        # Clear cache periodically
        if (prompt_idx + 1) % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    
    # Remove hooks
    collector.remove_hooks()
    
    return collector
