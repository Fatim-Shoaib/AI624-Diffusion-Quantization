"""
PixArt Model Handler for Qronos-DiT quantization.

This module handles loading the PixArt model, collecting calibration data,
and managing the quantization process.
"""
import os
import gc
import json
import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any, Tuple
from tqdm import tqdm
from pathlib import Path

from .qronos import QronosDiT, QronosDiTSimple
from .qlinear import find_linear_layers, QLinearLayer
from .quant_utils import Quantizer


class PixArtQuantizer:
    """
    Quantizer for PixArt-α diffusion models.
    
    This class manages the full quantization pipeline:
    1. Load the PixArt model
    2. Collect calibration data from COCO
    3. Apply Qronos quantization layer by layer
    4. Save checkpoints and final model
    """
    
    def __init__(
        self,
        model_id: str = "PixArt-alpha/PixArt-XL-2-512x512",
        bits: int = 8,
        sym: bool = True,
        group_size: int = -1,
        device: str = "cuda",
        checkpoint_dir: str = "./checkpoints",
        skip_layers: List[str] = None,
    ):
        """
        Initialize the PixArt quantizer.
        
        Args:
            model_id: HuggingFace model ID or local path
            bits: Quantization bit-width
            sym: Use symmetric quantization
            group_size: Group size for quantization (-1 for per-channel)
            device: Device to use for quantization
            checkpoint_dir: Directory to save checkpoints
            skip_layers: List of layer name patterns to skip (e.g., ['to_k', 'to_v'])
        """
        self.model_id = model_id
        self.bits = bits
        self.sym = sym
        self.group_size = group_size
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Skip K and V projection layers as per requirement
        self.skip_layers = skip_layers or ['to_k', 'to_v']
        
        self.pipe = None
        self.transformer = None
        self.quantized_layers = set()
        
        # Track progress for checkpointing
        self.current_block_idx = 0
        self.total_blocks = 0
    
    def load_model(self):
        """Load the PixArt model from HuggingFace."""
        from diffusers import PixArtAlphaPipeline
        
        print(f"Loading PixArt model: {self.model_id}")
        self.pipe = PixArtAlphaPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
        )
        
        # Get the transformer (DiT) component
        self.transformer = self.pipe.transformer
        self.transformer.eval()
        
        # Count transformer blocks
        if hasattr(self.transformer, 'transformer_blocks'):
            self.total_blocks = len(self.transformer.transformer_blocks)
        else:
            self.total_blocks = 28  # Default for PixArt-XL
        
        print(f"Model loaded. Transformer has {self.total_blocks} blocks.")
        
        return self.pipe
    
    def get_layers_to_quantize(self) -> Dict[str, nn.Linear]:
        """
        Get all linear layers to quantize, excluding skip_layers.
        
        Returns:
            Dictionary mapping layer names to nn.Linear modules
        """
        if self.transformer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        all_layers = find_linear_layers(self.transformer, layers_to_skip=self.skip_layers)
        
        # Filter out already quantized layers
        layers = {
            name: layer for name, layer in all_layers.items()
            if name not in self.quantized_layers
        }
        
        print(f"Found {len(layers)} layers to quantize (skipping: {self.skip_layers})")
        return layers
    
    def collect_calibration_data(
        self,
        prompts: List[str],
        num_inference_steps: int = 20,
        guidance_scale: float = 4.5,
        num_timesteps_to_sample: int = 10,
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Collect calibration activations from the model.
        
        This runs the diffusion model on calibration prompts and collects
        the input activations for each linear layer.
        
        Args:
            prompts: List of text prompts for calibration
            num_inference_steps: Number of diffusion steps
            guidance_scale: Classifier-free guidance scale
            num_timesteps_to_sample: Number of timesteps to sample activations from
        
        Returns:
            Dictionary mapping layer names to lists of activation tensors
        """
        print(f"Collecting calibration data from {len(prompts)} prompts...")
        
        # Move model to device
        self.pipe.to(self.device)
        
        # Get layers to hook
        layers = self.get_layers_to_quantize()
        
        # Storage for activations
        activations = {name: [] for name in layers.keys()}
        
        # Register hooks to collect activations
        hooks = []
        
        def make_hook(name):
            def hook(module, input, output):
                # Store input activation
                inp = input[0].detach().float()
                activations[name].append(inp.cpu())
            return hook
        
        for name, layer in layers.items():
            hooks.append(layer.register_forward_hook(make_hook(name)))
        
        # Run inference on calibration prompts
        timesteps_to_sample = torch.linspace(
            0, num_inference_steps - 1, num_timesteps_to_sample
        ).long().tolist()
        
        with torch.no_grad():
            for prompt_idx, prompt in enumerate(tqdm(prompts, desc="Calibration")):
                try:
                    # Generate with the prompt
                    _ = self.pipe(
                        prompt=prompt,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        output_type="latent",
                    )
                except Exception as e:
                    print(f"Warning: Failed on prompt {prompt_idx}: {e}")
                    continue
                
                # Clear cache periodically
                if prompt_idx % 10 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Concatenate activations
        for name in activations:
            if activations[name]:
                activations[name] = torch.cat(activations[name], dim=0)
            else:
                activations[name] = None
        
        print(f"Collected activations for {len(activations)} layers")
        return activations
    
    def quantize_layer(
        self,
        layer_name: str,
        layer: nn.Linear,
        activations: torch.Tensor,
        blocksize: int = 128,
        percdamp: float = 0.01,
        alpha: float = 1e-6,
    ) -> Dict[str, Any]:
        """
        Quantize a single layer using Qronos.
        
        Args:
            layer_name: Name of the layer
            layer: The nn.Linear layer to quantize
            activations: Calibration activations for this layer
            blocksize: Block size for GPTQ
            percdamp: Percentage damping
            alpha: Qronos damping factor
        
        Returns:
            Dictionary with quantization statistics
        """
        print(f"  Quantizing {layer_name}...")
        
        # Move layer to device
        layer = layer.to(self.device)
        
        # Create Qronos quantizer
        qronos = QronosDiTSimple(layer, layer_name)
        
        # Configure quantizer
        qronos.quantizer.configure(
            bits=self.bits,
            perchannel=True,
            channel_group=1,
            sym=self.sym,
            clip_ratio=1.0,
        )
        
        # Process activations in batches
        batch_size = 256
        for i in range(0, len(activations), batch_size):
            batch = activations[i:i + batch_size].to(self.device)
            qronos.add_batch(batch)
        
        # Quantize
        stats = qronos.fasterquant(
            blocksize=blocksize,
            percdamp=percdamp,
            groupsize=self.group_size,
            alpha=alpha,
        )
        
        # Cleanup
        qronos.free()
        
        return stats
    
    def quantize_block(
        self,
        block_idx: int,
        activations: Dict[str, torch.Tensor],
        blocksize: int = 128,
        percdamp: float = 0.01,
        alpha: float = 1e-6,
    ) -> List[Dict[str, Any]]:
        """
        Quantize all layers in a transformer block.
        
        Args:
            block_idx: Index of the transformer block
            activations: Dictionary of activations
            blocksize: Block size for GPTQ
            percdamp: Percentage damping
            alpha: Qronos damping factor
        
        Returns:
            List of quantization statistics for each layer
        """
        print(f"\nQuantizing Block {block_idx}/{self.total_blocks}...")
        
        # Find layers in this block
        block_prefix = f"transformer_blocks.{block_idx}."
        block_layers = {
            name: layer for name, layer in find_linear_layers(self.transformer).items()
            if name.startswith(block_prefix) and name not in self.quantized_layers
        }
        
        # Filter out skip layers
        block_layers = {
            name: layer for name, layer in block_layers.items()
            if not any(skip in name for skip in self.skip_layers)
        }
        
        stats_list = []
        
        for name, layer in block_layers.items():
            if name not in activations or activations[name] is None:
                print(f"  Skipping {name}: No activations")
                continue
            
            stats = self.quantize_layer(
                name, layer, activations[name],
                blocksize=blocksize,
                percdamp=percdamp,
                alpha=alpha,
            )
            stats_list.append(stats)
            
            self.quantized_layers.add(name)
        
        return stats_list
    
    def save_checkpoint(self, checkpoint_name: str = None):
        """
        Save a checkpoint of the current quantization state.
        
        Args:
            checkpoint_name: Name for the checkpoint (default: auto-generated)
        """
        if checkpoint_name is None:
            checkpoint_name = f"checkpoint_block_{self.current_block_idx}"
        
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Save transformer state dict
        torch.save(
            self.transformer.state_dict(),
            checkpoint_path / "transformer.pt"
        )
        
        # Save metadata
        metadata = {
            'model_id': self.model_id,
            'bits': self.bits,
            'sym': self.sym,
            'group_size': self.group_size,
            'current_block_idx': self.current_block_idx,
            'total_blocks': self.total_blocks,
            'quantized_layers': list(self.quantized_layers),
            'skip_layers': self.skip_layers,
        }
        
        with open(checkpoint_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """
        Load a checkpoint to resume quantization.
        
        Args:
            checkpoint_path: Path to the checkpoint directory
        
        Returns:
            True if checkpoint was loaded successfully
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            print(f"Checkpoint not found: {checkpoint_path}")
            return False
        
        # Load metadata
        with open(checkpoint_path / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        # Verify compatibility
        if metadata['model_id'] != self.model_id:
            print(f"Warning: Model ID mismatch. Checkpoint: {metadata['model_id']}, Current: {self.model_id}")
        
        # Load state
        self.current_block_idx = metadata['current_block_idx']
        self.quantized_layers = set(metadata['quantized_layers'])
        
        # Load transformer weights
        if self.transformer is None:
            self.load_model()
        
        self.transformer.load_state_dict(
            torch.load(checkpoint_path / "transformer.pt", map_location='cpu')
        )
        
        print(f"Checkpoint loaded. Resuming from block {self.current_block_idx}")
        return True
    
    def quantize_full_model(
        self,
        prompts: List[str],
        blocksize: int = 128,
        percdamp: float = 0.01,
        alpha: float = 1e-6,
        checkpoint_interval: int = 4,
        resume_from: str = None,
    ) -> Dict[str, Any]:
        """
        Quantize the full model block by block.
        
        Args:
            prompts: List of calibration prompts
            blocksize: Block size for GPTQ
            percdamp: Percentage damping
            alpha: Qronos damping factor
            checkpoint_interval: Save checkpoint every N blocks
            resume_from: Path to checkpoint to resume from
        
        Returns:
            Dictionary with overall quantization statistics
        """
        # Load model if not loaded
        if self.transformer is None:
            self.load_model()
        
        # Resume from checkpoint if specified
        if resume_from:
            self.load_checkpoint(resume_from)
        
        all_stats = []
        
        # Collect activations
        print("\n" + "=" * 60)
        print("STEP 1: Collecting Calibration Data")
        print("=" * 60)
        activations = self.collect_calibration_data(prompts)
        
        # Quantize block by block
        print("\n" + "=" * 60)
        print("STEP 2: Quantizing Layers")
        print("=" * 60)
        
        # Get all layers grouped by block
        for block_idx in range(self.current_block_idx, self.total_blocks):
            self.current_block_idx = block_idx
            
            stats = self.quantize_block(
                block_idx, activations,
                blocksize=blocksize,
                percdamp=percdamp,
                alpha=alpha,
            )
            all_stats.extend(stats)
            
            # Checkpoint
            if (block_idx + 1) % checkpoint_interval == 0:
                self.save_checkpoint()
            
            # Memory cleanup
            gc.collect()
            torch.cuda.empty_cache()
        
        # Quantize non-block layers (patch embed, final layer, etc.)
        print("\nQuantizing non-block layers...")
        non_block_layers = {
            name: layer for name, layer in find_linear_layers(self.transformer).items()
            if 'transformer_blocks' not in name and name not in self.quantized_layers
        }
        non_block_layers = {
            name: layer for name, layer in non_block_layers.items()
            if not any(skip in name for skip in self.skip_layers)
        }
        
        for name, layer in non_block_layers.items():
            if name in activations and activations[name] is not None:
                stats = self.quantize_layer(
                    name, layer, activations[name],
                    blocksize=blocksize,
                    percdamp=percdamp,
                    alpha=alpha,
                )
                all_stats.append(stats)
                self.quantized_layers.add(name)
        
        # Final checkpoint
        self.save_checkpoint("final")
        
        return {
            'total_layers': len(all_stats),
            'total_loss': sum(s.get('loss', 0) for s in all_stats),
            'total_time': sum(s.get('time', 0) for s in all_stats),
            'layer_stats': all_stats,
        }
    
    def save_quantized_model(self, output_path: str):
        """
        Save the quantized model.
        
        Args:
            output_path: Path to save the model
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save transformer
        torch.save(
            self.transformer.state_dict(),
            output_path / "transformer_quantized.pt"
        )
        
        # Save config
        config = {
            'model_id': self.model_id,
            'bits': self.bits,
            'sym': self.sym,
            'group_size': self.group_size,
            'quantized_layers': list(self.quantized_layers),
            'skip_layers': self.skip_layers,
        }
        
        with open(output_path / "quant_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Quantized model saved to {output_path}")
    
    def get_pipeline(self):
        """Get the pipeline with quantized transformer."""
        if self.pipe is None:
            raise ValueError("Model not loaded")
        return self.pipe
