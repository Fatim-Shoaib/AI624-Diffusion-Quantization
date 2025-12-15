"""
=============================================================================
Combined Metrics Module
=============================================================================

This module provides a unified interface for all benchmark metrics:
- FID Score (image quality)
- CLIP Score (text-image alignment)
- Model Size (disk usage)
- Peak VRAM Usage (memory efficiency)
- Inference Time (speed)
"""

import torch
import torch.nn as nn
import numpy as np
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Any, Union
from pathlib import Path
import logging
from datetime import datetime
from PIL import Image
from tqdm import tqdm
import gc

from .fid_score import FIDCalculator
from .clip_score import CLIPScoreCalculator

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkMetrics:
    """
    Container for all benchmark metrics.
    """
    # Identification
    model_name: str = ""
    quantization_config: str = "FP16"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Quality Metrics
    fid_score: Optional[float] = None
    clip_score_mean: Optional[float] = None
    clip_score_std: Optional[float] = None
    
    # Size Metrics (in GB)
    model_size_gb: Optional[float] = None
    transformer_size_gb: Optional[float] = None
    
    # Memory Metrics (in GB)
    peak_vram_gb: Optional[float] = None
    vram_allocated_gb: Optional[float] = None
    
    # Speed Metrics
    inference_time_mean: Optional[float] = None
    inference_time_std: Optional[float] = None
    images_per_second: Optional[float] = None
    
    # Generation Config
    num_images: int = 0
    num_inference_steps: int = 28
    guidance_scale: float = 4.5
    resolution: str = "1024x1024"
    
    # Individual results (not saved in summary)
    clip_scores: List[float] = field(default_factory=list)
    inference_times: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding large lists)."""
        d = asdict(self)
        # Remove large lists for summary
        d.pop('clip_scores', None)
        d.pop('inference_times', None)
        return d
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            f"Benchmark Results: {self.model_name} ({self.quantization_config})",
            "=" * 60,
            f"Timestamp: {self.timestamp}",
            "",
            "Quality Metrics:",
            f"  FID Score:        {self.fid_score:.2f}" if self.fid_score else "  FID Score:        N/A",
            f"  CLIP Score:       {self.clip_score_mean:.4f} ± {self.clip_score_std:.4f}" if self.clip_score_mean else "  CLIP Score:       N/A",
            "",
            "Size Metrics:",
            f"  Model Size:       {self.model_size_gb:.2f} GB" if self.model_size_gb else "  Model Size:       N/A",
            f"  Transformer Size: {self.transformer_size_gb:.2f} GB" if self.transformer_size_gb else "  Transformer Size: N/A",
            "",
            "Memory Metrics:",
            f"  Peak VRAM:        {self.peak_vram_gb:.2f} GB" if self.peak_vram_gb else "  Peak VRAM:        N/A",
            "",
            "Speed Metrics:",
            f"  Inference Time:   {self.inference_time_mean:.2f}s ± {self.inference_time_std:.2f}s" if self.inference_time_mean else "  Inference Time:   N/A",
            f"  Throughput:       {self.images_per_second:.3f} img/s" if self.images_per_second else "  Throughput:       N/A",
            "",
            "Generation Config:",
            f"  Images Generated: {self.num_images}",
            f"  Inference Steps:  {self.num_inference_steps}",
            f"  Guidance Scale:   {self.guidance_scale}",
            f"  Resolution:       {self.resolution}",
            "=" * 60,
        ]
        return "\n".join(lines)


class VRAMTracker:
    """
    Context manager for tracking peak VRAM usage.
    """
    
    def __init__(self, device: torch.device = torch.device("cuda")):
        self.device = device
        self.start_allocated = 0
        self.start_reserved = 0
        self.peak_allocated = 0
        
    def __enter__(self):
        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.synchronize(self.device)
        
        self.start_allocated = torch.cuda.memory_allocated(self.device)
        self.start_reserved = torch.cuda.memory_reserved(self.device)
        
        return self
    
    def __exit__(self, *args):
        torch.cuda.synchronize(self.device)
        self.peak_allocated = torch.cuda.max_memory_allocated(self.device)
    
    @property
    def peak_gb(self) -> float:
        return self.peak_allocated / (1024 ** 3)
    
    @property
    def delta_gb(self) -> float:
        return (self.peak_allocated - self.start_allocated) / (1024 ** 3)


def get_model_size_gb(model: nn.Module) -> float:
    """
    Calculate model size in GB from parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Size in GB
    """
    total_bytes = 0
    for param in model.parameters():
        total_bytes += param.numel() * param.element_size()
    for buffer in model.buffers():
        total_bytes += buffer.numel() * buffer.element_size()
    return total_bytes / (1024 ** 3)


def get_model_disk_size_gb(path: Union[str, Path]) -> float:
    """
    Get model size on disk.
    
    Args:
        path: Path to model file or directory
        
    Returns:
        Size in GB
    """
    path = Path(path)
    
    if path.is_file():
        return path.stat().st_size / (1024 ** 3)
    elif path.is_dir():
        total_size = sum(
            f.stat().st_size for f in path.rglob('*') if f.is_file()
        )
        return total_size / (1024 ** 3)
    else:
        return 0.0


def evaluate_model(
    pipeline,
    prompts: List[str],
    output_dir: Path,
    reference_path: Optional[Path] = None,
    num_inference_steps: int = 28,
    guidance_scale: float = 4.5,
    seed: int = 42,
    batch_size: int = 1,
    calculate_fid: bool = True,
    calculate_clip: bool = True,
    device: str = "cuda",
) -> BenchmarkMetrics:
    """
    Run full evaluation on a model.
    
    Args:
        pipeline: SD3 pipeline (or compatible)
        prompts: List of prompts to generate images for
        output_dir: Directory to save generated images
        reference_path: Path to reference images for FID
        num_inference_steps: Inference steps per image
        guidance_scale: CFG scale
        seed: Random seed for reproducibility
        batch_size: Batch size for generation
        calculate_fid: Whether to calculate FID
        calculate_clip: Whether to calculate CLIP score
        device: Device for evaluation models
        
    Returns:
        BenchmarkMetrics with all results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = BenchmarkMetrics(
        num_images=len(prompts),
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )
    
    # Get model info
    if hasattr(pipeline, 'transformer'):
        metrics.transformer_size_gb = get_model_size_gb(pipeline.transformer)
    
    generated_images = []
    inference_times = []
    
    # Generate images
    logger.info(f"Generating {len(prompts)} images...")
    
    generator = torch.Generator(device=device).manual_seed(seed)
    
    with VRAMTracker(torch.device(device)) as vram_tracker:
        for i, prompt in enumerate(tqdm(prompts, desc="Generating")):
            # Reset generator for reproducibility per image
            generator.manual_seed(seed + i)
            
            # Time the generation
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            with torch.no_grad():
                result = pipeline(
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                )
            
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            # Store results
            image = result.images[0]
            generated_images.append(image)
            inference_times.append(end_time - start_time)
            
            # Save image
            image_path = output_dir / f"{i:06d}.png"
            image.save(image_path)
    
    # Record VRAM usage
    metrics.peak_vram_gb = vram_tracker.peak_gb
    metrics.vram_allocated_gb = torch.cuda.memory_allocated(device) / (1024 ** 3)
    
    # Record timing
    metrics.inference_times = inference_times
    metrics.inference_time_mean = float(np.mean(inference_times))
    metrics.inference_time_std = float(np.std(inference_times))
    metrics.images_per_second = 1.0 / metrics.inference_time_mean
    
    # Calculate CLIP score
    if calculate_clip:
        logger.info("Calculating CLIP scores...")
        clip_calc = CLIPScoreCalculator(device=torch.device(device))
        clip_mean, clip_scores = clip_calc.calculate_score_batch(
            generated_images, prompts, batch_size=32
        )
        metrics.clip_score_mean = clip_mean
        metrics.clip_score_std = float(np.std(clip_scores))
        metrics.clip_scores = clip_scores
        
        # Cleanup CLIP model
        del clip_calc
        gc.collect()
        torch.cuda.empty_cache()
    
    # Calculate FID
    if calculate_fid and reference_path:
        logger.info("Calculating FID score...")
        fid_calc = FIDCalculator(device=torch.device(device))
        metrics.fid_score = fid_calc.calculate_fid(
            output_dir,
            reference_path,
            max_images=len(prompts),
        )
        
        # Cleanup FID model
        del fid_calc
        gc.collect()
        torch.cuda.empty_cache()
    
    logger.info(f"Evaluation complete:\n{metrics.summary()}")
    
    return metrics


def compare_models(
    metrics_list: List[BenchmarkMetrics],
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Compare metrics from multiple models.
    
    Args:
        metrics_list: List of BenchmarkMetrics to compare
        output_path: Optional path to save comparison
        
    Returns:
        Comparison dictionary
    """
    comparison = {
        "models": [],
        "metrics": {
            "fid": [],
            "clip": [],
            "vram": [],
            "speed": [],
            "size": [],
        }
    }
    
    for m in metrics_list:
        comparison["models"].append(f"{m.model_name} ({m.quantization_config})")
        comparison["metrics"]["fid"].append(m.fid_score)
        comparison["metrics"]["clip"].append(m.clip_score_mean)
        comparison["metrics"]["vram"].append(m.peak_vram_gb)
        comparison["metrics"]["speed"].append(m.inference_time_mean)
        comparison["metrics"]["size"].append(m.transformer_size_gb)
    
    # Generate comparison text
    lines = [
        "=" * 80,
        "Model Comparison",
        "=" * 80,
        "",
        f"{'Model':<40} {'FID':>10} {'CLIP':>10} {'VRAM (GB)':>12} {'Time (s)':>10} {'Size (GB)':>10}",
        "-" * 80,
    ]
    
    for i, name in enumerate(comparison["models"]):
        fid = comparison["metrics"]["fid"][i]
        clip = comparison["metrics"]["clip"][i]
        vram = comparison["metrics"]["vram"][i]
        speed = comparison["metrics"]["speed"][i]
        size = comparison["metrics"]["size"][i]
        
        lines.append(
            f"{name:<40} "
            f"{fid:>10.2f}" if fid else f"{'N/A':>10}" 
            f"{clip:>10.4f}" if clip else f"{'N/A':>10}"
            f"{vram:>12.2f}" if vram else f"{'N/A':>12}"
            f"{speed:>10.2f}" if speed else f"{'N/A':>10}"
            f"{size:>10.2f}" if size else f"{'N/A':>10}"
        )
    
    lines.append("=" * 80)
    comparison["summary"] = "\n".join(lines)
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(comparison, f, indent=2, default=str)
        logger.info(f"Comparison saved to {output_path}")
    
    return comparison


def save_metrics(metrics: BenchmarkMetrics, path: Path) -> None:
    """
    Save metrics to JSON file.
    
    Args:
        metrics: Metrics to save
        path: Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(metrics.to_dict(), f, indent=2, default=str)
    
    logger.info(f"Metrics saved to {path}")


def load_metrics(path: Path) -> BenchmarkMetrics:
    """
    Load metrics from JSON file.
    
    Args:
        path: Path to metrics file
        
    Returns:
        BenchmarkMetrics instance
    """
    with open(path) as f:
        data = json.load(f)
    
    return BenchmarkMetrics(**data)
