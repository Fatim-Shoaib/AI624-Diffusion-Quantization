"""
=============================================================================
Evaluation Metrics for Diffusion Model Quantization
=============================================================================

This module provides utilities for evaluating quantized diffusion models:
- FID Score (Fréchet Inception Distance)
- CLIP Score (text-image alignment)
- VRAM Usage tracking
- Inference timing
"""

import gc
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

# Try to import evaluation libraries
try:
    import open_clip
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False
    print("Warning: open_clip not available. CLIP scores will be skipped.")

try:
    from torchvision import transforms
    from torchvision.models import inception_v3, Inception_V3_Weights
    INCEPTION_AVAILABLE = True
except ImportError:
    INCEPTION_AVAILABLE = False
    print("Warning: torchvision inception not available. FID will be skipped.")

try:
    from scipy import linalg
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class VRAMTracker:
    """Track peak VRAM usage during operations."""
    
    def __init__(self):
        self.peak_memory = 0
        self.start_memory = 0
    
    def start(self):
        """Start tracking."""
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        self.start_memory = torch.cuda.memory_allocated()
    
    def stop(self) -> float:
        """Stop tracking and return peak memory in GB."""
        torch.cuda.synchronize()
        self.peak_memory = torch.cuda.max_memory_allocated()
        return self.peak_memory / (1024 ** 3)
    
    def get_current(self) -> float:
        """Get current memory usage in GB."""
        return torch.cuda.memory_allocated() / (1024 ** 3)


class InferenceTimer:
    """Time inference operations using CUDA events."""
    
    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.times = []
    
    def start(self):
        """Start timing."""
        torch.cuda.synchronize()
        self.start_event.record()
    
    def stop(self) -> float:
        """Stop timing and return elapsed time in seconds."""
        self.end_event.record()
        torch.cuda.synchronize()
        elapsed = self.start_event.elapsed_time(self.end_event) / 1000.0
        self.times.append(elapsed)
        return elapsed
    
    def get_stats(self) -> Dict[str, float]:
        """Get timing statistics."""
        if not self.times:
            return {"mean": 0, "std": 0, "min": 0, "max": 0}
        return {
            "mean": np.mean(self.times),
            "std": np.std(self.times),
            "min": np.min(self.times),
            "max": np.max(self.times),
        }


class CLIPScorer:
    """Calculate CLIP scores for text-image pairs."""
    
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str = "cuda",
    ):
        if not OPEN_CLIP_AVAILABLE:
            raise ImportError("open_clip is required for CLIP scores")
        
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(device).eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
    
    @torch.no_grad()
    def compute_score(self, image: Image.Image, text: str) -> float:
        """
        Compute CLIP score for a single image-text pair.
        
        Args:
            image: PIL Image
            text: Text prompt
            
        Returns:
            CLIP score (cosine similarity)
        """
        # Process image
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Process text
        text_tokens = self.tokenizer([text]).to(self.device)
        
        # Get embeddings
        image_features = self.model.encode_image(image_tensor)
        text_features = self.model.encode_text(text_tokens)
        
        # Normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Compute cosine similarity
        similarity = (image_features @ text_features.T).item()
        
        return similarity
    
    @torch.no_grad()
    def compute_batch_scores(
        self, images: List[Image.Image], texts: List[str]
    ) -> List[float]:
        """Compute CLIP scores for multiple image-text pairs."""
        scores = []
        for image, text in zip(images, texts):
            scores.append(self.compute_score(image, text))
        return scores


class FIDCalculator:
    """Calculate FID score using Inception V3 features."""
    
    def __init__(self, device: str = "cuda"):
        if not INCEPTION_AVAILABLE:
            raise ImportError("torchvision is required for FID calculation")
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy is required for FID calculation")
        
        self.device = device
        
        # Load Inception V3
        self.model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        self.model.fc = nn.Identity()  # Remove classifier
        self.model = self.model.to(device).eval()
        
        # Preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    @torch.no_grad()
    def extract_features(self, images: List[Image.Image], batch_size: int = 32) -> np.ndarray:
        """Extract Inception features from images."""
        features = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            tensors = torch.stack([self.preprocess(img.convert("RGB")) for img in batch])
            tensors = tensors.to(self.device)
            
            batch_features = self.model(tensors)
            features.append(batch_features.cpu().numpy())
        
        return np.concatenate(features, axis=0)
    
    def compute_statistics(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute mean and covariance of features."""
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma
    
    def compute_fid(
        self,
        mu1: np.ndarray, sigma1: np.ndarray,
        mu2: np.ndarray, sigma2: np.ndarray,
    ) -> float:
        """
        Compute FID between two sets of statistics.
        
        FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
        """
        diff = mu1 - mu2
        
        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        
        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        
        return float(fid)
    
    def compute_fid_from_images(
        self,
        generated_images: List[Image.Image],
        reference_images: List[Image.Image],
        batch_size: int = 32,
    ) -> float:
        """
        Compute FID between generated and reference images.
        
        Args:
            generated_images: List of generated PIL images
            reference_images: List of reference PIL images
            batch_size: Batch size for feature extraction
            
        Returns:
            FID score
        """
        print("Extracting features from generated images...")
        gen_features = self.extract_features(generated_images, batch_size)
        
        print("Extracting features from reference images...")
        ref_features = self.extract_features(reference_images, batch_size)
        
        print("Computing FID...")
        mu1, sigma1 = self.compute_statistics(gen_features)
        mu2, sigma2 = self.compute_statistics(ref_features)
        
        return self.compute_fid(mu1, sigma1, mu2, sigma2)


def generate_images_for_evaluation(
    pipe,
    prompts: List[str],
    num_inference_steps: int = 28,
    guidance_scale: float = 4.5,
    height: int = 1024,
    width: int = 1024,
    seed: int = 42,
    device: str = "cuda",
    track_metrics: bool = True,
) -> Tuple[List[Image.Image], Dict[str, any]]:
    """
    Generate images for evaluation.
    
    Args:
        pipe: StableDiffusion3Pipeline
        prompts: List of prompts
        num_inference_steps: Number of denoising steps
        guidance_scale: CFG scale
        height: Image height
        width: Image width
        seed: Random seed
        device: Device
        track_metrics: Whether to track VRAM and timing
        
    Returns:
        Tuple of (generated images, metrics dict)
    """
    images = []
    metrics = {
        "inference_times": [],
        "peak_vram_gb": 0,
    }
    
    timer = InferenceTimer() if track_metrics else None
    vram_tracker = VRAMTracker() if track_metrics else None
    
    generator = torch.Generator(device=device).manual_seed(seed)
    
    for i, prompt in enumerate(tqdm(prompts, desc="Generating images")):
        if track_metrics:
            vram_tracker.start()
            timer.start()
        
        with torch.no_grad():
            result = pipe(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
        
        if track_metrics:
            elapsed = timer.stop()
            peak_vram = vram_tracker.stop()
            metrics["inference_times"].append(elapsed)
            metrics["peak_vram_gb"] = max(metrics["peak_vram_gb"], peak_vram)
        
        images.append(result.images[0])
        
        # Periodically clear cache
        if (i + 1) % 50 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    
    # Compute timing statistics
    if track_metrics and metrics["inference_times"]:
        times = metrics["inference_times"]
        metrics["mean_inference_time"] = np.mean(times)
        metrics["std_inference_time"] = np.std(times)
    
    return images, metrics


def run_full_evaluation(
    pipe,
    prompts: List[str],
    reference_images: Optional[List[Image.Image]] = None,
    num_inference_steps: int = 28,
    guidance_scale: float = 4.5,
    height: int = 1024,
    width: int = 1024,
    seed: int = 42,
    device: str = "cuda",
    compute_fid: bool = True,
    compute_clip: bool = True,
) -> Dict[str, any]:
    """
    Run full evaluation pipeline.
    
    Args:
        pipe: StableDiffusion3Pipeline
        prompts: List of prompts
        reference_images: Reference images for FID (optional)
        num_inference_steps: Number of denoising steps
        guidance_scale: CFG scale
        height: Image height
        width: Image width
        seed: Random seed
        device: Device
        compute_fid: Whether to compute FID
        compute_clip: Whether to compute CLIP scores
        
    Returns:
        Dictionary of evaluation metrics
    """
    results = {}
    
    # Generate images
    print("\n=== Generating Images ===")
    images, gen_metrics = generate_images_for_evaluation(
        pipe=pipe,
        prompts=prompts,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        seed=seed,
        device=device,
        track_metrics=True,
    )
    
    results["num_images"] = len(images)
    results["mean_inference_time"] = gen_metrics.get("mean_inference_time", 0)
    results["std_inference_time"] = gen_metrics.get("std_inference_time", 0)
    results["peak_vram_gb"] = gen_metrics.get("peak_vram_gb", 0)
    
    # Compute CLIP scores
    if compute_clip and OPEN_CLIP_AVAILABLE:
        print("\n=== Computing CLIP Scores ===")
        try:
            clip_scorer = CLIPScorer(device=device)
            clip_scores = clip_scorer.compute_batch_scores(images, prompts)
            results["mean_clip_score"] = np.mean(clip_scores)
            results["std_clip_score"] = np.std(clip_scores)
            del clip_scorer
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"CLIP score computation failed: {e}")
            results["mean_clip_score"] = None
    
    # Compute FID
    if compute_fid and reference_images and INCEPTION_AVAILABLE and SCIPY_AVAILABLE:
        print("\n=== Computing FID Score ===")
        try:
            fid_calc = FIDCalculator(device=device)
            fid_score = fid_calc.compute_fid_from_images(images, reference_images)
            results["fid_score"] = fid_score
            del fid_calc
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"FID computation failed: {e}")
            results["fid_score"] = None
    
    return results, images
