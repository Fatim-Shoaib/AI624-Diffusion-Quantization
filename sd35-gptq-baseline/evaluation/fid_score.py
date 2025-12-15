"""
=============================================================================
FID Score Calculation
=============================================================================

Fréchet Inception Distance (FID) measures the quality of generated images
by comparing the statistics of generated images to real images in the 
feature space of an Inception network.

Lower FID = Better quality (closer to real image distribution)

Reference: https://arxiv.org/abs/1706.08500
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List, Union
from pathlib import Path
import logging
from tqdm import tqdm
from PIL import Image
import os

from scipy import linalg
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights

logger = logging.getLogger(__name__)


class InceptionV3Features(nn.Module):
    """
    Inception V3 model modified to output features for FID calculation.
    
    Outputs the 2048-dimensional features from the final pooling layer.
    """
    
    def __init__(self, device: torch.device = torch.device("cuda")):
        super().__init__()
        
        # Load pretrained InceptionV3
        self.model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        self.model.fc = nn.Identity()  # Remove classification head
        self.model.eval()
        self.model = self.model.to(device)
        self.device = device
        
        # Preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((299, 299), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from images.
        
        Args:
            x: Batch of images [B, 3, H, W] normalized to [0, 1]
            
        Returns:
            Features [B, 2048]
        """
        # Resize to Inception input size
        x = nn.functional.interpolate(
            x, size=(299, 299), mode='bilinear', align_corners=False
        )
        
        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        x = (x - mean) / std
        
        return self.model(x)
    
    def get_features_from_pil(self, images: List[Image.Image]) -> np.ndarray:
        """
        Extract features from a list of PIL images.
        
        Args:
            images: List of PIL images
            
        Returns:
            Features array [N, 2048]
        """
        features = []
        
        for img in tqdm(images, desc="Extracting features"):
            # Ensure RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Transform
            x = self.transform(img).unsqueeze(0).to(self.device)
            
            # Extract features
            feat = self.model(x).cpu().numpy()
            features.append(feat)
        
        return np.concatenate(features, axis=0)
    
    def get_features_from_folder(
        self,
        folder: Path,
        batch_size: int = 32,
        max_images: Optional[int] = None,
    ) -> np.ndarray:
        """
        Extract features from all images in a folder.
        
        Args:
            folder: Path to folder containing images
            batch_size: Batch size for processing
            max_images: Maximum number of images to process
            
        Returns:
            Features array [N, 2048]
        """
        folder = Path(folder)
        
        # Find all images
        extensions = {'.png', '.jpg', '.jpeg', '.webp'}
        image_files = [
            f for f in folder.iterdir()
            if f.suffix.lower() in extensions
        ]
        
        if max_images:
            image_files = image_files[:max_images]
        
        logger.info(f"Processing {len(image_files)} images from {folder}")
        
        features = []
        
        for i in tqdm(range(0, len(image_files), batch_size), desc="Extracting features"):
            batch_files = image_files[i:i + batch_size]
            batch_tensors = []
            
            for f in batch_files:
                img = Image.open(f).convert('RGB')
                x = self.transform(img)
                batch_tensors.append(x)
            
            batch = torch.stack(batch_tensors).to(self.device)
            
            with torch.no_grad():
                feat = self.model(batch).cpu().numpy()
            
            features.append(feat)
        
        return np.concatenate(features, axis=0)


def calculate_frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """
    Calculate Fréchet Distance between two multivariate Gaussians.
    
    FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
    
    Args:
        mu1, sigma1: Mean and covariance of first distribution
        mu2, sigma2: Mean and covariance of second distribution
        eps: Small constant for numerical stability
        
    Returns:
        FID score (float)
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    assert mu1.shape == mu2.shape, "Means have different lengths"
    assert sigma1.shape == sigma2.shape, "Covariances have different shapes"
    
    diff = mu1 - mu2
    
    # Product of covariances
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    # Numerical stability
    if not np.isfinite(covmean).all():
        logger.warning(
            "FID calculation produced singular product; adding eps to diagonal"
        )
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    # Handle imaginary components from numerical errors
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            logger.warning(f"Imaginary component {m}")
        covmean = covmean.real
    
    tr_covmean = np.trace(covmean)
    
    return (
        diff.dot(diff) +
        np.trace(sigma1) +
        np.trace(sigma2) -
        2 * tr_covmean
    )


def compute_statistics(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and covariance statistics from features.
    
    Args:
        features: Feature array [N, D]
        
    Returns:
        Tuple of (mean [D], covariance [D, D])
    """
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


class FIDCalculator:
    """
    FID Calculator with caching support for reference statistics.
    """
    
    def __init__(
        self,
        device: torch.device = torch.device("cuda"),
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize FID calculator.
        
        Args:
            device: Device for Inception model
            cache_dir: Directory to cache reference statistics
        """
        self.device = device
        self.inception = InceptionV3Features(device)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache for reference statistics
        self._ref_stats_cache = {}
    
    def get_reference_statistics(
        self,
        reference_path: Union[str, Path],
        max_images: int = 5000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get or compute reference statistics.
        
        Args:
            reference_path: Path to reference images folder or .npz file
            max_images: Maximum number of images to use
            
        Returns:
            Tuple of (mean, covariance)
        """
        reference_path = Path(reference_path)
        cache_key = f"{reference_path.name}_{max_images}"
        
        # Check memory cache
        if cache_key in self._ref_stats_cache:
            return self._ref_stats_cache[cache_key]
        
        # Check disk cache
        if self.cache_dir:
            cache_file = self.cache_dir / f"fid_stats_{cache_key}.npz"
            if cache_file.exists():
                logger.info(f"Loading cached statistics from {cache_file}")
                data = np.load(cache_file)
                mu, sigma = data['mu'], data['sigma']
                self._ref_stats_cache[cache_key] = (mu, sigma)
                return mu, sigma
        
        # Compute statistics
        if reference_path.suffix == '.npz':
            # Load pre-computed statistics
            data = np.load(reference_path)
            if 'mu' in data and 'sigma' in data:
                mu, sigma = data['mu'], data['sigma']
            else:
                # Assume it contains images as arr_0
                images = data['arr_0']
                features = self._compute_features_from_array(images)
                mu, sigma = compute_statistics(features)
        else:
            # Compute from folder
            features = self.inception.get_features_from_folder(
                reference_path, max_images=max_images
            )
            mu, sigma = compute_statistics(features)
        
        # Cache to disk
        if self.cache_dir:
            np.savez(cache_file, mu=mu, sigma=sigma)
            logger.info(f"Cached statistics to {cache_file}")
        
        self._ref_stats_cache[cache_key] = (mu, sigma)
        return mu, sigma
    
    def _compute_features_from_array(
        self,
        images: np.ndarray,
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Compute features from numpy array of images.
        
        Args:
            images: Images array [N, H, W, 3] in uint8
            batch_size: Batch size for processing
            
        Returns:
            Features [N, 2048]
        """
        features = []
        
        for i in tqdm(range(0, len(images), batch_size), desc="Computing features"):
            batch = images[i:i + batch_size]
            
            # Convert to tensor [B, 3, H, W] normalized to [0, 1]
            batch = torch.from_numpy(batch).permute(0, 3, 1, 2).float() / 255.0
            batch = batch.to(self.device)
            
            with torch.no_grad():
                feat = self.inception(batch).cpu().numpy()
            
            features.append(feat)
        
        return np.concatenate(features, axis=0)
    
    def calculate_fid(
        self,
        generated_images: Union[Path, List[Image.Image], np.ndarray],
        reference_path: Union[str, Path],
        max_images: int = 5000,
    ) -> float:
        """
        Calculate FID between generated images and reference.
        
        Args:
            generated_images: Generated images (folder path, list of PIL, or numpy array)
            reference_path: Path to reference images
            max_images: Maximum images to use
            
        Returns:
            FID score
        """
        # Get generated features
        if isinstance(generated_images, (str, Path)):
            gen_features = self.inception.get_features_from_folder(
                generated_images, max_images=max_images
            )
        elif isinstance(generated_images, list):
            gen_features = self.inception.get_features_from_pil(generated_images)
        elif isinstance(generated_images, np.ndarray):
            gen_features = self._compute_features_from_array(generated_images)
        else:
            raise ValueError(f"Unsupported type: {type(generated_images)}")
        
        gen_mu, gen_sigma = compute_statistics(gen_features)
        
        # Get reference statistics
        ref_mu, ref_sigma = self.get_reference_statistics(reference_path, max_images)
        
        # Calculate FID
        fid = calculate_frechet_distance(gen_mu, gen_sigma, ref_mu, ref_sigma)
        
        return float(fid)


def calculate_fid(
    generated_path: Union[str, Path],
    reference_path: Union[str, Path],
    device: str = "cuda",
    max_images: int = 5000,
) -> float:
    """
    Convenience function to calculate FID between two image sets.
    
    Args:
        generated_path: Path to generated images folder
        reference_path: Path to reference images folder or stats file
        device: Device for computation
        max_images: Maximum images to use
        
    Returns:
        FID score
    """
    calculator = FIDCalculator(device=torch.device(device))
    return calculator.calculate_fid(generated_path, reference_path, max_images)
