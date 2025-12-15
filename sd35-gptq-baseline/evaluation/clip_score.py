"""
=============================================================================
CLIP Score Calculation
=============================================================================

CLIP Score measures the alignment between generated images and their text prompts
using the CLIP (Contrastive Language-Image Pre-training) model.

Higher CLIP Score = Better text-image alignment

We use OpenCLIP for compatibility with PyTorch 2.6+ security requirements.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Union, Optional, Tuple
from pathlib import Path
import logging
from tqdm import tqdm
from PIL import Image

import open_clip

logger = logging.getLogger(__name__)


class CLIPScoreCalculator:
    """
    CLIP Score calculator using OpenCLIP.
    
    Calculates the cosine similarity between image and text embeddings
    as produced by a CLIP model.
    """
    
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: torch.device = torch.device("cuda"),
    ):
        """
        Initialize CLIP score calculator.
        
        Args:
            model_name: CLIP model architecture (e.g., "ViT-B-32", "ViT-L-14")
            pretrained: Pretrained weights to use
            device: Device for computation
        """
        self.device = device
        
        logger.info(f"Loading CLIP model: {model_name} ({pretrained})")
        
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=device,
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        
        self.model.eval()
        
        logger.info("CLIP model loaded successfully")
    
    @torch.no_grad()
    def encode_images(
        self,
        images: Union[List[Image.Image], torch.Tensor],
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Encode images to CLIP feature space.
        
        Args:
            images: List of PIL images or tensor [B, 3, H, W]
            normalize: Whether to L2 normalize features
            
        Returns:
            Image features [B, D]
        """
        if isinstance(images, list):
            # Preprocess PIL images
            tensors = [self.preprocess(img).unsqueeze(0) for img in images]
            images = torch.cat(tensors, dim=0)
        
        images = images.to(self.device)
        
        features = self.model.encode_image(images)
        
        if normalize:
            features = F.normalize(features, dim=-1)
        
        return features
    
    @torch.no_grad()
    def encode_text(
        self,
        texts: List[str],
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Encode text prompts to CLIP feature space.
        
        Args:
            texts: List of text prompts
            normalize: Whether to L2 normalize features
            
        Returns:
            Text features [B, D]
        """
        tokens = self.tokenizer(texts).to(self.device)
        
        features = self.model.encode_text(tokens)
        
        if normalize:
            features = F.normalize(features, dim=-1)
        
        return features
    
    @torch.no_grad()
    def calculate_score(
        self,
        images: Union[List[Image.Image], torch.Tensor],
        texts: List[str],
    ) -> Tuple[float, List[float]]:
        """
        Calculate CLIP score between images and their corresponding texts.
        
        Args:
            images: List of PIL images or tensor
            texts: List of text prompts (one per image)
            
        Returns:
            Tuple of (mean_score, individual_scores)
        """
        assert len(images) == len(texts), "Number of images and texts must match"
        
        # Encode
        image_features = self.encode_images(images)
        text_features = self.encode_text(texts)
        
        # Compute cosine similarity (features are already normalized)
        # For matching pairs, we compute diagonal of similarity matrix
        scores = (image_features * text_features).sum(dim=-1)
        
        # Convert to list of floats
        individual_scores = scores.cpu().tolist()
        mean_score = float(scores.mean())
        
        return mean_score, individual_scores
    
    @torch.no_grad()
    def calculate_score_batch(
        self,
        images: Union[Path, List[Image.Image]],
        texts: List[str],
        batch_size: int = 32,
    ) -> Tuple[float, List[float]]:
        """
        Calculate CLIP score for a large batch of images and texts.
        
        Args:
            images: Path to image folder or list of PIL images
            texts: List of text prompts
            batch_size: Batch size for processing
            
        Returns:
            Tuple of (mean_score, individual_scores)
        """
        # Load images if path provided
        if isinstance(images, (str, Path)):
            images = Path(images)
            image_files = sorted([
                f for f in images.iterdir()
                if f.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp'}
            ])
            pil_images = [Image.open(f).convert('RGB') for f in image_files]
        else:
            pil_images = images
        
        assert len(pil_images) == len(texts), \
            f"Number of images ({len(pil_images)}) != number of texts ({len(texts)})"
        
        all_scores = []
        
        for i in tqdm(range(0, len(pil_images), batch_size), desc="Computing CLIP scores"):
            batch_images = pil_images[i:i + batch_size]
            batch_texts = texts[i:i + batch_size]
            
            _, scores = self.calculate_score(batch_images, batch_texts)
            all_scores.extend(scores)
        
        mean_score = float(np.mean(all_scores))
        
        return mean_score, all_scores
    
    @torch.no_grad()
    def calculate_image_quality_score(
        self,
        images: Union[List[Image.Image], torch.Tensor],
        quality_prompts: Optional[List[str]] = None,
    ) -> Tuple[float, List[float]]:
        """
        Calculate image quality score using generic quality prompts.
        
        This measures general image quality rather than text alignment.
        
        Args:
            images: Images to evaluate
            quality_prompts: Custom quality prompts (default: built-in prompts)
            
        Returns:
            Tuple of (mean_quality_score, individual_scores)
        """
        if quality_prompts is None:
            quality_prompts = [
                "a high quality, detailed image",
                "a professional photograph",
                "a sharp, clear image with good lighting",
                "an aesthetically pleasing image",
            ]
        
        # Encode images
        image_features = self.encode_images(images)  # [N, D]
        
        # Encode quality prompts
        text_features = self.encode_text(quality_prompts)  # [M, D]
        
        # Compute similarity to each quality prompt, then average
        # [N, D] @ [D, M] -> [N, M]
        similarities = image_features @ text_features.T
        
        # Average over prompts for each image
        scores = similarities.mean(dim=-1)
        
        individual_scores = scores.cpu().tolist()
        mean_score = float(scores.mean())
        
        return mean_score, individual_scores


def calculate_clip_score(
    images: Union[Path, List[Image.Image]],
    texts: List[str],
    model_name: str = "ViT-B-32",
    device: str = "cuda",
    batch_size: int = 32,
) -> float:
    """
    Convenience function to calculate CLIP score.
    
    Args:
        images: Path to images folder or list of PIL images
        texts: List of text prompts (one per image)
        model_name: CLIP model to use
        device: Device for computation
        batch_size: Batch size for processing
        
    Returns:
        Mean CLIP score
    """
    calculator = CLIPScoreCalculator(
        model_name=model_name,
        device=torch.device(device),
    )
    
    mean_score, _ = calculator.calculate_score_batch(
        images, texts, batch_size=batch_size
    )
    
    return mean_score


def calculate_clip_similarity_matrix(
    images: List[Image.Image],
    texts: List[str],
    model_name: str = "ViT-B-32",
    device: str = "cuda",
) -> np.ndarray:
    """
    Calculate full similarity matrix between images and texts.
    
    Useful for retrieval metrics and detailed analysis.
    
    Args:
        images: List of PIL images
        texts: List of text prompts
        model_name: CLIP model to use
        device: Device for computation
        
    Returns:
        Similarity matrix [num_images, num_texts]
    """
    calculator = CLIPScoreCalculator(
        model_name=model_name,
        device=torch.device(device),
    )
    
    image_features = calculator.encode_images(images)
    text_features = calculator.encode_text(texts)
    
    similarity = (image_features @ text_features.T).cpu().numpy()
    
    return similarity
