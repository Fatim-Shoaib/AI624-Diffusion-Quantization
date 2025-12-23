"""
Evaluation utilities for Qronos-DiT.

Provides functions for:
- CLIP score computation
- Image generation and visualization
- Peak VRAM measurement
"""
import os
import gc
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import json


def measure_peak_vram() -> float:
    """
    Measure peak VRAM usage in GB.
    
    Returns:
        Peak VRAM usage in GB
    """
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 3)
    return 0.0


def reset_vram_stats():
    """Reset VRAM statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        gc.collect()


class CLIPScorer:
    """
    CLIP Score calculator for image-text similarity.
    """
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cuda"):
        """
        Initialize CLIP scorer.
        
        Args:
            model_name: HuggingFace CLIP model name
            device: Device to use
        """
        self.device = device
        self.model_name = model_name
        self.model = None
        self.processor = None
    
    def load_model(self):
        """Load CLIP model."""
        from transformers import CLIPProcessor, CLIPModel
        
        print(f"Loading CLIP model: {self.model_name}")
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
    
    @torch.no_grad()
    def compute_score(self, images: List[Image.Image], texts: List[str]) -> List[float]:
        """
        Compute CLIP scores for image-text pairs.
        
        Args:
            images: List of PIL images
            texts: List of text prompts
        
        Returns:
            List of CLIP scores (cosine similarity * 100)
        """
        if self.model is None:
            self.load_model()
        
        scores = []
        
        for image, text in zip(images, texts):
            inputs = self.processor(
                text=[text],
                images=[image],
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            outputs = self.model(**inputs)
            
            # Compute cosine similarity
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds
            
            similarity = torch.nn.functional.cosine_similarity(
                image_embeds, text_embeds
            ).item()
            
            # Scale to 0-100
            scores.append(similarity * 100)
        
        return scores
    
    def compute_average_score(self, images: List[Image.Image], texts: List[str]) -> float:
        """
        Compute average CLIP score.
        
        Args:
            images: List of PIL images
            texts: List of text prompts
        
        Returns:
            Average CLIP score
        """
        scores = self.compute_score(images, texts)
        return sum(scores) / len(scores) if scores else 0.0


class ImageGenerator:
    """
    Image generator for evaluation.
    """
    
    def __init__(self, pipe, device: str = "cuda"):
        """
        Initialize image generator.
        
        Args:
            pipe: Diffusers pipeline
            device: Device to use
        """
        self.pipe = pipe
        self.device = device
    
    @torch.no_grad()
    def generate_images(
        self,
        prompts: List[str],
        num_inference_steps: int = 20,
        guidance_scale: float = 4.5,
        seed: int = 42,
    ) -> List[Image.Image]:
        """
        Generate images from prompts.
        
        Args:
            prompts: List of text prompts
            num_inference_steps: Number of diffusion steps
            guidance_scale: Classifier-free guidance scale
            seed: Random seed for reproducibility
        
        Returns:
            List of generated PIL images
        """
        self.pipe.to(self.device)
        
        images = []
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        for prompt in tqdm(prompts, desc="Generating images"):
            image = self.pipe(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images[0]
            
            images.append(image)
            
            # Reset generator for reproducibility
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        return images
    
    def save_images(
        self,
        images: List[Image.Image],
        prompts: List[str],
        output_dir: str,
        prefix: str = "",
    ):
        """
        Save generated images.
        
        Args:
            images: List of PIL images
            prompts: List of prompts (for naming)
            output_dir: Output directory
            prefix: Filename prefix
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, (image, prompt) in enumerate(zip(images, prompts)):
            # Create safe filename from prompt
            safe_prompt = "".join(c if c.isalnum() else "_" for c in prompt[:50])
            filename = f"{prefix}_{i:04d}_{safe_prompt}.png"
            
            image.save(output_dir / filename)
        
        # Save prompt list
        with open(output_dir / f"{prefix}_prompts.json", 'w') as f:
            json.dump(prompts, f, indent=2)
        
        print(f"Saved {len(images)} images to {output_dir}")


class Evaluator:
    """
    Full evaluation pipeline for quantized models.
    """
    
    def __init__(
        self,
        fp16_pipe,
        quantized_pipe,
        device: str = "cuda",
        output_dir: str = "./evaluation_results",
    ):
        """
        Initialize evaluator.
        
        Args:
            fp16_pipe: Original FP16 pipeline
            quantized_pipe: Quantized pipeline
            device: Device to use
            output_dir: Output directory for results
        """
        self.fp16_pipe = fp16_pipe
        self.quantized_pipe = quantized_pipe
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.clip_scorer = CLIPScorer(device=device)
        self.fp16_generator = ImageGenerator(fp16_pipe, device)
        self.quant_generator = ImageGenerator(quantized_pipe, device)
    
    def evaluate(
        self,
        prompts: List[str],
        num_inference_steps: int = 20,
        guidance_scale: float = 4.5,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Run full evaluation.
        
        Args:
            prompts: List of evaluation prompts
            num_inference_steps: Number of diffusion steps
            guidance_scale: Classifier-free guidance scale
            seed: Random seed
        
        Returns:
            Dictionary with evaluation results
        """
        results = {}
        
        # === FP16 Baseline ===
        print("\n" + "=" * 60)
        print("Evaluating FP16 Baseline")
        print("=" * 60)
        
        reset_vram_stats()
        
        fp16_images = self.fp16_generator.generate_images(
            prompts, num_inference_steps, guidance_scale, seed
        )
        
        fp16_vram = measure_peak_vram()
        fp16_clip_scores = self.clip_scorer.compute_score(fp16_images, prompts)
        
        self.fp16_generator.save_images(
            fp16_images, prompts, self.output_dir / "fp16_images", "fp16"
        )
        
        results['fp16'] = {
            'clip_scores': fp16_clip_scores,
            'avg_clip_score': sum(fp16_clip_scores) / len(fp16_clip_scores),
            'peak_vram_gb': fp16_vram,
        }
        
        # Clear GPU memory
        gc.collect()
        torch.cuda.empty_cache()
        
        # === Quantized Model ===
        print("\n" + "=" * 60)
        print("Evaluating Quantized Model")
        print("=" * 60)
        
        reset_vram_stats()
        
        quant_images = self.quant_generator.generate_images(
            prompts, num_inference_steps, guidance_scale, seed
        )
        
        quant_vram = measure_peak_vram()
        quant_clip_scores = self.clip_scorer.compute_score(quant_images, prompts)
        
        self.quant_generator.save_images(
            quant_images, prompts, self.output_dir / "quantized_images", "quant"
        )
        
        results['quantized'] = {
            'clip_scores': quant_clip_scores,
            'avg_clip_score': sum(quant_clip_scores) / len(quant_clip_scores),
            'peak_vram_gb': quant_vram,
        }
        
        # === Comparison ===
        results['comparison'] = {
            'clip_score_diff': results['quantized']['avg_clip_score'] - results['fp16']['avg_clip_score'],
            'vram_reduction': fp16_vram - quant_vram,
            'vram_reduction_pct': (fp16_vram - quant_vram) / fp16_vram * 100 if fp16_vram > 0 else 0,
        }
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _save_results(self, results: Dict[str, Any]):
        """Save evaluation results to file."""
        # Convert to JSON-serializable format
        json_results = {
            'fp16': {
                'avg_clip_score': results['fp16']['avg_clip_score'],
                'peak_vram_gb': results['fp16']['peak_vram_gb'],
            },
            'quantized': {
                'avg_clip_score': results['quantized']['avg_clip_score'],
                'peak_vram_gb': results['quantized']['peak_vram_gb'],
            },
            'comparison': results['comparison'],
        }
        
        with open(self.output_dir / "evaluation_results.json", 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # Print summary
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"FP16 CLIP Score:      {results['fp16']['avg_clip_score']:.2f}")
        print(f"Quantized CLIP Score: {results['quantized']['avg_clip_score']:.2f}")
        print(f"CLIP Score Diff:      {results['comparison']['clip_score_diff']:.2f}")
        print(f"FP16 Peak VRAM:       {results['fp16']['peak_vram_gb']:.2f} GB")
        print(f"Quantized Peak VRAM:  {results['quantized']['peak_vram_gb']:.2f} GB")
        print(f"VRAM Reduction:       {results['comparison']['vram_reduction_pct']:.1f}%")
        print("=" * 60)


def create_side_by_side_comparison(
    fp16_image: Image.Image,
    quant_image: Image.Image,
    prompt: str,
    output_path: str,
):
    """
    Create a side-by-side comparison image.
    
    Args:
        fp16_image: FP16 generated image
        quant_image: Quantized generated image
        prompt: Text prompt
        output_path: Output file path
    """
    # Get dimensions
    width, height = fp16_image.size
    
    # Create combined image
    combined = Image.new('RGB', (width * 2 + 20, height + 50), color='white')
    
    # Paste images
    combined.paste(fp16_image, (0, 40))
    combined.paste(quant_image, (width + 20, 40))
    
    # Add labels (requires PIL ImageDraw)
    try:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(combined)
        
        # Try to use a default font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        draw.text((10, 10), "FP16", fill='black', font=font)
        draw.text((width + 30, 10), "Quantized", fill='black', font=font)
    except ImportError:
        pass
    
    combined.save(output_path)


def load_coco_captions(
    annotations_file: str,
    num_samples: int = 256,
    seed: int = 42,
) -> List[str]:
    """
    Load captions from COCO annotations file.
    
    Args:
        annotations_file: Path to COCO captions JSON file
        num_samples: Number of samples to load
        seed: Random seed for sampling
    
    Returns:
        List of caption strings
    """
    import random
    
    with open(annotations_file, 'r') as f:
        data = json.load(f)
    
    captions = [ann['caption'] for ann in data['annotations']]
    
    # Sample randomly
    random.seed(seed)
    sampled = random.sample(captions, min(num_samples, len(captions)))
    
    return sampled


def get_default_eval_prompts(num_prompts: int = 50) -> List[str]:
    """
    Get default evaluation prompts.
    
    Args:
        num_prompts: Number of prompts to return
    
    Returns:
        List of prompt strings
    """
    prompts = [
        "A photo of a cat sitting on a windowsill",
        "A beautiful sunset over the ocean",
        "A modern city skyline at night",
        "A peaceful forest path in autumn",
        "A delicious pizza on a wooden table",
        "A cute puppy playing in the park",
        "A mountain landscape with snow-capped peaks",
        "A colorful bouquet of flowers",
        "A vintage car parked on a street",
        "A cozy coffee shop interior",
        "A tropical beach with palm trees",
        "A starry night sky over a desert",
        "A bowl of fresh fruit on a kitchen counter",
        "A person reading a book in a library",
        "A butterfly on a flower",
        "A rainy day in a European city",
        "A beautiful garden with roses",
        "A modern living room interior",
        "A waterfall in a tropical jungle",
        "A street market with colorful goods",
        "A hot air balloon over a valley",
        "A snowy cabin in the mountains",
        "A lighthouse on a rocky coast",
        "A field of sunflowers",
        "A medieval castle on a hill",
        "A Japanese garden with a koi pond",
        "A colorful coral reef underwater",
        "A steaming cup of coffee",
        "A bicycle by a canal",
        "A fireworks display over a city",
        "A peaceful lake at dawn",
        "A busy train station",
        "A beautiful wedding cake",
        "A herd of elephants in the savanna",
        "A futuristic robot",
        "A cozy bedroom with soft lighting",
        "A plate of sushi",
        "A cherry blossom tree in spring",
        "A vintage typewriter",
        "A beautiful waterfall",
        "A cute baby panda",
        "A colorful hot air balloon festival",
        "A peaceful zen garden",
        "A stunning northern lights display",
        "A delicious chocolate cake",
        "A majestic lion",
        "A beautiful mosque",
        "A cozy fireplace",
        "A stunning aurora borealis",
        "A colorful parrot",
    ]
    
    return prompts[:num_prompts]
