"""
=============================================================================
SD 3.5 Medium GPTQ Baseline - Configuration
=============================================================================
Central configuration file for all constants and default settings.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Base directories
PROJECT_ROOT = Path(__file__).parent
CALIBRATION_DATA_DIR = PROJECT_ROOT / "calibration_data"
QUANTIZED_MODEL_DIR = PROJECT_ROOT / "quantized_model"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RESULTS_DIR = PROJECT_ROOT / "results"
PROMPTS_DIR = PROJECT_ROOT / "prompts"

# Create directories if they don't exist
for dir_path in [CALIBRATION_DATA_DIR, QUANTIZED_MODEL_DIR, OUTPUTS_DIR, RESULTS_DIR, PROMPTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for SD 3.5 Medium model."""
    
    # Model identifier on Hugging Face
    model_id: str = "stabilityai/stable-diffusion-3.5-medium"
    
    # Model components precision
    transformer_dtype: str = "float16"  # Main transformer
    text_encoder_dtype: str = "float16"  # CLIP text encoders
    text_encoder_3_dtype: str = "float16"  # T5 encoder (can use float8 for memory)
    vae_dtype: str = "float16"
    
    # Generation defaults
    default_height: int = 1024
    default_width: int = 1024
    default_num_inference_steps: int = 28  # SD 3.5 default
    default_guidance_scale: float = 4.5  # SD 3.5 recommended
    
    # Memory optimization
    enable_attention_slicing: bool = False  # Not needed with 24GB
    enable_vae_slicing: bool = True
    enable_model_cpu_offload: bool = False  # Disabled as per user request


# =============================================================================
# QUANTIZATION CONFIGURATION
# =============================================================================

@dataclass
class QuantizationConfig:
    """Configuration for GPTQ quantization."""
    
    # Bit-widths
    weight_bits: int = 4  # W4
    activation_bits: int = 8  # A8
    
    # GPTQ parameters
    group_size: int = 128  # Quantization group size
    actorder: bool = False  # Activation ordering (can improve quality but slower)
    percdamp: float = 0.01  # Dampening percentage for Hessian
    block_size: int = 128  # Block size for lazy batch updates
    
    # Symmetry settings
    weight_symmetric: bool = True
    activation_symmetric: bool = False
    
    # Calibration
    num_calibration_samples: int = 256
    calibration_batch_size: int = 4
    calibration_sequence_length: int = 77  # CLIP max length
    
    # Which components to quantize
    quantize_transformer: bool = True
    quantize_text_encoder: bool = False  # Keep text encoders in FP16
    quantize_vae: bool = False  # Keep VAE in FP16
    
    # Layers to skip (regex patterns)
    skip_layers: List[str] = field(default_factory=lambda: [
        "time_embed",  # Time embeddings are sensitive
        "label_embed",  # Label embeddings
        "proj_out",  # Final projection layers
    ])


# =============================================================================
# EVALUATION CONFIGURATION
# =============================================================================

@dataclass
class EvaluationConfig:
    """Configuration for benchmark evaluation."""
    
    # FID calculation
    num_fid_images: int = 5000  # Number of images for FID
    fid_batch_size: int = 4
    reference_dataset: str = "coco2017"  # Options: coco2017, coco2014
    
    # CLIP score
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"
    
    # Visual inspection
    num_visual_samples: int = 50
    visual_prompts_file: str = str(PROMPTS_DIR / "visual_inspection.txt")
    
    # Generation settings for evaluation
    eval_num_inference_steps: int = 28
    eval_guidance_scale: float = 4.5
    eval_seed: int = 42  # Fixed seed for reproducibility
    
    # Output settings
    save_individual_images: bool = True
    save_comparison_grid: bool = True
    
    # VRAM tracking
    track_vram: bool = True
    vram_sample_interval: float = 0.1  # seconds


# =============================================================================
# COCO DATASET CONFIGURATION
# =============================================================================

@dataclass
class COCOConfig:
    """Configuration for COCO dataset handling."""
    
    # Dataset paths (will be downloaded if not present)
    coco_year: str = "2017"
    split: str = "val"
    
    # Annotation file URL
    annotations_url: str = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    
    # Number of captions to use
    max_captions: int = 5000
    
    # Caption preprocessing
    min_caption_length: int = 10
    max_caption_length: int = 200


# =============================================================================
# DEFAULT INSTANCES
# =============================================================================

# Create default config instances
MODEL_CONFIG = ModelConfig()
QUANT_CONFIG = QuantizationConfig()
EVAL_CONFIG = EvaluationConfig()
COCO_CONFIG = COCOConfig()


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "level": "INFO",
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": str(RESULTS_DIR / "baseline.log"),
            "formatter": "standard",
            "level": "DEBUG",
        },
    },
    "root": {
        "handlers": ["console", "file"],
        "level": "INFO",
    },
}


# =============================================================================
# VISUAL INSPECTION PROMPTS
# =============================================================================

# Default prompts for visual inspection (will be written to file)
DEFAULT_VISUAL_PROMPTS = [
    # Photorealistic - People
    "A professional portrait photograph of a woman with curly auburn hair, soft studio lighting, sharp focus",
    "An elderly man with weathered hands playing chess in a sunlit park, documentary photography style",
    
    # Photorealistic - Nature
    "A majestic snow-capped mountain reflected in a crystal clear alpine lake at golden hour",
    "Close-up macro photography of morning dew drops on a spider web, bokeh background",
    "A dense forest with sunbeams filtering through the canopy, mystical atmosphere",
    
    # Photorealistic - Animals
    "A red fox in a snowy forest, wildlife photography, National Geographic style",
    "An owl perched on a branch at twilight, detailed feathers, piercing yellow eyes",
    
    # Photorealistic - Urban
    "A bustling Tokyo street at night with neon signs and light trails, cyberpunk atmosphere",
    "A cozy coffee shop interior with warm lighting, wooden furniture, and steaming cups",
    "Modern glass skyscraper reflecting sunset clouds, architectural photography",
    
    # Artistic - Paintings
    "An oil painting of a stormy seascape in the style of J.M.W. Turner, dramatic waves",
    "A watercolor illustration of a Japanese garden with cherry blossoms and a red bridge",
    "An impressionist painting of a Parisian café terrace at night, Vincent van Gogh style",
    
    # Artistic - Fantasy/Sci-Fi
    "A massive steampunk airship floating above Victorian London, detailed brass and copper",
    "An alien planet landscape with bioluminescent plants and two moons in the sky",
    "A cyberpunk samurai standing in the rain, neon reflections on armor, cinematic lighting",
    "A mystical wizard's library with floating books and magical artifacts, fantasy art",
    
    # Artistic - Abstract/Surreal
    "A surrealist dreamscape with melting clocks and floating islands, Salvador Dali inspired",
    "Abstract geometric art with flowing liquid metal and prismatic light reflections",
    
    # Objects/Still Life
    "A vintage camera, old photographs, and dried flowers on a rustic wooden table",
    "Fresh fruits arranged on a marble counter, dramatic side lighting, food photography",
    "An antique pocket watch with intricate mechanical details, macro photography",
    
    # Complex Compositions
    "A medieval marketplace bustling with merchants, knights, and common folk, detailed scene",
    "An underwater coral reef ecosystem with diverse tropical fish and sea creatures",
    "A space station orbiting Earth with astronauts performing a spacewalk",
    
    # Text Rendering (challenging for diffusion models)
    "A neon sign saying 'OPEN' on a rainy city street at night",
    "A vintage travel poster for 'PARIS' featuring the Eiffel Tower",
    
    # Challenging Elements
    "Human hands holding a transparent glass sphere with a miniature world inside",
    "A reflection of a mountain landscape in a person's sunglasses",
    "A cat and its reflection in a mirror, both looking at the viewer",
    
    # Lighting Challenges
    "A portrait lit only by candlelight, chiaroscuro effect, Renaissance painting style",
    "Silhouette of a dancer against a bright sunset, dramatic backlighting",
    "A product photograph of a perfume bottle with caustic light patterns",
    
    # Texture/Material Focus
    "Close-up of weathered leather journal with gold embossing and aged pages",
    "Detailed shot of woven textile with intricate patterns and vibrant colors",
    "Ice crystals forming on a window pane, macro photography, blue hour light",
    
    # Seasonal/Weather
    "A cozy cabin in a snowy forest during a gentle snowfall, warm light from windows",
    "A thunderstorm over vast wheat fields, dramatic lightning strike in the distance",
    "Cherry blossom petals falling along a Japanese river path in spring",
    
    # Food Photography
    "A gourmet burger with melting cheese and fresh vegetables, professional food photography",
    "An elegant sushi platter on a black slate board, Japanese cuisine photography",
    "Steaming bowl of ramen with perfectly cooked egg, chopsticks lifting noodles",
    
    # Architecture
    "Gothic cathedral interior with sunbeams through stained glass windows",
    "Modern minimalist house with floor-to-ceiling windows overlooking the ocean",
    "Ancient ruins of a Greek temple at sunset, golden light on marble columns",
    
    # Cinematic/Movie Stills
    "A noir detective in a smoky office, venetian blind shadows, 1940s aesthetic",
    "An epic battle scene with armies clashing, cinematic wide shot, Lord of the Rings style",
    
    # Cute/Whimsical
    "A tiny mouse reading a tiny book under a mushroom, fairytale illustration style",
    "A fluffy corgi puppy wearing a tiny crown, sitting on a velvet cushion",
]
