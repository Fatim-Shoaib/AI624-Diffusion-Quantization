"""
=============================================================================
Configuration for Qronos Quantization of SD 3.5 Medium
=============================================================================

This configuration matches the GPTQ baseline for fair comparison.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"
QUANTIZATION_DIR = PROJECT_ROOT / "quantization"
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CALIBRATION_DIR = PROJECT_ROOT / "calibration_data"
QUANTIZED_MODEL_DIR = PROJECT_ROOT / "quantized_model"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories
for dir_path in [OUTPUTS_DIR, CALIBRATION_DIR, QUANTIZED_MODEL_DIR, RESULTS_DIR, PROMPTS_DIR]:
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
    transformer_dtype: str = "float16"
    text_encoder_dtype: str = "float16"
    text_encoder_3_dtype: str = "float16"  # T5 encoder
    vae_dtype: str = "float16"
    
    # Generation defaults (SD 3.5 recommended)
    default_height: int = 1024
    default_width: int = 1024
    default_num_inference_steps: int = 28
    default_guidance_scale: float = 4.5
    
    # Memory optimization
    enable_attention_slicing: bool = False  # Not needed with 24GB
    enable_vae_slicing: bool = True
    enable_model_cpu_offload: bool = False


# =============================================================================
# QRONOS QUANTIZATION CONFIGURATION
# =============================================================================

@dataclass
class QronosConfig:
    """Configuration for Qronos quantization (matches GPTQ baseline where applicable)."""
    
    # Bit-widths (same as GPTQ baseline)
    weight_bits: int = 4  # W4
    activation_bits: int = 8  # A8
    
    # Quantization parameters (same as GPTQ baseline)
    group_size: int = 128
    weight_symmetric: bool = True
    activation_symmetric: bool = False
    
    # Qronos-specific parameters
    percdamp: float = 1e-5  # Regularization using spectral norm (different from GPTQ's 0.01)
    num_blocks: int = 100  # Number of sub-blocks for Cholesky updates
    act_order: bool = False  # Activation ordering (same as GPTQ baseline)
    
    # Calibration (same as GPTQ baseline)
    num_calibration_samples: int = 256
    calibration_batch_size: int = 4
    
    # Timestep sampling for diffusion-specific calibration
    num_timesteps_per_sample: int = 5  # Sample multiple timesteps per prompt
    timestep_strategy: str = "uniform"  # Options: "uniform", "linear", "quadratic"
    
    # Which components to quantize (same as GPTQ baseline)
    quantize_transformer: bool = True
    quantize_text_encoder: bool = False  # Keep FP16
    quantize_vae: bool = False  # Keep FP16
    
    # Layers to skip (sensitive layers)
    skip_layers: List[str] = field(default_factory=lambda: [
        "time_embed",
        "label_embed", 
        "proj_out",
        "pos_embed",
    ])


# =============================================================================
# EVALUATION CONFIGURATION (Same as GPTQ baseline)
# =============================================================================

@dataclass
class EvaluationConfig:
    """Configuration for benchmark evaluation."""
    
    # FID calculation
    num_fid_images: int = 5000
    fid_batch_size: int = 4
    reference_dataset: str = "coco2017"
    
    # CLIP score
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"
    
    # Visual inspection
    num_visual_samples: int = 50
    visual_prompts_file: str = str(PROMPTS_DIR / "visual_inspection.txt")
    
    # Generation settings for evaluation (same as GPTQ baseline)
    eval_num_inference_steps: int = 28
    eval_guidance_scale: float = 4.5
    eval_seed: int = 42
    
    # Output settings
    save_individual_images: bool = True
    save_comparison_grid: bool = True
    
    # VRAM tracking
    track_vram: bool = True
    vram_sample_interval: float = 0.1


# =============================================================================
# DEFAULT INSTANCES
# =============================================================================

MODEL_CONFIG = ModelConfig()
QRONOS_CONFIG = QronosConfig()
EVAL_CONFIG = EvaluationConfig()


# =============================================================================
# 50 VISUAL INSPECTION PROMPTS (Same as GPTQ baseline)
# =============================================================================

DEFAULT_VISUAL_PROMPTS = [
    # Photorealistic - People (3)
    "A professional portrait photograph of a woman with curly auburn hair, soft studio lighting, sharp focus",
    "An elderly man with weathered hands playing chess in a sunlit park, documentary photography style",
    "A young child playing in autumn leaves, golden hour photography, candid moment",
    
    # Photorealistic - Nature (5)
    "A majestic snow-capped mountain reflected in a crystal clear alpine lake at golden hour",
    "Close-up macro photography of morning dew drops on a spider web, bokeh background",
    "A dense forest with sunbeams filtering through the canopy, mystical atmosphere",
    "A dramatic thunderstorm over vast wheat fields, moody atmosphere, landscape photography",
    "Cherry blossoms falling along a Japanese river path in spring, serene landscape",
    
    # Photorealistic - Animals (4)
    "A red fox in a snowy forest, wildlife photography, National Geographic style",
    "An owl perched on a branch at twilight, detailed feathers, piercing yellow eyes",
    "A golden retriever running on a sandy beach at sunset, action photography",
    "A colorful parrot perched on a tropical branch, macro photography, vibrant colors",
    
    # Photorealistic - Urban (4)
    "A bustling Tokyo street at night with neon signs and light trails, cyberpunk atmosphere",
    "A cozy coffee shop interior with warm lighting, wooden furniture, and steaming cups",
    "Modern glass skyscraper reflecting sunset clouds, architectural photography",
    "A busy street market in Marrakech, vibrant colors, documentary style",
    
    # Artistic - Oil Paintings (4)
    "An oil painting of a stormy seascape in the style of J.M.W. Turner, dramatic waves",
    "A Renaissance-style portrait of a noble woman, sfumato technique, dark background",
    "Impressionist painting of a garden party, dappled sunlight, Monet-inspired",
    "Van Gogh style starry night over a modern city skyline, swirling brushstrokes",
    
    # Artistic - Watercolor (3)
    "A delicate watercolor painting of spring flowers, soft edges, botanical illustration",
    "Watercolor landscape of rolling Tuscan hills at sunset, loose brushwork",
    "Japanese ink wash painting of misty mountains, minimalist, zen aesthetic",
    
    # Artistic - Digital Art (4)
    "A cyberpunk cityscape with flying cars and holographic billboards, neon glow",
    "Fantasy concept art of a floating castle in the clouds, epic scale",
    "Steampunk mechanical dragon, intricate gears and brass details, digital painting",
    "Surrealist digital art of melting clocks in a desert, Dali-inspired",
    
    # Artistic - Illustration (3)
    "A children's book illustration of a friendly dragon and a princess, whimsical style",
    "Art nouveau poster of a beautiful woman with flowing hair and flowers",
    "Comic book style superhero landing pose, dynamic action lines, bold colors",
    
    # Food Photography (4)
    "A gourmet burger with melting cheese and fresh vegetables, professional food photography",
    "An elegant sushi platter on a black slate board, Japanese cuisine photography",
    "Steaming bowl of ramen with perfectly cooked egg, chopsticks lifting noodles",
    "A colorful smoothie bowl with fresh berries and granola, overhead shot",
    
    # Architecture (4)
    "Gothic cathedral interior with sunbeams through stained glass windows",
    "Modern minimalist house with floor-to-ceiling windows overlooking the ocean",
    "Ancient ruins of a Greek temple at sunset, golden light on marble columns",
    "Futuristic eco-friendly building covered in vertical gardens, sustainable architecture",
    
    # Cinematic (3)
    "A noir detective in a smoky office, venetian blind shadows, 1940s aesthetic",
    "An epic battle scene with armies clashing, cinematic wide shot, Lord of the Rings style",
    "A astronaut standing alone on Mars, vast red desert, cinematic composition",
    
    # Cute/Whimsical (3)
    "A tiny mouse reading a tiny book under a mushroom, fairytale illustration style",
    "A fluffy corgi puppy wearing a tiny crown, sitting on a velvet cushion",
    "A magical treehouse in an enchanted forest, fairy lights, cozy atmosphere",
    
    # Abstract/Conceptual (3)
    "Abstract art with vibrant colors and geometric shapes, Kandinsky inspired",
    "A surreal scene of a door opening to the cosmos, floating islands",
    "Minimalist composition of a single red rose on white background, high contrast",
    
    # Seasonal (3)
    "A pumpkin patch with scarecrows under autumn sky, warm fall colors",
    "A Christmas market with snow and twinkling lights, festive atmosphere",
    "A summer beach scene with colorful umbrellas and crystal clear water",
]


# =============================================================================
# SD 3.5 MEDIUM MMDiT LAYER STRUCTURE
# =============================================================================

# These are the linear layers that will be quantized in each transformer block
TRANSFORMER_LINEAR_LAYERS = [
    "attn.to_q",
    "attn.to_k", 
    "attn.to_v",
    "attn.to_out.0",
    "ff.net.0.proj",  # GEGLU first linear
    "ff.net.2",       # FF second linear
]

# Number of transformer blocks in SD 3.5 Medium
NUM_TRANSFORMER_BLOCKS = 24
