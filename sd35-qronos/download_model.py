import os

# 1. Enable high-performance transfer (Must be before importing huggingface_hub)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from huggingface_hub import snapshot_download

print("Starting download with hf_transfer enabled...")

# 2. Download the model
snapshot_download(
    repo_id="stabilityai/stable-diffusion-3.5-medium",
    resume_download=True,
    # Note: force_download=True will restart the download from scratch. 
    # If you want to resume where you left off, change this to False.
    force_download=True
)