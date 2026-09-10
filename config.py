import torch

# Dataset & Model Configurations
DATASET_NAME = "Hajorda/flameye-wildfire-detection"
MODEL_NAME = "google/vit-base-patch16-224"

# ViT Architecture Specifications
NUM_LAYERS = 12
NUM_HEADS_PER_LAYER = 12
TOTAL_HEADS = 144
IMAGE_SIZE = 224
PATCH_SIZE = 16
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2  # 196 patches + 1 CLS token = 197 tokens

# Hardware Setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[CONFIG LOADED] Target Device: {DEVICE} | Total ViT Attention Heads: {TOTAL_HEADS}")