import torch
from datasets import load_dataset
from transformers import ViTImageProcessor
from config import DATASET_NAME, MODEL_NAME

class WildfireDatasetLoader:
    def __init__(self, split="test"):
        """
        Loads HuggingFace wildfire dataset and initializes ViT processor
        """
        print(f"[DATASET] Loading '{split}' split from {DATASET_NAME}...")
        self.raw_dataset = load_dataset(DATASET_NAME, split=split)
        self.processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
        print(f"[DATASET] Loaded {len(self.raw_dataset)} samples successfully.")

    def get_sample(self, index=0):
        """
        Returns raw RGB image and processed PyTorch tensor for a given index
        """
        sample = self.raw_dataset[index]
        raw_image = sample['image'].convert("RGB")
        
        # Preprocess image for ViT (224x224 resize and normalization)
        inputs = self.processor(images=raw_image, return_tensors="pt")
        
        # Class label (if present in dataset)
        label = sample.get('label', None)
        
        return {
            "raw_image": raw_image,
            "pixel_values": inputs['pixel_values'],
            "label": label,
            "index": index
        }

    def filter_forest_vs_nonforest(self):
        """
        Categorizes samples for causal intervention testing
        """
        forest_samples = []
        non_forest_samples = []
        
        for idx, item in enumerate(self.raw_dataset):
            # Checking metadata tags for terrain distinction
            source_info = str(item.get('source', '')).lower()
            if 'forest' in source_info or 'aiformankind' in source_info:
                forest_samples.append(idx)
            else:
                non_forest_samples.append(idx)
                
        return forest_samples, non_forest_samples