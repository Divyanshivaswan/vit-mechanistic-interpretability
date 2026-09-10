import torch
from transformers import ViTForImageClassification, ViTImageProcessor

class ViTEngine:
    def __init__(self, model_name="google/vit-base-patch16-224"):
        print(f"[MODEL] Loading {model_name}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTForImageClassification.from_pretrained(
            model_name, 
            output_attentions=True
        ).to(self.device)
        self.model.eval()
        print(f"[MODEL] Loaded successfully on {self.device}.")

    def forward_pass(self, pixel_values):
        pixel_values = pixel_values.to(self.device)
        with torch.no_grad():
            outputs = self.model(pixel_values)
        
        return {
            "logits": outputs.logits,
            "attentions": outputs.attentions
        }