# vit-mechanistic-interpretability
Causal ablation scan and attention-logit analysis of google/vit-base-patch16-224 to isolate feature-specific neural circuits (L11H0) for smoke plume detection.
# Vision Transformer Circuit Dissection (ViT MechInterp)

This project explores the internal mechanics of `google/vit-base-patch16-224` to understand how specific visual features are processed inside the model before a prediction is made.

Instead of treating the Vision Transformer as a black box, I performed a causal ablation scan across all 144 attention heads to find which specific circuits track high-contrast features like smoke plumes.

---

## Key Findings

* **Circuit Isolated:** Layer 11, Head 0 (L11H0) acts as a specialized feature detector for volumetric smoke and plume patterns.
* **Causal Dependence:** When L11H0 is zero-ablated (blocked), the model's confidence for smoke-related target classes drops significantly, proving its direct influence on the output.
* **Control Check:** The head shows minimal activation on non-smoke background images (like open skies and clouds), confirming it does not trigger on random background noise.

---

## Methodology & Logic

1. **Input Sequence Mapping:**
   * Input image size: `224x224x3`
   * Patch size: `16x16` -> splits the image into `196` patches.
   * Total sequence length: `197` tokens (196 patches + 1 `[CLS]` classification token).
   * Each token vector dimension: `768`.

2. **Attention Extraction:**
   * Intercepted the $197 \times 197$ raw attention weight matrix ($\text{Softmax}(QK^T / \sqrt{d_k})$) using Forward Hooks in PyTorch.
   * Extracted the 0th-row attention weights corresponding to the `[CLS]` token to map spatial focus.

3. **Causal Ablation:**
   * Systematically set output representations of target heads to zero during the forward pass.
   * Tracked changes in output class logits and softmax probabilities to measure head importance.

---

## Repository Structure

├── config.py              # Model parameters and configuration
├── src/
│   ├── dataset_loader.py  # Image processing and patch tokenization
│   ├── ablation_engine.py # PyTorch forward hooks and zero-ablation logic
│   └── visualizer.py      # Attention heatmap overlays
├── notebooks/
│   └── circuit_scan.ipynb # End-to-end execution and results
├── requirements.txt       # Dependencies
└── README.md
