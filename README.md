# Vision Transformer Circuit Dissection (ViT- Mechanistic-interpretability)
This repository contains Causal ablation scan and attention-logit analysis of 'google/vit-base-patch16-224' to isolate feature-specific neural circuits (L11H0) for smoke plume detection. The primary objective is to move beyond black-box classification by locating, isolating, and validating feature-specific neural circuits responsible for detecting dense, high-contrast visual features (volumetric smoke plumes).

---

## Abstract & Core Findings

Instead of treating transformer representations as monolithic embeddings, I applied a systematic causal ablation scan across all 144 attention heads (12 layers × 12 heads). 

Key experimental observations include:
* **Specific Circuit Localization:** Identified **Layer 11, Head 0 (L11H0)** as a highly specialized functional circuit for localized smoke plume detection.
* **Causal Impact:** Zero-ablating the output of L11H0 leads to a substantial drop in target class logits (e.g., ImageNet proxies for smoke/volcano), establishing direct causal dependence rather than passive correlation.
* **Spatial Alignment:** Extracting raw attention weight matrices reveals that L11H0 routes spatial token attention directly to the high-contrast boundaries of the plume.
* **Control Validation:** Running the model on non-fire control images (e.g., clear skies, open landscapes, and clouds) yields near-zero activation for L11H0, verifying resistance to false-positive background noise.

---

## Experimental Workflow & Mathematical Logic
```text
Input Image (224x224x3)
       │
       ▼
Patch Extraction (16x16)  ──►  196 Patch Tokens + 1 [CLS] Token
       │
       ▼
Transformer Encoder (12 Layers, 144 Total Attention Heads)
       │
       ├─► Forward Hook Interception  ──► Raw Attention Weights: Softmax(QK^T / sqrt(d_k))
       │
       └─► Causal Zero-Ablation      ──► Block L11H0 Vector Output
       │
       ▼
Logit & Probability Metrics  ──► Measure Confidence Drop & Target Class Sensitivity
```
### 1. Patch Tokenization & Dimension Mapping
* The input image $I \in \mathbb{R}^{224 \times 224 \times 3}$ is partitioned into $16 \times 16$ non-overlapping patches.
* Grid size: $(224 / 16) \times (224 / 16) = 14 \times 14 = 196$ patch tokens.
* Including the classification token (`[CLS]` at index 0), the total sequence length is $197$ tokens.
* Each patch is linearly projected into a $768$-dimensional vector embedding ($16 \times 16 \times 3 = 768$).

### 2. Attention Matrix Interception
Using PyTorch forward hooks (`register_forward_hook`), I intercepted the raw attention weight matrices:

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where $d_k = 768 / 12 = 64$. For each head, this yields a $197 \times 197$ weight matrix. Isolating row 0 provides the direct attention weights given by the `[CLS]` token to every visual patch.

### 3. Systematic Causal Ablation
To test whether L11H0 is causally required for classification:
1. Ran an unablated forward pass to establish baseline logit distribution.
2. Modified the forward pass of Layer 11 Head 0 by forcing its activation vector output to $0$.
3. Computed the prediction error and probability change across ImageNet target classes.

---

## Repository Structure

```text
vit-mechanistic-interpretability/
├── config.py                 # Configuration for model weights and patch dimensions
├── requirements.txt          # Python environment dependencies
├── README.md                 # Research documentation and findings
├── src/
│   ├── dataset_loader.py     # Image preprocessing and patch token sequence pipeline
│   ├── ablation_engine.py    # PyTorch forward hooks and causal ablation module
│   └── visualizer.py         # Heatmap generator for attention weight overlays
└── notebooks/
    └── circuit_scan.ipynb    # End-to-end execution notebook with attention plots
```
## Execution Steps

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Divyanshivaswan/vit-mechanistic-interpretability.git](https://github.com/Divyanshivaswan/vit-mechanistic-interpretability.git)
2. **Navigate into the directory:**
   ```bash
   cd vit-mechanistic-interpretability
4. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
6. **Run the Analysis Notebook:**
    Launch Jupyter and execute ```bash notebooks/circuit_scan.ipynb to run the 144-head ablation scan and generate    attention heatmap overlays.

