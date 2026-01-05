# DNN: Visual Storytelling with Contrastive Multimodal Alignment

## Quick Links
- **[Experiments Notebook]** — Full experimental workflow  
- **[Figures]** — Generated plots and visualizations  
- **[Tables]** — Evaluation metrics and logs  

---

## Innovation Summary
This project introduces **contrastive multimodal alignment** prior to temporal sequence modeling to improve narrative coherence and cross-modal grounding in visual storytelling tasks.

By aligning image and text representations in a shared latent space before fusion, the model learns stronger semantic correspondence across modalities, resulting in more consistent story reasoning even under limited computational resources.

---

## Project Overview
- **Project Name:** StoryReasoning-Multimodal  
- **Author:** Thota Kalpana  
- **Version:** 1.0  

This project addresses the challenge of multimodal sequence reasoning, where models must jointly process images and text while capturing temporal relationships.  
Given a sequence of **K = 4 image–text pairs**, the system predicts the **(K+1) image and textual continuation**, integrating computer vision, natural language processing, and temporal reasoning.

---

## Key Results

| Metric | Score  |
|--------|-------|
| BLEU   | 0.3328 |
| ROUGE  | 0.8889 |
| METEOR | 1.0000 |

**Evaluation:** BLEU: 0.3328, ROUGE: 0.8889, METEOR: 1.0000  

These results demonstrate strong semantic overlap and narrative consistency despite frozen encoders, limited training data, and CPU-only execution.

---

## Most Important Finding
The **contrastive alignment mechanism** significantly improves cross-modal consistency, enabling coherent narrative reasoning with minimal training and computational constraints.  

Supporting visualizations are available in: `results/figures/`

---

## Dataset
- **Source:** Hugging Face  
- **Dataset:** daniel3303/StoryReasoning  
- **Split:** Train  
- **Samples Used:** 300  
- **Streaming:** Enabled (no disk write)  
- **Sequence Length (K):** 4  
- **Max Text Length:** 32  

Independent padding for images and text was applied to handle variable-length sequences robustly.

---

## Model Architecture

### Visual Encoder
- **Backbone:** ResNet-18 (pretrained, frozen)  
- **Projection:** Fully connected layer → 256-D embedding  
Efficient semantic feature extraction via transfer learning.

### Text Encoder
- **Backbone:** DistilBERT (distilbert-base-uncased)  
- **Representation:** CLS token → 256-D embedding  
Backbone frozen for memory and speed efficiency.

### Multimodal Alignment (Key Innovation)
Prior to fusion, visual and textual embeddings are explicitly aligned using **CLIP-style contrastive learning**:  
- L2-normalized embeddings  
- Symmetric image-to-text and text-to-image loss  
- Alignment loss weighted by λ = 0.5  

This enforces a shared semantic space and reduces modality dominance.

### Multimodal Fusion
Aligned visual and textual embeddings are concatenated and projected through a fully connected fusion layer, producing a unified multimodal representation suitable for sequence modeling.

### Temporal Modeling
- **Model:** LSTM  
- **Input:** Fused multimodal embeddings  
- **Output:** Context vector summarizing narrative progression  

The final hidden state captures story evolution and conditions both image and text generation.

### Dual Decoders
**Image Decoder**  
- Deconvolutional (ConvTranspose) network  
- Progressive upsampling to 224 × 224 RGB  
- Sigmoid activation for pixel normalization  

**Text Decoder**  
- GRU-based decoder with teacher forcing  
- Context vector initializes hidden state  
- Cross-entropy loss with padding ignored

---

## Training Strategy
Training was conducted in a **CPU-only environment**, requiring careful optimization:  
- Pretrained encoders frozen  
- Batch size reduced to 2  
- Explicit memory cleanup during training  
- Single-epoch training for stability  

**Total Loss:**  

Image reconstruction loss (MSE) +
Text generation loss (Cross-Entropy) +
Contrastive alignment loss (λ = 0.5)

### Outputs
The training pipeline automatically saves:  
- **Figures:** `results/figures/`  
- **Tables:** `results/tables/`  

**TRAINING COMPLETE**  
Saved → `results/figures/`  
Saved → `results/tables/`

## How to Reproduce
1. **Install dependencies**  
```bash
pip install -r requirements.txt

## Run dataset preprocessing and visualization
python src/utils.py
## Run model inspection
python src/model.py
## Train and evaluate
python src/train.py

# Conclusion
This project demonstrates that explicit contrastive multimodal alignment combined with recurrent temporal modeling enables effective visual story reasoning under constrained resources.
Despite limited data and frozen backbones, the model produces coherent narratives with strong semantic grounding.
Future work may include transformer-based decoders, diffusion image generation, and human-in-the-loop evaluation for story coherence.


