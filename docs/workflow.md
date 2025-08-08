# WallyCL – Package for WACV Submission

*A group-based odd-one-out supervised contrastive learning method for Ultra-FGVC.*

---

## 1) Training & Validation Workflow

### 1.1 Overview

WallyCL trains a token-based backbone (e.g., ViT-B or Swin-B) with a **group-level odd-one-out objective** plus **multi-positive supervised contrastive** and **token-subset contrastive** losses. Each mini-batch is partitioned into groups of size $K=m+1$: $m$ positives from the same class and 1 **odd** sample from a different class.

### 1.2 Data Preparation

* **Datasets**: Use the UFGVC benchmark subsets (Cotton80, SoyLocal, SoyGene, SoyAgeing, SoyGlobal). Standard train/val splits; keep per-class stratification.
* **Preprocessing**: Resize to 256–320 on the short side, center-crop or random-resized-crop to 224 or 256; normalize with ImageNet statistics.
* **Augmentations**:

  * *Weak*: random crop, horizontal flip, color jitter, random gray.
  * *Strong (optional)*: mix of Cutout/RandomErasing, slight Gaussian blur.
  * *Structure-aware*: optional small **masking/shuffling** of patch tokens (à la CLE-ViT) applied consistently across positives to create hard positives.

### 1.3 Model

* **Backbone**: ViT-B/16 or Swin-B with token embeddings $\{z_t\}_{t=1}^T$. Use a projection head $g(\cdot)$ to map to contrastive space $\mathbb{R}^{d}$ (2-layer MLP with BN, ReLU, output L2-normalized).
* **Token-Subset Miner**: Differentiable Top-$k$ (Gumbel-Top-k or Sinkhorn-based relaxation) that selects a small subset $S \subset [T], |S|=k$ per image.
* **Classification Head**: Linear or 2-layer MLP on the \[CLS] token or pooled tokens.

### 1.4 Group Construction (per iteration)

1. Sample $G_j = \{x_1^{(j)},\dots,x_m^{(j)}, x_*^{(j)}\}$ with $y(x_i^{(j)}) = y_j$ and $y(x_*^{(j)}) \neq y_j$.
2. Apply augmentations to all images; optionally create a second view for contrastive terms.
3. Forward pass to obtain token embeddings, pooled embeddings, and logits.

### 1.5 Losses

Let $f(\cdot)$ be the normalized image embedding, $h(\cdot)$ the normalized group-pooled positive embedding, and $\bar z_S$ the normalized mean of selected tokens $S$.

* **Odd-One-Out Identification Loss (Group-Odd)**

  * For each group, compute an **intra-group inconsistency score**: $s(x) = \frac{1}{m}\sum_{i=1}^m d\big(f(x), f(x_i)\big)$ with $d(u,v)=\|u-v\|_2^2$ or $1-\langle u,v\rangle$.
  * Softmax over members; cross-entropy with the odd-index as label.

* **Supervised Multi-Positive Contrastive (Group-SupCon)**

  * Positives: all $m$ in-group samples from class $y_j$; negatives: the odd sample plus all other groups in the batch.
  * Temperature $\tau \in [0.05, 0.2]$, NT-Xent-style.

* **Token-Subset Contrastive (TokCon)**

  * Select $S$ on the odd sample and the positive aggregate (or per-positive), compute InfoNCE between $\bar z^{*}_S$ and $\bar z^+_{S}$ against negatives $\bar z^{(n)}_S$.

* **Classification CE** on the global pooled or \[CLS] token.

**Total**: $\mathcal{L} = \mathcal{L}_{\text{odd}} + \lambda_{\text{sup}}\mathcal{L}_{\text{sup}} + \lambda_{\text{tok}}\mathcal{L}_{\text{tok}} + \lambda_{\text{cls}}\mathcal{L}_{\text{CE}}$.

### 1.6 Optimization & Schedule

* **Optimizer**: AdamW, lr=$2\times 10^{-4}$ for ViT-B, weight decay=0.05–0.1; linear warmup 5 epochs; cosine decay to 20–30% of peak.
* **Batching**: 4–8 groups per batch; $m=3$–5; token subset size $k=6$–12.
* **Training Length**: 100–200 epochs depending on backbone and subset size.
* **EMA**: optionally maintain an EMA of backbone weights for smoother validation.

### 1.7 Validation Protocols

* **Standard Classification**: Top-1/Top-5 accuracy, macro/micro-F1.
* **Group Odd-One-Out**: Form pseudo-groups in validation (sample $m$ same-class + 1 different-class) and measure **Odd-ID accuracy**.
* **Region Attribution**: visualize selected tokens via token masks/heatmaps; compute **Pointing Game** or **Mask Drop** score to quantify faithfulness.
* **Calibration**: ECE on classification head.
* **Retrieval** (optional): mAP using embeddings; query by an image and retrieve same-class images.

### 1.8 Reproducibility & Logging

* Fix seeds; deterministic dataloader where feasible; log with TensorBoard/W\&B: losses, learning rate, gradient norms, selection sparsity (|S|/T), cosine similarities.
* Save checkpoints every N epochs and best-by-val metric.

---
