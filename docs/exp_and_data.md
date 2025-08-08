## 2) Experiments & Data to Collect

### 2.1 Main Comparisons

* **Baselines**: ResNet50+DCL, TransFG, CLE-ViT, (optionally) MaskCOV/Part-based methods.
* **Our Variants**:

  1. Full WallyCL (Odd + SupCon + TokCon + CE).
  2. w/o TokCon (no token mining) – tests micro-region effect.
  3. w/o Odd (SupCon+CE only) – tests group odd-one-out contribution.
  4. w/o SupCon (Odd+TokCon+CE) – tests multi-positive contrastive role.
  5. Replace Gumbel-Top-k with Hard Top-k (STE) or Sinkhorn – tests selection relaxation.

### 2.2 Ablations & Sensitivity

* Group size $m$ \in {2,3,4,5}.
* Token subset size $k$ \in {4,6,8,12,16}.
* Temperatures $\tau$ \in {0.05, 0.1, 0.2}.
* Distance types $d$: cosine vs. squared Euclidean.
* With/without structure-aware masking/shuffling.
* Backbone: ViT-B vs. Swin-B.

### 2.3 Robustness & Generalization

* **Few-shot regime**: 1/2/4 shots per class finetuning.
* **Cross-subset transfer**: train on SoyLocal, test on SoyGlobal; Cotton80 ↔ Soy subsets.
* **Noise tolerance**: randomly flip a small % of labels; report degradation curves.
* **Domain shift**: lighting/contrast perturbations; report accuracy and ECE.

### 2.4 Efficiency & Scalability

* Throughput (imgs/s), GPU memory, training time/epoch.
* Convergence plots; variance of gradients (estimate via running variance over steps).

### 2.5 Metrics to Collect

* **Core**: Top-1, Top-5, macro/micro-F1 per subset.
* **Odd-ID**: odd-one-out accuracy, AUROC of oddness score.
* **Retrieval**: mAP@{1,5,10}.
* **Attribution**: Pointing-Game %, Mask-Drop (Δ prob when masking selected tokens), sparsity |S|.
* **Calibration**: ECE, NLL.
* **Efficiency**: TFLOPs/step (approx), GPU mem (GB), wall-clock.

### 2.6 Tables & Plots to Produce

* Main results table across UFGVC subsets.
* Ablation heatmaps for (m, k), line plots for $\tau$ and distance type.
* Convergence curves (train/val losses, Top-1, Odd-ID acc).
* Attribution qualitative grid with token masks; faithfulness scores.
* Efficiency bar charts.

---