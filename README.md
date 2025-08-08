# WallyCL: Group-based Contrastive Learning for Ultra-Fine-Grained Visual Classification

WallyCL is a novel approach for Ultra-Fine-Grained Visual Classification (UFGVC) that uses group-based odd-one-out contrastive learning and token subset mining to identify discriminative micro-features.

## Overview

This implementation brings the "Where's Wally?" concept to ultra-fine-grained classification:
- **Group-based Learning**: Train with groups of m positives + 1 odd sample
- **Token Subset Mining**: Automatically discover discriminative micro-regions
- **Multi-objective Loss**: Combines odd-one-out, supervised contrastive, and token contrastive losses

## Key Features

- ✅ **Differentiable Token Selection**: Gumbel-Top-K for selecting discriminative tokens
- ✅ **Group-based Sampling**: Custom sampler for creating training groups
- ✅ **Multi-loss Training**: Odd-one-out + SupCon + TokCon + Classification losses
- ✅ **UFGVC Dataset Support**: Built-in support for Cotton80, Soybean, and other datasets
- ✅ **Visualization Tools**: Token attention maps and odd-one-out predictions
- ✅ **Comprehensive Evaluation**: Classification, odd-detection, and attribution metrics

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd WallyCL

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Training

```bash
# Train on Cotton80 dataset with default configuration
python train.py --dataset cotton80

# Train with custom config
python train.py --config configs/default.yaml --dataset soybean

# Resume training from checkpoint
python train.py --resume checkpoints/best_checkpoint.pth
```

### 2. Evaluation

```bash
# Evaluate trained model
python evaluate.py --checkpoint checkpoints/best_checkpoint.pth --dataset cotton80

# Evaluate specific split
python evaluate.py --checkpoint checkpoints/best_checkpoint.pth --split test
```

### 3. Inference

```bash
# Single image classification with attention visualization
python inference.py --checkpoint checkpoints/best_checkpoint.pth --image path/to/image.jpg

# Odd-one-out detection on a group of images
python inference.py --checkpoint checkpoints/best_checkpoint.pth --group img1.jpg img2.jpg img3.jpg img4.jpg img5.jpg
```

## Dataset Support

The implementation includes built-in support for UFGVC datasets:

- **Cotton80**: Cotton leaf classification (80 classes)
- **Soybean**: Soybean leaf classification
- **Soy Ageing R1-R6**: Soybean ageing datasets

Datasets are automatically downloaded when first used:

```python
from src.dataset.ufgvc import UFGVCDataset

# Load dataset
dataset = UFGVCDataset(
    dataset_name="cotton80",
    root="./data",
    split="train",
    download=True
)
```

## Architecture

### Model Components

1. **Backbone**: ViT-Base or Swin-Base for token extraction
2. **Token Scorer**: MLP for computing token importance scores
3. **Gumbel-Top-K**: Differentiable token selection mechanism
4. **Projection Heads**: For contrastive learning in embedding space

### Loss Functions

1. **Odd-One-Out Loss**: Cross-entropy for identifying the outlier in groups
2. **Supervised Contrastive Loss**: Multi-positive contrastive learning
3. **Token Contrastive Loss**: Contrastive learning on selected token subsets
4. **Classification Loss**: Standard cross-entropy for final predictions

## Configuration

Training behavior is controlled via YAML configuration files:

```yaml
# configs/default.yaml
model:
  backbone: "vit_base_patch16_224"
  k_tokens: 8  # Number of tokens to select
  embed_dim: 128

data:
  dataset_name: "cotton80"
  group_size: 5  # 4 positives + 1 odd
  groups_per_batch: 4

loss:
  lambda_sup: 1.0   # SupCon weight
  lambda_tok: 0.5   # Token contrastive weight
  lambda_ce: 1.0    # Classification weight
  tau_sup: 0.1      # SupCon temperature
  tau_tok: 0.07     # Token temperature

training:
  epochs: 100
  use_ema: true
```

## Results and Metrics

The evaluation script provides comprehensive metrics:

- **Classification**: Accuracy, F1-score, Top-5 accuracy
- **Odd-one-out Detection**: Accuracy on pseudo-groups
- **Token Attribution**: Sparsity and concentration metrics
- **Calibration**: Expected Calibration Error (ECE)

## Ablation Studies

Run ablation studies by modifying the configuration:

```bash
# Without token contrastive loss
python train.py --config configs/ablation_no_token.yaml

# Without supervised contrastive loss
python train.py --config configs/ablation_no_supcon.yaml

# Strong augmentation
python train.py --config configs/strong_augmentation.yaml
```

## File Structure

```
WallyCL/
├── src/
│   ├── dataset/
│   │   └── ufgvc.py              # UFGVC dataset implementation
│   ├── models/
│   │   ├── wallycl.py            # Main WallyCL model
│   │   ├── losses.py             # Loss functions
│   │   └── token_utils.py        # Token selection utilities
│   ├── data/
│   │   ├── group_sampler.py      # Group-based sampling
│   │   └── transforms.py         # Data augmentations
│   └── utils/
│       ├── metrics.py            # Evaluation metrics
│       ├── checkpoint.py         # Checkpoint management
│       └── ema.py               # Exponential moving average
├── configs/
│   └── default.yaml             # Default configuration
├── train.py                     # Training script
├── evaluate.py                  # Evaluation script
├── inference.py                 # Inference and visualization
└── requirements.txt             # Dependencies
```

## Key Implementation Details

### Group Construction
```python
# Each training group contains:
# - m positive samples (same class)
# - 1 odd sample (different class)
group = [pos1, pos2, pos3, pos4, odd]  # m=4 case
```

### Token Selection
```python
# Differentiable top-k selection using Gumbel-Top-K
scores = token_scorer(tokens)  # [batch, num_tokens]
weights = gumbel_topk(scores, k=8)  # [batch, num_tokens]
selected = compute_token_average(tokens, weights)  # [batch, dim]
```

### Loss Computation
```python
total_loss = (odd_loss + 
              lambda_sup * supcon_loss + 
              lambda_tok * token_loss + 
              lambda_ce * classification_loss)
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{wallycl2024,
  title={WallyCL: Group-based Contrastive Learning for Ultra-Fine-Grained Visual Classification},
  author={Your Name},
  journal={Conference/Journal},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built on top of [timm](https://github.com/rwightman/pytorch-image-models) for backbone models
- Inspired by supervised contrastive learning and CLE-ViT methodologies
- UFGVC datasets provided by the agricultural research community