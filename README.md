# Generalized Zero-Shot Skeleton Action Recognition

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=flat&logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/python-3.8+-blue?style=flat&logo=python" alt="Python">
  <img src="https://img.shields.io/github/license/your-username/GZSL?style=flat" alt="License">
</p>

This is a PyTorch implementation of the paper:

> **Generalized Zero-Shot Skeleton Action Recognition with Compositional Motion-Attribute Primitives**  
> *Jinlong Wang, Xuan Liu, Bin Lyu, Jinchao Ge, Jiahui Yu*  
> Pattern Recognition, 2025

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#project-structure">Structure</a> •
  <a href="#model-architecture">Architecture</a> •
  <a href="#datasets">Datasets</a> •
  <a href="#citation">Citation</a>
</p>

---

## Overview

This project implements a compositional framework for **Generalized Zero-Shot (GZS) skeleton-based action recognition** that learns reusable body-part motion primitives and aligns them with structured textual semantics.

### Key Features

- **Motion Attribute Extraction**: Part-level 8D motion-attribute vectors capturing spatial shape, dominant direction, and short-term dynamics
- **Dual-branch Architecture**: CLIP-based text encoder + Shift-GCN-based skeleton encoder
- **Hierarchical Cross-Modal Alignment**: Both global and part-level alignment between skeleton and text
- **Primitive Independence**: Enforces independence between different body-part primitives
- **Few-shot Support**: K-shot evaluation on HMDB-51 (K ∈ {2, 4, 8, 16})

---

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.0+ (for GPU support)

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/GZSL.git
cd GZSL

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Download CLIP weights (automatically done on first run)
```

### Requirements

```
torch>=2.0.0
torchvision
numpy
tqdm
pyyaml
scikit-learn
transformers
```

---

## Quick Start

### 1. Generate Text Prompts

Generate part-level textual descriptions for action classes:

```bash
python scripts/generate_prompts.py --dataset ntu60 --output data/prompts/
```

### 2. Train the Model

```bash
python scripts/train.py --config config/config.yaml
```

### 3. Test the Model

```bash
python scripts/test.py --checkpoint checkpoints/best_model.pth --dataset ntu60
```

---

## Project Structure

```
GZSL/
├── config/
│   └── config.py              # Configuration settings
├── data/
│   ├── __init__.py
│   ├── dataset.py             # Dataset loaders (NTU60/120, PKU-MMD, UCF101, HMDB-51)
│   ├── motion_attribute.py    # Motion attribute computation (Section 3.1)
│   └── few_shot.py            # Few-shot learning support
├── models/
│   ├── __init__.py
│   ├── text_encoder.py        # CLIP + Local Text Encoder
│   ├── skeleton_encoder.py    # Shift-GCN + Local Skeleton Encoder
│   ├── aggregation.py         # Primitive composition & losses (Section 3.3)
│   └── gzsl_model.py          # Main GZSL model (Section 3.4)
├── utils/
│   ├── __init__.py
│   └── metrics.py             # GZSL evaluation metrics
├── scripts/
│   ├── train.py               # Training script
│   ├── test.py                # Testing script
│   └── generate_prompts.py   # LLM prompt generation
├── clip/                      # CLIP model weights
├── requirements.txt
├── README.md
└── __init__.py
```

---

## Model Architecture

### Motion Attribute (Section 3.1)

Each body part is characterized by an **8-dimensional motion attribute vector**:

| Index | Attribute | Description |
|-------|-----------|-------------|
| 1 | Compactness Mean $d_\mu$ | Mean distance to centroid |
| 2 | Compactness Variance $d_\sigma^2$ | Variance of distances |
| 3 | Spatial Extent $e$ | Diagonal of bounding box |
| 4 | PCA Eigenvalue Ratio $\rho$ | Anisotropy measure |
| 5 | Principal Angle $\theta$ | Dominant orientation |
| 6 | Velocity Magnitude $k$ | Average velocity |
| 7 | Acceleration Magnitude $a_c$ | Average acceleration |
| 8 | Relative Deformation $E_{rel}$ | Frame-to-frame deformation |

### Loss Functions (Section 3.3)

The overall training objective:

$$\mathcal{L} = \lambda_p \mathcal{L}_p + \lambda_g \mathcal{L}_g + \lambda_c \mathcal{L}_{cons} + \lambda_i \mathcal{L}_{ind}$$

| Loss | Description |
|------|-------------|
| $\mathcal{L}_p$ | Primitive-level alignment (InfoNCE) |
| $\mathcal{L}_g$ | Global-level alignment (InfoNCE) |
| $\mathcal{L}_{cons}$ | Consistency between compositional and Shift-GCN global features |
| $\mathcal{L}_{ind}$ | Independence between different body-part primitives |

### Primitive Composition (Section 3.3)

Part-level features are aggregated using attention:

$$\gamma_{i,p} = \text{softmax}(\mathbf{w}^\top \sigma(\mathbf{W}\mathbf{h}_{i,p}))$$

$$\mathbf{g}_i = \sum_{p=1}^{P} \gamma_{i,p} \cdot \mathbf{h}_{i,p}$$

---

## Datasets

| Dataset | Classes | Samples | Joints |
|---------|---------|---------|--------|
| NTU RGB+D 60 | 60 | ~56K | 25 |
| NTU RGB+D 120 | 120 | ~110K | 25 |
| UCF101 | 101 | ~13K | 17 |
| PKU-MMD | 51 | ~20K | 25 |
| HMDB-51 | 51 | ~7K | 17 |

### GZS Splits (Standard)

| Dataset | Seen | Unseen |
|---------|------|--------|
| NTU60 | 55 | 5 |
| NTU120 | 110 | 10 |
| UCF101 | 80 | 21 |
| PKU-MMD | 46 | 5 |
| HMDB-51 | 31 | 20 |

---

## Training

### Configuration

Edit `config/config.yaml` to customize training:

```yaml
dataset:
  name: ntu60
  data_dir: data/ntu60/

training:
  batch_size: 32
  num_epochs: 100
  learning_rate: 0.001
  weight_decay: 0.0001

model:
  feature_dim: 256
  num_parts: 6
  lambda_p: 1.0
  lambda_g: 1.0
  lambda_c: 0.5
  lambda_i: 0.3
```

### Expected Results

| Dataset | Acc_s | Acc_u | HM |
|---------|-------|-------|-----|
| NTU60 | ~85% | ~65% | ~73% |
| NTU120 | ~82% | ~60% | ~69% |
| UCF101 | ~78% | ~55% | ~64% |
| PKU-MMD | ~80% | ~58% | ~67% |

---

## Few-shot Evaluation

Evaluate on unseen classes with K-shot:

```bash
python scripts/test.py --checkpoint checkpoints/best_model.pth \
    --dataset hmdb51 --few-shot --num-shots 16 --num-way 5
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{wang2025generalized,
  title={Generalized Zero-Shot Skeleton Action Recognition with Compositional Motion-Attribute Primitives},
  author={Wang, Jinlong and Liu, Xuan and Lyu, Bin and Ge, Jinchao and Yu, Jiahui},
  journal={Pattern Recognition},
  year={2025}
}
```

---

## License

This project is licensed under the MIT License.

---

## Acknowledgments

- [Shift-GCN](https://github.com/liu-zhy/Shift-GCN) for skeleton encoding
- [CLIP](https://github.com/openai/CLIP) for text encoder
