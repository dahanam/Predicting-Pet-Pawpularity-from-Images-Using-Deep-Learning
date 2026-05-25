# Predicting Pet Pawpularity from Images Using Deep Learning

An image regression project that predicts a continuous **Pawpularity score 
(0–100)** directly from pet photos using pretrained deep learning models, 
Grad-CAM interpretability, and performance improvement techniques.

**Authors:** Dahana Moz Ruiz & Maria Santos Perez — Kean University  
**Dataset:** [PetFinder.my Pawpularity Contest (Kaggle)](https://www.kaggle.com/competitions/petfinder-pawpularity-score)

---

## Project Objective

Given a photo of a pet, predict how engaging or attractive the photo is 
based on normalized page view statistics from PetFinder.my. This is a 
pure **image regression** task — no metadata used, images only.

**Dataset:** 9,912 labeled pet images with 12 binary metadata features 
(Eyes, Face, Near, Action, etc.) and a continuous Pawpularity score.  
**Split:** 7,929 train / 1,983 validation

---

## Models Evaluated

| Model | Architecture | Val MSE | Val MAE | Val R² |
|-------|-------------|---------|---------|--------|
| **ResNet18** (Baseline) | CNN, pretrained ImageNet | 395.19 | ~14.0 | 0.106 |
| **EfficientNet-B3** | CNN, pretrained ImageNet | ~429.58 | ~14.79 | 0.028 |
| **ViT-B/16** | Vision Transformer, pretrained ImageNet | ~460.17 | ~15.13 | -0.041 |

All models use a single-output regression head replacing the final 
classification layer. ResNet18 achieved the best validation performance.

---

## Performance Improvement

Improved training strategy applied to ResNet18 and EfficientNet-B3:

- **Huber Loss (SmoothL1Loss, β=10)** — more robust to outliers than MSE
- **WeightedRandomSampler** — addresses class imbalance in Pawpularity scores

---

## Model Interpretability

**Grad-CAM** visualizations implemented for ResNet18 and EfficientNet-B3 
to highlight which image regions influence the predicted Pawpularity score, 
showing the model's attention on pet faces, eyes, and focal subjects.

---

## Tech Stack

- **Python** — Core language
- **PyTorch** — Model training, custom Dataset/DataLoader
- **TorchVision** — Pretrained models (ResNet18, EfficientNet-B3, ViT-B/16)
- **Scikit-learn** — R² scoring, train/val split
- **NumPy / Pandas / Matplotlib** — Data processing and visualization
- **Google Colab** — T4 GPU training environment

---

## Data Augmentation

Training transforms applied to improve generalization:
- Random horizontal flip
- Random rotation (±15°)
- Color jitter (brightness, contrast, saturation, hue)
- ImageNet normalization

---

## Setup:

### Requirements
```bash
pip install torch torchvision scikit-learn pandas numpy matplotlib pillow
```

### Dataset
Download from [Kaggle](https://www.kaggle.com/competitions/petfinder-pawpularity-score) 
and place in your Google Drive. Update `BASE_DIR` in the notebook to your path.

### Run
Open in Google Colab with a T4 GPU runtime and run all cells in order.

---

## Authors

Dahana Moz Ruiz & Maria Santos Perez — Kean University, Fall 2025
