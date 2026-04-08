# 🧠 Brain Tumor Detection & Segmentation using Deep Learning

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.19-orange?style=for-the-badge&logo=tensorflow"/>
  <img src="https://img.shields.io/badge/Platform-Kaggle-20BEFF?style=for-the-badge&logo=kaggle"/>
  <img src="https://img.shields.io/badge/Accuracy-85.25%25-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Dice_Score-0.871-brightgreen?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge"/>
</p>

<p align="center">
  <b>A Deep Learning system that detects and segments brain tumors from MRI images using VGG16 Transfer Learning and U-Net Architecture.</b>
</p>

---

> ⚠️ **Disclaimer:** This project is developed for **academic and research purposes only**. It is **not intended for clinical diagnosis or medical decision-making**. Always consult a qualified medical professional for health concerns.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Demo](#-demo)
- [Features](#-features)
- [Results](#-results)
- [Datasets](#-datasets)
- [Model Architecture](#-model-architecture)
- [Technical Details](#-technical-details)
- [Requirements](#-requirements)
- [How to Run](#-how-to-run)
- [Project Structure](#-project-structure)
- [Evaluation Metrics](#-evaluation-metrics)
- [Limitations](#-limitations)
- [Future Work](#-future-work)
- [References](#-references)
- [Author](#-author)
- [License](#-license)

---

## 🔍 Overview

Brain tumors are among the most dangerous and life-threatening diseases. Early and accurate detection is critical for successful treatment. This project builds an **end-to-end AI pipeline** that:

1. **Classifies** the type of brain tumor from MRI images using **VGG16 Transfer Learning**
2. **Segments** the exact tumor region at pixel level using **U-Net Deep Learning**

### Tumor Types Detected
| Type | Description |
|------|-------------|
| 🔴 **Glioma** | Most dangerous, irregular shape, aggressive |
| 🟡 **Meningioma** | Grows on brain membrane, usually benign |
| 🟢 **No Tumor** | Healthy brain MRI |
| 🔵 **Pituitary** | Grows on pituitary gland |

---

## 🎬 Demo

```
Input MRI Image → VGG16 → "Glioma (85% confidence)"
                → U-Net → Tumor Region Highlighted
```

### Pipeline Flow
```
MRI Image (128×128)
       │
       ├──→ VGG16 Classifier ──→ Tumor Type + Confidence %
       │
       └──→ U-Net Segmentation ──→ Pixel-level Tumor Mask
                                           │
                                    Overlay on MRI
```

---

## ✨ Features

- ✅ **Multi-class Classification** — 4 tumor types detected
- ✅ **Pixel-level Segmentation** — Exact tumor location marked
- ✅ **Transfer Learning** — VGG16 pretrained on ImageNet
- ✅ **Real Ground Truth Masks** — BraTS2020 dataset used
- ✅ **Two-Phase Training** — Head training + Fine-tuning
- ✅ **Anti-Overfitting** — Dropout, L2, BatchNorm, Augmentation
- ✅ **Auto Path Detection** — Works on any Kaggle setup
- ✅ **Combined Pipeline** — Both models work together
- ✅ **Detailed Evaluation** — ROC, Confusion Matrix, Dice, IoU

---

## 📊 Results

### 🏆 VGG16 Classification Results

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Glioma | 0.85 | 0.73 | 0.78 | 400 |
| Meningioma | 0.74 | 0.85 | 0.79 | 400 |
| No Tumor | 0.91 | 0.99 | 0.95 | 400 |
| Pituitary | 0.94 | 0.84 | 0.89 | 400 |
| **Overall** | **0.86** | **0.85** | **0.85** | **1600** |

**✅ Overall Accuracy: 85.25%**

### 🏆 U-Net Segmentation Results

| Metric | Train | Validation | Gap |
|--------|-------|------------|-----|
| Dice Coefficient | 0.8776 | 0.8710 | 0.0066 |
| IoU Score | 0.7640 | 0.7710 | 0.007 |
| Loss | 0.2116 | 0.1616 | — |

**✅ Dice Score: 0.871 | IoU: 0.77 | No Overfitting!**

---

## 🗂️ Datasets

### Dataset 1 — Brain Tumor MRI (Classification)
```
Name    : Brain Tumor MRI Dataset
Author  : masoudnickparvar
Source  : Kaggle
Link    : https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
Classes : Glioma, Meningioma, No Tumor, Pituitary
Total   : ~7,023 images
Split   : Training (5712) + Testing (1311)
Format  : JPG / PNG
```

### Dataset 2 — BraTS2020 (Segmentation)
```
Name    : Brain Tumor Segmentation 2020
Author  : awsaf49
Source  : Kaggle
Link    : https://www.kaggle.com/datasets/awsaf49/brats2020-training-data
Total   : 57,198 pre-sliced 2D .h5 files
Contains: 4 MRI modalities + Ground truth masks
Masks   : Necrosis, Edema, Enhancing Tumor
Format  : HDF5 (.h5)
```

---

## 🏗️ Model Architecture

### Model 1 — VGG16 Classifier

```
┌─────────────────────────────────────────┐
│           INPUT (128×128×3)             │
├─────────────────────────────────────────┤
│    VGG16 Base (Pretrained ImageNet)     │
│    ├── Block 1: Conv64 × 2              │
│    ├── Block 2: Conv128 × 2             │
│    ├── Block 3: Conv256 × 3             │
│    ├── Block 4: Conv512 × 3             │
│    └── Block 5: Conv512 × 3 (unfrozen) │
├─────────────────────────────────────────┤
│         GlobalAveragePooling2D          │
├─────────────────────────────────────────┤
│     BatchNormalization + Dropout(0.4)   │
├─────────────────────────────────────────┤
│       Dense(512, ReLU) + L2 Reg         │
├─────────────────────────────────────────┤
│     BatchNormalization + Dropout(0.3)   │
├─────────────────────────────────────────┤
│          Dense(256, ReLU)               │
├─────────────────────────────────────────┤
│             Dropout(0.2)                │
├─────────────────────────────────────────┤
│       Dense(4, Softmax) → OUTPUT        │
└─────────────────────────────────────────┘
```

### Model 2 — U-Net Segmentation

```
INPUT (128×128×3)
        │
   ┌────▼────────────────────────────────────┐
   │  ENCODER (Downsampling)                 │
   │  Level 1: Conv64  → 128×128 [skip s1]  │
   │  Level 2: Conv128 → 64×64  [skip s2]   │
   │  Level 3: Conv256 → 32×32  [skip s3]   │
   │  Level 4: Conv512 → 16×16  [skip s4]   │
   └────────────────────────────────────────┘
        │
   ┌────▼────────────────────────────────────┐
   │  BOTTLENECK: Conv1024 → 8×8             │
   └────────────────────────────────────────┘
        │
   ┌────▼────────────────────────────────────┐
   │  DECODER (Upsampling)                   │
   │  Level 1: ConvT + s4 → 16×16  Conv512  │
   │  Level 2: ConvT + s3 → 32×32  Conv256  │
   │  Level 3: ConvT + s2 → 64×64  Conv128  │
   │  Level 4: ConvT + s1 → 128×128 Conv64  │
   └────────────────────────────────────────┘
        │
   Conv(1, sigmoid) → OUTPUT MASK (128×128×1)
```

**Key Feature: Skip Connections**
- Encoder features directly connected to Decoder
- Preserves spatial location information
- Enables pixel-perfect segmentation

---

## ⚙️ Technical Details

### Anti-Overfitting Techniques
| Technique | Where Used | Purpose |
|-----------|-----------|---------|
| Dropout (0.2-0.4) | Both models | Randomly drops neurons |
| L2 Regularization | Dense + Conv layers | Penalises large weights |
| BatchNormalization | Both models | Stabilises training |
| SpatialDropout2D | U-Net encoder | Drops feature maps |
| Data Augmentation | Training only | More diverse training |
| EarlyStopping | Both models | Stops before overfit |
| Class Weights | VGG16 | Handles imbalanced data |

### Anti-Underfitting Techniques
| Technique | Details |
|-----------|---------|
| Two-Phase Training | Head → Fine-tune |
| Unfreeze 12 layers | More VGG16 capacity |
| ReduceLROnPlateau | Auto LR adjustment |
| Custom Class Weights | Glioma=4.5, Meningioma=2.0 |
| Sufficient Epochs | Up to 50 epochs |

### Training Configuration
```
Optimizer      : Adam
Batch Size     : 20
Image Size     : 128 × 128
VGG16 LR Phase1: 0.001
VGG16 LR Phase2: 0.00001
U-Net LR       : 0.0001
VGG16 Loss     : Sparse Categorical Cross-Entropy
U-Net Loss     : BCE + Dice Loss (Combined)
GPU            : Kaggle T4 x2
```

### Callbacks Used
```python
EarlyStopping      → Stops when no improvement
ReduceLROnPlateau  → Reduces LR on plateau
ModelCheckpoint    → Saves best model only
LambdaCallback     → Monitors overfitting gap
```

---

## 📦 Requirements

```txt
tensorflow>=2.19.0
numpy>=1.24.0
opencv-python>=4.8.0
h5py>=3.9.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
Pillow>=10.0.0
nibabel>=5.0.0
```

Install all:
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### ✅ Option 1 — Kaggle (Recommended)

```
1. Go to https://www.kaggle.com
2. Create a new notebook
3. Add datasets:
   → + Add Data → search "brain-tumor-mri-dataset" (masoudnickparvar)
   → + Add Data → search "brats2020-training-data" (awsaf49)
4. Settings → Accelerator → GPU T4 x2
5. Settings → Internet → ON
6. Copy brain_tumor_final.py code into notebook
7. Run All
```

### ✅ Option 2 — Local Machine

```bash
# Step 1: Clone repository
git clone https://github.com/yourusername/brain-tumor-detection.git
cd brain-tumor-detection

# Step 2: Install requirements
pip install -r requirements.txt

# Step 3: Download datasets from Kaggle
# Place in correct folder structure

# Step 4: Update dataset paths in code
# Change '/kaggle/input/...' to your local path

# Step 5: Run
python brain_tumor_final.py
# OR
jupyter notebook brain_tumor_final.ipynb
```

### ✅ Option 3 — Load Pretrained Models

```python
import tensorflow as tf

# Custom functions needed
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.*intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)

def combined_loss(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(
        y_true, y_pred) + dice_loss(y_true, y_pred)

def iou_metric(y_true, y_pred, smooth=1e-6):
    y_bin = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_bin)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_bin) - intersection
    return (intersection + smooth) / (union + smooth)

# Load models
clf_model = tf.keras.models.load_model('best_vgg16.keras')

unet_model = tf.keras.models.load_model(
    'best_unet_brats.keras',
    custom_objects={
        'combined_loss'   : combined_loss,
        'dice_coefficient': dice_coefficient,
        'iou_metric'      : iou_metric
    }
)

print("✅ Models loaded successfully!")
```

---

## 📁 Project Structure

```
brain-tumor-detection/
│
├── 📄 README.md                    ← This file
├── 📄 brain_tumor_final.py         ← Main training code
├── 📄 requirements.txt             ← Dependencies
│
├── 📁 models/
│   ├── best_vgg16.keras            ← Best VGG16 model
│   └── best_unet_brats.keras       ← Best U-Net model
│
├── 📁 results/
│   ├── confusion_matrix.png        ← Confusion matrix plot
│   ├── roc_curves.png              ← ROC curves plot
│   ├── training_history.png        ← Training curves
│   ├── segmentation_results.png    ← U-Net predictions
│   └── per_class_accuracy.png      ← Class accuracy bar chart
│
└── 📁 notebooks/
    └── brain_tumor_kaggle.ipynb    ← Kaggle notebook version
```

---

## 📏 Evaluation Metrics

### Classification Metrics
- **Accuracy** — Overall correct predictions
- **Precision** — Correct positive predictions ratio
- **Recall** — True positive detection rate
- **F1-Score** — Harmonic mean of Precision and Recall
- **ROC-AUC** — Area under ROC curve per class
- **Confusion Matrix** — Detailed prediction breakdown

### Segmentation Metrics

**Dice Coefficient**
```
         2 × |A ∩ B|
Dice = ─────────────────
           |A| + |B|

Range: 0 to 1 (higher = better)
Our result: 0.871 ✅
```

**IoU (Intersection over Union)**
```
         |A ∩ B|
IoU  = ─────────────
         |A ∪ B|

Range: 0 to 1 (higher = better)
Our result: 0.77 ✅
```

**Why not Accuracy for Segmentation?**
> Brain tumor pixels = ~2-5% of total MRI pixels.
> A model predicting all background would get 95%+ accuracy
> but detect ZERO tumors. Dice and IoU specifically measure
> overlap quality — perfect for imbalanced medical imaging.

---

## ⚠️ Limitations

```
1. Glioma recall is 73% — needs improvement
2. Trained on 2D slices — not full 3D MRI volumes
3. Not clinically validated by medical professionals
4. Academic research use only — not for clinical diagnosis
5. Small dataset compared to clinical AI systems
6. Single modality (FLAIR only) for segmentation
```

---

## 🔮 Future Work

- [ ] Replace VGG16 with ResNet50/EfficientNet for better accuracy
- [ ] Train on full 3D MRI volumes for complete analysis
- [ ] Add Grad-CAM visualization to show model attention
- [ ] Deploy as Flask/FastAPI web application
- [ ] Clinical validation with real hospital data
- [ ] Multi-modal MRI fusion (FLAIR + T1 + T2 + T1ce)
- [ ] Real-time inference optimization
- [ ] Mobile app deployment (TensorFlow Lite)

---

## 📚 References

1. **U-Net Paper:**
   Ronneberger, O., Fischer, P., & Brox, T. (2015).
   *U-Net: Convolutional Networks for Biomedical Image Segmentation.*
   MICCAI 2015. [Paper Link](https://arxiv.org/abs/1505.04597)

2. **VGG16 Paper:**
   Simonyan, K., & Zisserman, A. (2014).
   *Very Deep Convolutional Networks for Large-Scale Image Recognition.*
   [Paper Link](https://arxiv.org/abs/1409.1556)

3. **BraTS 2020 Challenge:**
   Menze, B. H., et al. (2015).
   *The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS).*
   IEEE Transactions on Medical Imaging.

4. **Transfer Learning:**
   Tajbakhsh, N., et al. (2016).
   *Convolutional Neural Networks for Medical Image Analysis.*
   IEEE Transactions on Medical Imaging.

5. **Brain Tumor MRI Dataset:**
   [Kaggle - masoudnickparvar](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

6. **BraTS2020 Dataset:**
   [Kaggle - awsaf49](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data)

---

## 👨‍💻 Author

```
Name       : Your Name
University : Your University Name
Department : Computer Science / AI
Course     : Your Course Name
Year       : 2025-2026
Email      : your.email@university.edu
GitHub     : https://github.com/yourusername
LinkedIn   : https://linkedin.com/in/yourprofile
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2025 Your Name

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files, to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of the
Software, subject to the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.
```

---

## 🙏 Acknowledgements

- **Kaggle** for providing free GPU resources (T4 x2)
- **masoudnickparvar** for the Brain Tumor MRI Dataset
- **awsaf49** for the BraTS2020 Dataset
- **Ronneberger et al.** for the original U-Net architecture
- **Simonyan & Zisserman** for VGG16 architecture
- **TensorFlow/Keras** team for the deep learning framework
- **University supervisors** for guidance and support

---

<p align="center">
  Made with ❤️ for Academic Research
  <br>
  ⭐ Star this repo if you found it helpful!
</p>
