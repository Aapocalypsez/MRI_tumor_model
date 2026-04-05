🧠 Brain Tumor Detection & Segmentation using VGG16 
Transfer Learning (85% accuracy) and U-Net Deep Learning 
(Dice: 0.871) on MRI Images | TensorFlow | Kaggle | BraTS2020


# 🧠 Brain Tumor Detection & Segmentation

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange)
![Kaggle](https://img.shields.io/badge/Platform-Kaggle-blue)
![Accuracy](https://img.shields.io/badge/Accuracy-85.25%25-green)
![Dice](https://img.shields.io/badge/Dice_Score-0.871-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📌 Overview
This project implements a deep learning pipeline for 
**Brain Tumor Detection and Segmentation** using MRI images.
It combines two powerful models:
- **VGG16** Transfer Learning for tumor classification
- **U-Net** Architecture for pixel-level tumor segmentation

> ⚠️ **Disclaimer:** This project is for academic and research 
> purposes only. Not intended for clinical diagnosis.

---

## 🎯 Features
- ✅ Multi-class tumor classification (4 classes)
- ✅ Pixel-level tumor segmentation
- ✅ Real BraTS2020 ground truth masks
- ✅ Two-phase training strategy
- ✅ Anti-overfitting techniques
- ✅ Combined VGG16 + U-Net pipeline
- ✅ Detailed evaluation metrics

---

## 📊 Results

### VGG16 Classification
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Glioma | 0.85 | 0.73 | 0.78 |
| Meningioma | 0.74 | 0.85 | 0.79 |
| No Tumor | 0.91 | 0.99 | 0.95 |
| Pituitary | 0.94 | 0.84 | 0.89 |
| **Overall** | **0.86** | **0.85** | **0.85** |

**Overall Accuracy: 85.25%** ✅

### U-Net Segmentation
| Metric | Score |
|--------|-------|
| Dice Coefficient | 0.871 ✅ |
| IoU Score | 0.770 ✅ |
| Train-Val Gap | 0.006 ✅ |
| Overfitting | None ✅ |

---

## 🗂️ Dataset

### Dataset 1 — Classification
- **Name:** Brain Tumor MRI Dataset
- **Source:** [Kaggle - masoudnickparvar](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **Classes:** Glioma, Meningioma, No Tumor, Pituitary
- **Total Images:** ~7,000

### Dataset 2 — Segmentation
- **Name:** BraTS2020 Training Data
- **Source:** [Kaggle - awsaf49](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data)
- **Total Files:** 57,198 pre-sliced .h5 files
- **Contains:** MRI images + Ground truth masks

---

## 🏗️ Model Architecture

### VGG16 Classifier
