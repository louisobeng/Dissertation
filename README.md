# Dissertation
Code, results and experiments for my MSc dissertation on Emotion & Gender Recognition

Fairness-Aware Facial Emotion Recognition

This repository contains the implementation, datasets, and results for our dissertation on bias mitigation in Facial Emotion Recognition (FER). The project introduces a closed-loop fairness pipeline that integrates bias detection, conditional augmentation, and fairness auditing into model training.

Due to the scope of the dissertation, we only update the CK+ Augmented dataset since it has the smaller size for the experiment.

## 📖 Overview  

Facial Emotion Recognition has made strong progress with deep learning, but fairness remains a persistent challenge, particularly with respect to gender bias.  

In this dissertation, we propose a **bias-gated conditional augmentation pipeline** that:  
1. Detects subgroup bias using fairness metrics (accuracy gap, equal opportunity).  
2. Generates targeted synthetic samples with Stable Diffusion XL, conditioned on emotion and gender.  
3. Verifies labels with DeepFace + MTCNN to ensure high-quality augmentation.  
4. Retrains and audits models, measuring both accuracy and fairness.  

Experiments were conducted on **CK+**, **FER2013+**, **AffectNet**, and **CelebA**, using three architectures trained from scratch:  
- Baseline CNN (HACNN)  
- MobileNet-V3  
- EfficientNet-B2  

The results show that targeted augmentation improves both recognition accuracy and fairness across gender subgroups.  

The results show that targeted augmentation improves both recognition accuracy and fairness across gender subgroups.

## 📂 Repository Structure  

```text
Dissertation/
│
├── src/                               # Core source code
│   ├── Conditional_Augmentation/      # Bias-gated conditional data augmentation
│   ├── fairness_evaluation/           # Fairness metrics & heatmap visualisations
│   ├── model_Training/model_training.py # Model architectures & training loop
│   └── Preprocessing/Data_Preprocessing.py # Data preprocessing pipeline
│
├── data/                              # Datasets & augmented samples
│   ├── Generated_Image_Samples/       # First-round generated synthetic images
│   ├── Second_Augmentation/           # Second augmentation iteration
│   │   ├── CKPLUS_Features_20250805_111308_faceprefixed_CLEAN11.csv # Augmented CSV
│   │   └── train/                     # Augmented training set
│
├── results/                           # Experimental outputs
│   ├── fairness_result_HeatMaps/      # Heatmap dashboards of bias metrics
│   │   └── fairness_dashboard_Second_Aug_result.pdf
│   └── basic_Training_results/        # Baseline training results
│       └── baseline/
│           └── Classification Report.pdf
│
└── README.md                          # This file
