# 🌿 Cross-Domain Plant Species Identification

**Group 12: VerdantVision** | **COS30082 – Applied Machine Learning** *Swinburne University of Technology*

---

## 📘 Overview

This project addresses the **domain shift** challenge in plant classification, where models trained on accessible *herbarium sheets* must identify species in *natural field photographs*.

**Key Challenges:**
- **Domain Gap:** Visual disparity between pressed sheets (white background) and natural photos (cluttered background).
- **Long-Tail Distribution:** High class imbalance with rare species.
- **Zero-Shot "Without Pairs":** Identifying species in the field that were never seen in the field during training.

[cite_start]**Dataset:** [PlantCLEF 2020 Challenge](https://www.imageclef.org/PlantCLEF2020) [cite: 18]
- **Classes:** 100 species total.
- **Training:** 4,744 images (3,700 Herbarium + 1,044 Field).
- **Testing:** 207 Field images (focusing on unseen/rare species).

---

## 🧠 Methodology & Approaches

We implemented two strong baselines and a novel hybrid approach combining generative data augmentation with metric learning.

| Type | Model / Method | Key Details |
|------|----------------|-------------|
| **Baseline 1** | **ConvNeXt-Base** (CNN) | A modern CNN architecture fine-tuned on the dataset. Serves as a strong supervised learning baseline. |
| **Baseline 2** | **DINOv2** (ViT-B/14) | Leverages the self-supervised Vision Transformer pre-trained on plant data. [cite_start]Used for robust feature extraction and fine-tuning. [cite: 36] |
| **New Approach** | **Metric-Hybrid Ensemble** | **1. Architecture:** An ensemble of a Generalist (Classifier) and a Specialist (Metric Learner) using Triplet Loss.<br>**2. Inference:** Dynamic thresholding to switch between models based on confidence.<br>**3. Data:** Augmented with **FastCut (CycleGAN)** synthetic images. |

---

## 🧪 Experimental Highlights

### 1. Data Augmentation with FastCut
To bridge the visual gap, we employed **FastCut** (a CycleGAN variant) to generate synthetic "Field" images from "Herbarium" samples.
- **Goal:** Augment the training set for species "without pairs."
- **Result:** Generated ~1,700 synthetic images, significantly improving model robustness.

### 2. The Failed Experiment: DANN
We attempted a **Domain-Adversarial Neural Network (DANN)** approach using taxonomy as an auxiliary task.
- **Outcome:** Failed to generalize (High Training Acc, Low Test Acc).
- **Lesson:** Adversarial training proved unstable for fine-grained species classification on this specific dataset.

### 3. The Solution: Metric-Hybrid Ensemble
Our final proposed model ensembles the DINOv2 backbone with a **Metric Learning head**.
- **Mechanism:** If the *Specialist* (Metric) model predicts with high confidence (high similarity score), its prediction is used. Otherwise, the model falls back to the *Generalist* (Cross-Entropy) head.
- **Performance:** Shows superior handling of "Without Pairs" species compared to standard classification.

---

## 📊 Results Summary

| Model | Top-1 Accuracy | Observation |
|-------|----------------|-------------|
| **ConvNeXt** | ~71.5% | High gap between "Seen" and "Unseen" domains. |
| **DINOv2** | ~72.9% | Better generalization due to ViT features. |
| **Hybrid Ensemble** | **Best** | Effectively closes the domain gap using synthetic data + metric learning. |

---

## 🛠️ Tech Stack
- **Frameworks:** PyTorch, `timm`, `transformers`
- **Generative Models:** CycleGAN (FastCut)
- **Architecture:** Vision Transformers (ViT), ConvNeXt
