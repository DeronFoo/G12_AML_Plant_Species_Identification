# 🌿 Cross-Domain Plant Species Identification

**COS30082 – Applied Machine Learning** <br>
**Group 12: VerdantVision** <br>
*Swinburne University of Technology*

---

## 📘 Overview

This project addresses the **domain shift** challenge in plant classification, where models trained on accessible *herbarium sheets* must identify species in *natural field photographs*.

**Key Challenges:**
- **Domain Gap:** Visual disparity between pressed sheets (white background) and natural photos (cluttered background).
- **Long-Tail Distribution:** High class imbalance with rare species.
- **Zero-Shot "Without Pairs":** Identifying species in the field that were never seen in the field during training.

**Dataset:** Subset of PlantCLEF 2020 Challenge
- **Classes:** 100 species total.
- **Training:** 4,744 images (3,700 Herbarium + 1,044 Field).
- **Testing:** 207 Field images (focusing on unseen/rare species).

---

## 🧠 Methodology & Approaches

The team implemented two decent baselines and a novel hybrid approach combining generative data augmentation with metric learning.

| Type | Model / Method | Key Details |
|------|----------------|-------------|
| **Baseline 1** | **ConvNeXt-Base** (CNN) | A modern CNN architecture fine-tuned on the dataset. Serves as a strong supervised learning baseline. |
| **Baseline 2** | **DINOv2** (ViT-B/14) | Leverages the self-supervised Vision Transformer pre-trained on plant data. Used for robust feature extraction and fine-tuning. |
| **New Approach** | **Metric-Hybrid Ensemble** | **1. Architecture:** An ensemble of a Generalist (Classifier) and a Specialist (Metric Learner) using Triplet Loss.<br>**2. Inference:** Dynamic thresholding to switch between models based on confidence.<br>**3. Data:** Augmented with **FastCut** synthetic images. |

---

## 🧪 Experimental Highlights

### 1. Data Augmentation with FastCut
To bridge the visual gap, the team employed **FastCut (Contrastive Unpaired Translation)** to generate synthetic "Field" images from "Herbarium" samples.
- **Why FastCut?** Unlike CycleGAN, FastCut uses contrastive loss to maximise mutual information between input and output patches, allowing for faster and more effective one-sided translation (Herbarium $\to$ Field).
- **Result:** Generated ~1,700 synthetic images, significantly improving model robustness on unseen classes.

### 2. The Failed Experiment: DANN
The team attempted a **Domain-Adversarial Neural Network (DANN)** approach using taxonomy as an auxiliary task.
- **Outcome:** Failed to generalise (High Training Acc, Low Test Acc).
- **Lesson:** Adversarial training proved unstable for fine-grained species classification on this specific dataset.

### 3. The Solution: Metric-Hybrid Ensemble
The team's final proposed model ensembles the DINOv2 backbone with a **Metric Learning head**.
- **Mechanism:** If the *Specialist* (Metric) model predicts with high confidence (high similarity score), its prediction is used. Otherwise, the model falls back to the *Generalist* (Cross-Entropy) head.
- **Performance:** Shows superior handling of "Without Pairs" species compared to standard classification.

---

## 📊 Results Summary

The table below shows the Top-1 Accuracy across different data splits. "Without Pairs" represents the critical zero-shot capability (species not seen in the field during training).

| Model | Overall Top-1 | With-Pair (Seen) | Without-Pair (Unseen) | Observation |
|:---|:---:|:---:|:---:|:---|
| **ConvNeXt** | 67.63% | 86.93% | 12.96% | Struggles significantly with unseen domains (massive gap). |
| **DINOv2** | 72.95% | 92.81% | 16.67% | Stronger features improve "Seen" accuracy, but the domain gap remains. |
| **Hybrid Ensemble** | **80.68%** | **94.77%** | **40.74%** | **Significant breakthrough.** Synthetic data + Metric learning more than doubled the performance on unseen classes. |

---

## 🛠️ Tech Stack
- **Frameworks:** PyTorch, `timm`, `transformers`
- **Generative Models:** FastCut
- **Architecture:** Vision Transformers (ViT), ConvNeXt
