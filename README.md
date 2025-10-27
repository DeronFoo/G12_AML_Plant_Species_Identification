# G12_AML_Plant_Species_Identification
A deep learning project for cross-domain plant species identification, using herbarium images for training and field images for testing. Implements baseline CNN and DINOv2 models, along with a novel approach to handle domain shift and class imbalance.

# 🌿 Cross-Domain Plant Species Identification

**Short Description:**  
This repository contains the implementation of a deep learning system for **cross-domain plant species identification** — identifying plant species in *field images* using *herbarium images* as training data.  
Developed as part of the **COS30082 – Applied Machine Learning** course at Swinburne University of Technology.

---

## 📘 Overview

This project explores the challenge of **domain shift** between herbarium and field image datasets.  
We implement **two baseline methods** and propose **one novel deep learning approach** to improve classification accuracy on unseen field images.

**Objectives:**
- Identify plant species from field photographs.
- Handle data imbalance and missing herbarium–field pairs.
- Develop a simple, user-friendly interface for prediction.

**Dataset:** [PlantCLEF 2020 Challenge](https://www.imageclef.org/PlantCLEF2020)  
- 100 species total  
- 4,744 training images (3,700 herbarium + 1,044 field)  
- 207 test field images  

---

## 🧠 Model Approaches

| Type | Approach | Description |
|------|-----------|--------------|
| **Baseline 1** | Mix-Stream CNN | CNN-based approach using pre-trained models (e.g., ResNet50, EfficientNet) trained jointly on both domains. |
| **Baseline 2** | DINOv2 Feature Extractor | Uses the plant-pretrained DINOv2 model as a feature extractor; optional fine-tuning for higher accuracy. |
| **New Approach** | Custom Deep Model | Our proposed architecture addressing imbalanced data and missing herbarium-field pairs. |

---

## 🧩 Repository Structure

