# Identifying Low-Resistivity Oil Layers During Drilling Using Small Samples

This repository contains the official implementation of the methodology described in:

> **"Identifying Low-Resistivity Oil Layers During Drilling Using Small Samples"**  
> by Jialing Zou  
> *Submitted to Computers & Geosciences*, 2025.

## 🔒 Data Availability Statement

Due to confidentiality agreements with the oilfield operator, **the real well logging dataset cannot be made publicly available**.

To ensure full reproducibility, we provide:
- A **synthetic data generator** (`data/generate_synthetic_data.py`) that mimics realistic GR, TG, and hydrocarbon gas features for three classes:
  - Low-Resistivity Oil Layer
  - Normal Oil Layer
  - Water Layer
- The complete pipeline: **Improved SMOTE → Kernel PCA → PSO-Optimized CatBoost**
- All results (metrics, plots) can be regenerated using synthetic data.

> Real data may be available upon reasonable request under an NDA.

##▶️ How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt