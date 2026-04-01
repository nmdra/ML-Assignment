# KNN Module Guide

This module contains the full notebook workflow for training and evaluating a **K-Nearest Neighbors (KNN)** classifier for student outcome prediction.

Target classes:
- Dropout
- Enrolled
- Graduate

---

## 1) Main File

- Notebook: `KNN.ipynb`

---

## 2) What This Notebook Covers

- Loading the UCI student outcome dataset
- Preparing features and multiclass target labels
- Applying preprocessing for KNN readiness
- Tuning `k` with stratified 5-fold cross-validation
- Evaluating baseline performance
- Comparing baseline vs. SMOTE-resampled training
- Exporting the trained model bundle for app use

---

## 3) Output Artifact

The notebook saves a model bundle at:
- `saved_models/knn_model.pkl`

This file is intended to be uploaded into `App/main.py` through the Streamlit sidebar.

---

## 4) Reported Results (Current Notebook)

- Best `k`: **17**
- CV accuracy: **0.7152**
- Test accuracy (no SMOTE): **0.7085**
- F1 macro (no SMOTE): **0.5942**
- F1 weighted (no SMOTE): **0.6773**

SMOTE improves minority recall in some cases, but with an overall metric trade-off.

---

## 5) How to Run

1. Open `KNN.ipynb` in Jupyter/Colab.
2. Install required dependencies in your environment.
3. Run all cells from top to bottom.
4. Confirm `saved_models/knn_model.pkl` is generated.

---

## 6) Notes

- KNN is sensitive to feature scaling and class distribution.
- Keep preprocessing and model steps bundled together for consistent inference.
