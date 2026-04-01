# SVM Module Guide

This module contains the full notebook workflow for training and evaluating a **Support Vector Machine (SVM)** classifier for student outcome prediction.

Target classes:
- Dropout
- Enrolled
- Graduate

---

## 1) Main File

- Notebook: `SVM.ipynb`

---

## 2) What This Notebook Covers

- Loading the UCI student outcome dataset
- Preparing features and multiclass target labels
- Applying preprocessing (including imputation/scaling/filtering)
- Tuning kernel and `C` with stratified 5-fold cross-validation
- Evaluating baseline performance
- Comparing baseline vs. SMOTE-resampled training
- Exporting the trained model bundle for app use

---

## 3) Output Artifact

The notebook saves a model bundle at:
- `saved_models/svm_model.pkl`

This file is intended to be uploaded into `App/main.py` through the Streamlit sidebar.

---

## 4) Reported Results (Current Notebook)

- Best kernel: **rbf**
- Best `C`: **100**
- Test accuracy (no SMOTE): **0.6915**
- F1 macro (no SMOTE): **0.6366**
- F1 weighted (no SMOTE): **0.6975**

SMOTE can improve selected metrics but may reduce macro-level balance depending on split behavior.

---

## 5) How to Run

1. Open `SVM.ipynb` in Jupyter/Colab.
2. Install required dependencies in your environment.
3. Run all cells from top to bottom.
4. Confirm `saved_models/svm_model.pkl` is generated.

---

## 6) Notes

- SVM performance depends on kernel and regularization (`C`) choices.
- The model uses class balancing to improve minority-class behavior.
