# Naive Bayes Module Guide

This module contains the full notebook workflow for training and evaluating a **Gaussian Naive Bayes** classifier for student outcome prediction.

Target classes:
- Dropout
- Enrolled
- Graduate

---

## 1) Main File

- Notebook: `NaiveBayes.ipynb`

---

## 2) What This Notebook Covers

- Loading the UCI student outcome dataset
- Preparing features and multiclass target labels
- Applying preprocessing (including imputation/scaling/filtering)
- Tuning `var_smoothing` with stratified 5-fold cross-validation
- Evaluating baseline performance
- Comparing baseline vs. SMOTE-resampled training
- Exporting the trained model bundle for app use

---

## 3) Output Artifact

The notebook saves a model bundle at:
- `saved_models/nb_model.pkl`

This file is intended to be uploaded into `App/main.py` through the Streamlit sidebar.

---

## 4) Reported Results (Current Notebook)

- Best `var_smoothing`: tuned via 5-fold CV over a log-scale grid
- Test accuracy and F1 scores reported after running all cells

SMOTE may improve minority-class recall depending on dataset split behavior.

---

## 5) How to Run

1. Open `NaiveBayes.ipynb` in Jupyter/Colab.
2. Install required dependencies in your environment.
3. Run all cells from top to bottom.
4. Confirm `saved_models/nb_model.pkl` is generated.

---

## 6) Notes

- GaussianNB assumes each feature follows a Gaussian distribution within each class.
- `var_smoothing` controls the portion of the largest variance added to all variances to improve numerical stability.
- Keep preprocessing and model steps bundled together for consistent inference.
