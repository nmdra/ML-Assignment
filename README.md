# ML Assignment
Student outcome prediction using K-Nearest Neighbors (KNN) and Support Vector Machine (SVM) on a multiclass dataset.

## Problem Statement
Predict whether a student will be **Dropout**, **Enrolled**, or **Graduate** from academic/demographic features.

## Dataset
UCI ML Repository: *Predict students dropout and academic success* (ID 697), with files stored in `/data`.

## Algorithms Applied
- Distance-weighted KNN (Euclidean)
- Support Vector Machine (RBF kernel, class-weight balanced)
- 5-fold cross-validation to choose `k`
- SMOTE comparison for class-imbalance analysis

## How to Run
1. Open `KNN/KNN.ipynb` or `SVM/SVM.ipynb` in Jupyter.
2. Install notebook dependencies (`ucimlrepo`, `imbalanced-learn`, `scikit-learn`).
3. Run all cells to reproduce metrics and figures.

## App
This repository includes a Streamlit app in `App/` for interactive student outcome prediction with trained KNN and SVM model bundles.

Quick start:
1. `cd App`
2. Install dependencies from `requirements.txt`.
3. Run `streamlit run main.py`.
4. Upload your trained `.pkl` model files (KNN and/or SVM) from the sidebar.
