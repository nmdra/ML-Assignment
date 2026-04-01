# ML Assignment
Student outcome prediction using K-Nearest Neighbors (KNN) on a multiclass dataset.

## Problem Statement
Predict whether a student will be **Dropout**, **Enrolled**, or **Graduate** from academic/demographic features.

## Dataset
UCI ML Repository: *Predict students dropout and academic success* (ID 697), with files stored in `/data`.

## Algorithms Applied
- Distance-weighted KNN (Euclidean)
- 5-fold cross-validation to choose `k`
- SMOTE comparison for class-imbalance analysis

## How to Run
1. Open `knn/KNN.ipynb` in Jupyter.
2. Install notebook dependencies (`ucimlrepo`, `imbalanced-learn`, `scikit-learn`).
3. Run all cells to reproduce metrics and figures.
