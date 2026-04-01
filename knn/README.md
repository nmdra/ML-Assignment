# KNN Module
This notebook trains and evaluates a K-Nearest Neighbors classifier for student outcome prediction (Dropout/Enrolled/Graduate).

## What KNN Does Here
- Loads UCI dataset 697 and prepares features/target
- Applies preprocessing (encoding, scaling, and variance filtering)
- Tunes `k` via 5-fold stratified cross-validation
- Evaluates baseline KNN and compares SMOTE resampling

## Key Results
- Best `k`: **17** (CV accuracy: **0.7152**)
- Test accuracy (no SMOTE): **0.7085**
- F1 macro (no SMOTE): **0.5942**
- F1 weighted (no SMOTE): **0.6773**
- SMOTE improves enrolled-class recall but lowers overall accuracy

## Observations
- Stronger performance on **Graduate** than **Enrolled** without balancing
- SMOTE introduces a minority-recall vs. overall-accuracy trade-off
- KNN remains simple/interpretable but sensitive to scaling and class balance
