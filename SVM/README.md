# SVM Analysis
This notebook trains and evaluates a Support Vector Machine classifier for student outcome prediction (Dropout/Enrolled/Graduate).

## What SVM Does Here
- Loads UCI dataset 697 and prepares features/target
- Applies preprocessing (imputation, encoding, scaling, and variance filtering)
- Selects kernel and tunes `C` via 5-fold stratified cross-validation
- Evaluates baseline SVM and compares SMOTE resampling

## Key Results
- Best kernel: **rbf**
- Best `C`: **100**
- Test accuracy (no SMOTE): **0.6915**
- F1 macro (no SMOTE): **0.6366**
- F1 weighted (no SMOTE): **0.6975**
- SMOTE slightly improves accuracy but lowers macro-F1, so no-SMOTE is selected as best overall

## Observations
- Performance is strongest on **Graduate**, with weaker recall on **Enrolled**
- `class_weight='balanced'` helps class imbalance while preserving overall performance
- SMOTE provides only marginal gains and introduces the usual minority-recall vs. overall-metric trade-off
