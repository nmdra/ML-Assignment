# Academic-Success-Predictor

This project predicts student academic outcomes (**Dropout**, **Enrolled**, **Graduate**) using machine learning models trained on academic and demographic data.

<img width="1835" height="982" alt="Screenshot 2026-04-02 at 08-08-32 Academic-Success-Predictor" src="https://github.com/user-attachments/assets/6c8d9112-32f7-45d0-aad0-a102de650b95" />



It includes:
- A full **KNN pipeline** (`KNN/KNN.ipynb`)
- A full **SVM pipeline** (`SVM/SVM.ipynb`)
- A full **Naive Bayes pipeline** (`NaiveBayes/NaiveBayes.ipynb`)
- A **Streamlit app** for interactive predictions (`App/main.py`)

---

## 1) Project Objective

Build and compare three multiclass classifiers to estimate student outcomes:
- **K-Nearest Neighbors (KNN)**
- **Support Vector Machine (SVM)**
- **Naive Bayes (Gaussian NB)**

Then expose model predictions through a simple web UI for easier demonstration and analysis.

---

## 2) Dataset

- Source: **UCI Machine Learning Repository**
- Dataset: *Predict students dropout and academic success* (Dataset ID 697)
- Local data folder: `data/`

Target classes:
- `Dropout`
- `Enrolled`
- `Graduate`

---

## 3) Repository Structure

```text
Academic-Success-Predictor/
├── App/
│   ├── main.py
│   ├── README.md
│   ├── requirements.txt
│   └── pyproject.toml
├── KNN/
│   ├── KNN.ipynb
│   └── README.md
├── SVM/
│   ├── SVM.ipynb
│   └── README.md
├── NaiveBayes/
│   ├── NaiveBayes.ipynb
│   └── README.md
├── data/
├── figures/
├── SECURITY.md
└── README.md
```

---

## 4) Modeling Approach

### KNN workflow
- Preprocessing and feature preparation
- Scaling + feature filtering
- `k` selection using stratified 5-fold cross-validation
- Baseline evaluation and SMOTE comparison

### SVM workflow
- Preprocessing (including imputation and scaling)
- Kernel and `C` tuning via stratified 5-fold cross-validation
- Baseline evaluation and SMOTE comparison
- `class_weight='balanced'` for class imbalance handling

### Naive Bayes workflow
- Preprocessing (including imputation and scaling)
- `var_smoothing` tuning via stratified 5-fold cross-validation
- Baseline evaluation and SMOTE comparison

---

## 5) Trained Model Files

Each notebook exports a model bundle (`.pkl`) that includes preprocessing objects and the trained estimator.

- KNN output: `KNN/saved_models/knn_model.pkl`
- SVM output: `SVM/saved_models/svm_model.pkl`
- Naive Bayes output: `NaiveBayes/saved_models/nb_model.pkl`

These `.pkl` files are used directly by the Streamlit app.

---

## 6) How to Run the Notebooks

1. Open one notebook:
   - `KNN/KNN.ipynb`
   - `SVM/SVM.ipynb`
   - `NaiveBayes/NaiveBayes.ipynb`
2. Install required Python packages (examples):
   - `scikit-learn`
   - `imbalanced-learn`
   - `pandas`
   - `numpy`
   - `ucimlrepo`
3. Run all cells to reproduce training, evaluation metrics, and saved model exports.

---

## 7) Run the Streamlit App

From repository root:

1. Change directory:
   - `cd App`
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Start app:
   - `streamlit run main.py`
4. In the sidebar, upload one or both model files:
   - `knn_model.pkl`
   - `svm_model.pkl`
5. Enter student attributes and click **Predict Outcome**.

If both models are uploaded, the app displays both predictions and agreement status.

---

## 8) Reproducibility Notes

- Use the notebook pipelines as-is for consistent preprocessing and evaluation.
- Keep model and preprocessing steps bundled together in exported `.pkl` files.
- Use the same feature definitions expected by each trained bundle.

---

## 9) Troubleshooting

- **ModuleNotFoundError**: install missing dependencies in your active Python environment.
- **Prediction button does nothing**: ensure at least one `.pkl` model is uploaded in the sidebar.
- **Model load/prediction errors**: regenerate model files from notebooks and re-upload.

---

## 10) Related Documentation

- KNN details: `KNN/README.md`
- SVM details: `SVM/README.md`
- Naive Bayes details: `NaiveBayes/README.md`
- App usage: `App/README.md`
