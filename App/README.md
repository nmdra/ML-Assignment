# Streamlit App Guide

This app provides an interactive UI to predict student outcomes:
- **Dropout**
- **Enrolled**
- **Graduate**

The app accepts trained model bundles from the KNN and SVM notebooks and runs inference on user-provided student inputs.

---

## 1) Files in This Folder

- `main.py` — Streamlit application
- `requirements.txt` — pip dependencies
- `pyproject.toml` — project metadata and pinned dependencies

---

## 2) Prerequisites

- Python **3.12+** (recommended from `pyproject.toml`)
- A generated model file from:
  - `KNN/saved_models/knn_model.pkl` and/or
  - `SVM/saved_models/svm_model.pkl`

---

## 3) Run Locally

From repository root:

1. `cd App`
2. `pip install -r requirements.txt`
3. `streamlit run main.py`

Streamlit will print a local URL (typically `http://localhost:8501`) for opening in your browser.

---

## 4) How to Use

1. In the sidebar, upload one or both model files (`.pkl`).
2. Fill in the form inputs (academic, profile, and socioeconomic fields).
3. Click **Predict Outcome**.

Behavior:
- If only one model is uploaded, the app shows one prediction.
- If both models are uploaded, the app shows:
  - KNN prediction
  - SVM prediction
  - Agreement/disagreement status

---

## 5) Expected Model Bundle Structure

The uploaded `.pkl` bundle is expected to include:
- `imputer`
- `selector`
- `scaler`
- `model`
- `encoder`

If this structure is missing, prediction may fail.

---

## 6) Troubleshooting

- **Upload succeeds but prediction fails**: regenerate model bundle from notebook and upload again.
- **Import or package errors**: re-run `pip install -r requirements.txt`.
- **No output**: ensure at least one model file is uploaded before clicking predict.
