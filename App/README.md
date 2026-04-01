# App

Streamlit app for interactive prediction of student outcomes (**Dropout**, **Enrolled**, **Graduate**) using trained KNN and SVM model bundles.

## Run locally
1. Open a terminal in this folder:
   - `cd App`
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Start the app:
   - `streamlit run main.py`

## Usage
1. Upload one or both trained `.pkl` model files from the sidebar.
2. Fill in student inputs in the form.
3. Click **Predict Outcome** to view the prediction (or model comparison if both models are uploaded).
