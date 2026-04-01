import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Outcome Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.main { background: #0f1117; }

.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    color: #f0f0f0;
    margin: 0;
    line-height: 1.1;
}
.hero-sub {
    color: #888;
    font-size: 1rem;
    margin-top: 0.4rem;
    font-weight: 300;
}
.result-card {
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-top: 1.5rem;
}
.result-graduate { background: #0d2b1a; border: 1px solid #1a5c34; }
.result-dropout  { background: #2b0d0d; border: 1px solid #5c1a1a; }
.result-enrolled { background: #1a1a0d; border: 1px solid #4a4a10; }

.result-label {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    font-weight: 400;
}
.result-graduate .result-label { color: #4ade80; }
.result-dropout  .result-label { color: #f87171; }
.result-enrolled .result-label { color: #facc15; }

.result-desc { color: #aaa; font-size: 0.9rem; margin-top: 0.5rem; }

.metric-box {
    background: #1a1d26;
    border: 1px solid #2a2d3a;
    border-radius: 12px;
    padding: 1.2rem 1rem;
    text-align: center;
}
.metric-val { font-size: 1.6rem; font-weight: 600; color: #e0e0e0; }
.metric-lbl { font-size: 0.75rem; color: #666; text-transform: uppercase;
               letter-spacing: 0.08em; margin-top: 0.2rem; }

.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.6rem;
    margin-top: 1.4rem;
}
.stButton > button {
    background: #2563eb;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.7rem 2rem;
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;
    font-weight: 500;
    width: 100%;
    cursor: pointer;
    transition: background 0.2s;
}
.stButton > button:hover { background: #1d4ed8; }
.compare-row {
    display: flex;
    gap: 1rem;
    margin-top: 1rem;
}
.compare-card {
    flex: 1;
    background: #1a1d26;
    border: 1px solid #2a2d3a;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
}
.compare-model { font-size: 0.75rem; color: #666; text-transform: uppercase;
                  letter-spacing: 0.1em; }
.compare-pred  { font-size: 1.3rem; font-weight: 500; margin-top: 0.4rem; }
.agree-badge {
    display: inline-block;
    background: #0d2b1a;
    color: #4ade80;
    border: 1px solid #1a5c34;
    border-radius: 20px;
    padding: 0.3rem 1rem;
    font-size: 0.8rem;
    margin-top: 0.8rem;
}
.disagree-badge {
    display: inline-block;
    background: #2b1a0d;
    color: #fb923c;
    border: 1px solid #5c3010;
    border-radius: 20px;
    padding: 0.3rem 1rem;
    font-size: 0.8rem;
    margin-top: 0.8rem;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def predict(bundle, student_df):
    X = bundle['imputer'].transform(student_df)
    X = bundle['selector'].transform(X)
    X = bundle['scaler'].transform(X)
    code = bundle['model'].predict(X)
    return bundle['encoder'].inverse_transform(code)[0]

DESCRIPTIONS = {
    'Graduate':  'This student is likely to complete their degree successfully.',
    'Dropout':   'This student shows risk factors associated with leaving the programme.',
    'Enrolled':  'This student is likely to remain enrolled and continue their studies.',
}
COLORS = {'Graduate': '#4ade80', 'Dropout': '#f87171', 'Enrolled': '#facc15'}
CLASSES = {'Graduate': 'result-graduate', 'Dropout': 'result-dropout', 'Enrolled': 'result-enrolled'}


# ── Sidebar — model loading ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎓 Load Models")
    st.markdown("<p style='color:#666;font-size:0.85rem'>Upload your trained .pkl files from Colab</p>",
                unsafe_allow_html=True)

    knn_file = st.file_uploader("KNN model (.pkl)", type="pkl", key="knn")
    svm_file = st.file_uploader("SVM model (.pkl)", type="pkl", key="svm")

    knn_bundle, svm_bundle = None, None

    if knn_file:
        knn_bundle = pickle.loads(knn_file.read())
        st.success("KNN loaded")
    if svm_file:
        svm_bundle = pickle.loads(svm_file.read())
        st.success("SVM loaded")

    st.markdown("---")
    st.markdown("<p style='color:#555;font-size:0.8rem'>No models yet? <br>Train them in Colab first, then download the .pkl files and upload here.</p>",
                unsafe_allow_html=True)


# ── Main layout ───────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding: 2rem 0 1.5rem 0'>
  <p class='hero-title'>Student Outcome Predictor</p>
  <p class='hero-sub'>Predict dropout risk using trained KNN & SVM models</p>
</div>
""", unsafe_allow_html=True)

col_form, col_result = st.columns([1.1, 0.9], gap="large")

with col_form:
    st.markdown("<p class='section-label'>Academic Performance</p>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        units1_enrolled  = st.number_input("Units enrolled (sem 1)", 0, 10, 6)
        units1_approved  = st.number_input("Units approved (sem 1)", 0, 10, 5)
        grade1           = st.number_input("Grade sem 1 (0-20)",    0.0, 20.0, 13.0, step=0.1)
    with c2:
        units2_enrolled  = st.number_input("Units enrolled (sem 2)", 0, 10, 6)
        units2_approved  = st.number_input("Units approved (sem 2)", 0, 10, 5)
        grade2           = st.number_input("Grade sem 2 (0-20)",    0.0, 20.0, 13.0, step=0.1)

    st.markdown("<p class='section-label'>Student Profile</p>", unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        age        = st.slider("Age at enrollment", 17, 60, 20)
        prev_grade = st.number_input("Prev qualification grade", 0.0, 200.0, 122.0, step=1.0)
    with c4:
        tuition_ok = st.selectbox("Tuition fees up to date", [1, 0],
                                   format_func=lambda x: "Yes" if x else "No")
        scholarship = st.selectbox("Scholarship holder", [0, 1],
                                    format_func=lambda x: "Yes" if x else "No")

    st.markdown("<p class='section-label'>Socioeconomic Context</p>", unsafe_allow_html=True)
    c5, c6 = st.columns(2)
    with c5:
        unemployment = st.slider("Unemployment rate (%)", 5.0, 20.0, 10.8, step=0.1)
        gdp          = st.slider("GDP growth rate",      -5.0, 5.0, 1.0, step=0.1)
    with c6:
        inflation    = st.slider("Inflation rate (%)",  -2.0, 5.0, 1.4, step=0.1)
        debtor       = st.selectbox("Debtor", [0, 1],
                                     format_func=lambda x: "Yes" if x else "No")

    predict_btn = st.button("Predict Outcome", use_container_width=True)


# ── Build full 36-feature row ─────────────────────────────────────────────────
def build_student_row():
    return pd.DataFrame([{
        'Marital Status': 1,
        'Application mode': 17,
        'Application order': 1,
        'Course': 171,
        'Daytime/evening attendance': 1,
        'Previous qualification': 1,
        'Previous qualification (grade)': prev_grade,
        'Nacionality': 1,
        "Mother's qualification": 19,
        "Father's qualification": 12,
        "Mother's occupation": 5,
        "Father's occupation": 5,
        'Admission grade': prev_grade * 0.8,
        'Displaced': 0,
        'Educational special needs': 0,
        'Debtor': debtor,
        'Tuition fees up to date': tuition_ok,
        'Gender': 1,
        'Scholarship holder': scholarship,
        'Age at enrollment': age,
        'International': 0,
        'Curricular units 1st sem (credited)': 0,
        'Curricular units 1st sem (enrolled)': units1_enrolled,
        'Curricular units 1st sem (evaluations)': units1_enrolled,
        'Curricular units 1st sem (approved)': units1_approved,
        'Curricular units 1st sem (grade)': grade1,
        'Curricular units 1st sem (without evaluations)': 0,
        'Curricular units 2nd sem (credited)': 0,
        'Curricular units 2nd sem (enrolled)': units2_enrolled,
        'Curricular units 2nd sem (evaluations)': units2_enrolled,
        'Curricular units 2nd sem (approved)': units2_approved,
        'Curricular units 2nd sem (grade)': grade2,
        'Curricular units 2nd sem (without evaluations)': 0,
        'Unemployment rate': unemployment,
        'Inflation rate': inflation,
        'GDP': gdp,
    }])


# ── Results panel ─────────────────────────────────────────────────────────────
with col_result:
    if predict_btn:
        if not knn_bundle and not svm_bundle:
            st.warning("Upload at least one model from the sidebar to get predictions.")
        else:
            student_df = build_student_row()

            if knn_bundle and not svm_bundle:
                # Single model — KNN only
                pred = predict(knn_bundle, student_df)
                st.markdown(f"""
                <div class='result-card {CLASSES[pred]}'>
                    <div style='font-size:0.75rem;color:#666;text-transform:uppercase;
                                letter-spacing:0.1em;margin-bottom:0.5rem'>KNN Prediction</div>
                    <div class='result-label'>{pred}</div>
                    <div class='result-desc'>{DESCRIPTIONS[pred]}</div>
                </div>""", unsafe_allow_html=True)

            elif svm_bundle and not knn_bundle:
                # Single model — SVM only
                pred = predict(svm_bundle, student_df)
                st.markdown(f"""
                <div class='result-card {CLASSES[pred]}'>
                    <div style='font-size:0.75rem;color:#666;text-transform:uppercase;
                                letter-spacing:0.1em;margin-bottom:0.5rem'>SVM Prediction</div>
                    <div class='result-label'>{pred}</div>
                    <div class='result-desc'>{DESCRIPTIONS[pred]}</div>
                </div>""", unsafe_allow_html=True)

            else:
                # Both models — show comparison
                pred_knn = predict(knn_bundle, student_df)
                pred_svm = predict(svm_bundle, student_df)
                agree    = pred_knn == pred_svm

                st.markdown("<p class='section-label' style='margin-top:0'>Model Predictions</p>",
                            unsafe_allow_html=True)
                st.markdown(f"""
                <div class='compare-row'>
                  <div class='compare-card'>
                    <div class='compare-model'>KNN</div>
                    <div class='compare-pred' style='color:{COLORS[pred_knn]}'>{pred_knn}</div>
                  </div>
                  <div class='compare-card'>
                    <div class='compare-model'>SVM</div>
                    <div class='compare-pred' style='color:{COLORS[pred_svm]}'>{pred_svm}</div>
                  </div>
                </div>
                <div style='text-align:center'>
                  {'<span class="agree-badge">Both models agree</span>' if agree
                   else '<span class="disagree-badge">Models disagree — review manually</span>'}
                </div>
                """, unsafe_allow_html=True)

                if agree:
                    st.markdown(f"""
                    <div class='result-card {CLASSES[pred_knn]}' style='margin-top:1rem'>
                        <div class='result-label'>{pred_knn}</div>
                        <div class='result-desc'>{DESCRIPTIONS[pred_knn]}</div>
                    </div>""", unsafe_allow_html=True)

            # ── Input summary ─────────────────────────────────────────────
            st.markdown("<p class='section-label'>Input Summary</p>", unsafe_allow_html=True)
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                st.markdown(f"""<div class='metric-box'>
                    <div class='metric-val'>{units1_approved}/{units1_enrolled}</div>
                    <div class='metric-lbl'>Sem 1 units</div></div>""",
                    unsafe_allow_html=True)
            with mc2:
                st.markdown(f"""<div class='metric-box'>
                    <div class='metric-val'>{grade1:.1f}</div>
                    <div class='metric-lbl'>Sem 1 grade</div></div>""",
                    unsafe_allow_html=True)
            with mc3:
                st.markdown(f"""<div class='metric-box'>
                    <div class='metric-val'>{age}</div>
                    <div class='metric-lbl'>Age</div></div>""",
                    unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style='background:#1a1d26;border:1px dashed #2a2d3a;border-radius:16px;
                    padding:3rem 2rem;text-align:center;margin-top:2rem'>
            <div style='font-size:2.5rem;margin-bottom:1rem'>🎓</div>
            <div style='color:#aaa;font-size:1rem;font-weight:300'>
                Fill in the student details<br>and click <strong>Predict Outcome</strong>
            </div>
            <div style='color:#555;font-size:0.8rem;margin-top:1rem'>
                Upload model files in the sidebar first
            </div>
        </div>
        """, unsafe_allow_html=True)
