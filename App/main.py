import streamlit as st
import pickle
import numpy as np
import pandas as pd

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
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700&family=DM+Mono:wght@300;400&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.main { background: #0a0c12; }

/* ── Hero ── */
.hero-wrap {
    padding: 2.2rem 0 1.8rem 0;
    border-bottom: 1px solid #1c1f2b;
    margin-bottom: 1.6rem;
}
.hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.18em;
    color: #4a6cf7;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    color: #f2f2f2;
    margin: 0;
    line-height: 1.1;
    letter-spacing: -0.02em;
}
.hero-sub {
    color: #555e78;
    font-size: 0.9rem;
    margin-top: 0.45rem;
    font-weight: 400;
}

/* ── Section labels ── */
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    font-weight: 400;
    color: #3a4060;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 0.5rem;
    margin-top: 1.6rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #141720;
}

/* ── Predict button ── */
.stButton > button {
    background: #4a6cf7;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.75rem 2rem;
    font-family: 'Syne', sans-serif;
    font-size: 0.95rem;
    font-weight: 600;
    width: 100%;
    cursor: pointer;
    letter-spacing: 0.03em;
    transition: all 0.18s ease;
    margin-top: 1rem;
}
.stButton > button:hover {
    background: #3a5ce6;
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(74,108,247,0.3);
}
.stButton > button:active { transform: translateY(0); }

/* ── Result cards ── */
.result-card {
    border-radius: 14px;
    padding: 1.8rem;
    text-align: center;
    margin-top: 1rem;
}
.result-graduate { background: #071a10; border: 1px solid #144d24; }
.result-dropout  { background: #1a0707; border: 1px solid #4d1414; }
.result-enrolled { background: #12120a; border: 1px solid #4a4a10; }
.result-unknown  { background: #111520; border: 1px solid #2a2d4a; }

.result-emoji { font-size: 2.4rem; margin-bottom: 0.6rem; }
.result-label {
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.02em;
}
.result-graduate .result-label { color: #3dda6e; }
.result-dropout  .result-label { color: #f26060; }
.result-enrolled .result-label { color: #f2c94c; }
.result-unknown  .result-label { color: #8899bb; }
.result-desc { color: #666e88; font-size: 0.85rem; margin-top: 0.5rem; line-height: 1.5; }

/* ── Probability bar ── */
.prob-row {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    margin: 0.35rem 0;
}
.prob-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #6a7290;
    min-width: 72px;
    text-align: right;
}
.prob-bar-bg {
    flex: 1;
    height: 6px;
    background: #141720;
    border-radius: 99px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 99px;
    transition: width 0.5s ease;
}
.prob-pct {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    min-width: 36px;
    text-align: right;
}

/* ── Compare ── */
.compare-row {
    display: flex;
    gap: 0.8rem;
    margin-top: 0.8rem;
}
.compare-card {
    flex: 1;
    background: #111420;
    border: 1px solid #1c2035;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.compare-model {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #3a4060;
    text-transform: uppercase;
    letter-spacing: 0.12em;
}
.compare-pred { font-size: 1.15rem; font-weight: 700; margin-top: 0.35rem; letter-spacing: -0.01em; }

.badge {
    display: inline-block;
    border-radius: 20px;
    padding: 0.25rem 0.9rem;
    font-size: 0.75rem;
    font-family: 'DM Mono', monospace;
    margin-top: 0.75rem;
    letter-spacing: 0.04em;
}
.badge-agree    { background: #071a10; color: #3dda6e; border: 1px solid #144d24; }
.badge-disagree { background: #1a1207; color: #f2a84c; border: 1px solid #4d3014; }

/* ── Metric boxes ── */
.metric-box {
    background: #0e1018;
    border: 1px solid #181b28;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.metric-val {
    font-size: 1.45rem;
    font-weight: 700;
    color: #dde0f0;
    letter-spacing: -0.01em;
}
.metric-lbl {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    color: #3a4060;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.2rem;
}

/* ── Model metadata ── */
.meta-box {
    background: #0b0d18;
    border: 1px solid #141720;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-top: 0.5rem;
}
.meta-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.3rem 0;
    border-bottom: 1px solid #141720;
}
.meta-row:last-child { border-bottom: none; }
.meta-key {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #3a4060;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.meta-val {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: #8899cc;
    font-weight: 400;
}

/* ── Validation warnings ── */
.warn-box {
    background: #1a1207;
    border: 1px solid #4d3014;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin-top: 0.8rem;
}
.warn-title {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #f2a84c;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.warn-item { font-size: 0.8rem; color: #a08060; margin: 0.15rem 0; }

/* ── Placeholder ── */
.placeholder-box {
    background: #0e1018;
    border: 1px dashed #1c2035;
    border-radius: 14px;
    padding: 3rem 2rem;
    text-align: center;
    margin-top: 1.5rem;
}
.placeholder-icon { font-size: 2.8rem; margin-bottom: 1rem; opacity: 0.6; }
.placeholder-text { color: #3a4060; font-size: 0.9rem; line-height: 1.6; }

/* ── Sidebar ── */
.sidebar-title {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #3a4060;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    margin-bottom: 0.8rem;
}
.model-loaded {
    background: #071a10;
    border: 1px solid #144d24;
    border-radius: 8px;
    padding: 0.6rem 0.8rem;
    margin: 0.3rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.model-loaded-name {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: #3dda6e;
}
.dot-green { width: 6px; height: 6px; border-radius: 50%; background: #3dda6e; display: inline-block; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model_from_bytes(data: bytes):
    return pickle.loads(data)

def safe_load_model(file_obj):
    """Load model with error handling. Returns (bundle, error_msg)."""
    try:
        data = file_obj.read()
        bundle = pickle.loads(data)
        # Validate required keys
        required = ['model', 'scaler', 'imputer', 'selector', 'encoder']
        missing = [k for k in required if k not in bundle]
        if missing:
            return None, f"Missing keys in pkl: {', '.join(missing)}"
        return bundle, None
    except pickle.UnpicklingError:
        return None, "File is not a valid pickle — was it corrupted?"
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"

def predict(bundle, student_df):
    """Run inference pipeline. Returns (class_label, probabilities_dict or None)."""
    X = bundle['imputer'].transform(student_df)
    X = bundle['selector'].transform(X)
    X = bundle['scaler'].transform(X)
    code = bundle['model'].predict(X)[0]
    label = bundle['encoder'].inverse_transform([code])[0]

    # Try to get probabilities (works for KNN; SVM needs probability=True)
    proba_dict = None
    if hasattr(bundle['model'], 'predict_proba'):
        try:
            proba = bundle['model'].predict_proba(X)[0]
            classes = bundle['encoder'].inverse_transform(
                range(len(bundle['encoder'].classes_))
            )
            proba_dict = dict(zip(classes, proba))
        except Exception:
            pass

    return label, proba_dict

def get_model_metadata(bundle, model_type: str) -> dict:
    """Extract human-readable metadata from a model bundle."""
    meta = {}
    model = bundle['model']
    classes = list(bundle['encoder'].classes_)
    meta['Classes'] = ', '.join(classes)
    meta['Features used'] = str(bundle['selector'].get_support().sum())

    if model_type == 'KNN':
        meta['Neighbors (K)'] = str(model.n_neighbors)
        meta['Weights'] = model.weights
        meta['Metric'] = model.metric
        meta['Algorithm'] = model.algorithm
    elif model_type == 'SVM':
        meta['Kernel'] = model.kernel
        meta['C (regularization)'] = str(model.C)
        meta['Class weight'] = str(model.class_weight)
        if model.kernel == 'rbf':
            meta['Gamma'] = str(model.gamma)
    return meta

def validate_inputs(u1e, u1a, u2e, u2a, g1, g2) -> list:
    """Return list of warning strings for suspicious input combos."""
    warnings = []
    if u1a > u1e:
        warnings.append("Units approved in sem 1 exceeds units enrolled.")
    if u2a > u2e:
        warnings.append("Units approved in sem 2 exceeds units enrolled.")
    if u1e == 0 and g1 > 0:
        warnings.append("Sem 1 grade > 0 but 0 units enrolled — check values.")
    if u2e == 0 and g2 > 0:
        warnings.append("Sem 2 grade > 0 but 0 units enrolled — check values.")
    return warnings

# ── Outcome configs (read classes from encoder dynamically, fallback to defaults) ──
DEFAULT_CONFIGS = {
    'Graduate':  {'emoji': '🎓', 'css': 'result-graduate', 'color': '#3dda6e',
                  'bar_color': '#3dda6e',
                  'desc': 'This student is on track to complete their degree successfully.'},
    'Dropout':   {'emoji': '⚠️', 'css': 'result-dropout',  'color': '#f26060',
                  'bar_color': '#f26060',
                  'desc': 'This student shows risk factors associated with leaving the programme.'},
    'Enrolled':  {'emoji': '📚', 'css': 'result-enrolled', 'color': '#f2c94c',
                  'bar_color': '#f2c94c',
                  'desc': 'This student is likely to remain enrolled and continue studying.'},
}
FALLBACK_CONFIG = {'emoji': '❓', 'css': 'result-unknown', 'color': '#8899bb',
                   'bar_color': '#8899bb', 'desc': 'Unknown outcome class.'}

def get_cfg(label):
    return DEFAULT_CONFIGS.get(label, FALLBACK_CONFIG)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<p class='sidebar-title'>Load models</p>", unsafe_allow_html=True)

    knn_file = st.file_uploader("KNN model (.pkl)", type="pkl", key="knn")
    svm_file = st.file_uploader("SVM model (.pkl)", type="pkl", key="svm")

    knn_bundle, svm_bundle = None, None

    if knn_file:
        bundle, err = safe_load_model(knn_file)
        if err:
            st.error(f"KNN load failed: {err}")
        else:
            knn_bundle = bundle
            st.markdown("""
            <div class='model-loaded'>
                <span class='dot-green'></span>
                <span class='model-loaded-name'>KNN model loaded</span>
            </div>""", unsafe_allow_html=True)

    if svm_file:
        bundle, err = safe_load_model(svm_file)
        if err:
            st.error(f"SVM load failed: {err}")
        else:
            svm_bundle = bundle
            st.markdown("""
            <div class='model-loaded'>
                <span class='dot-green'></span>
                <span class='model-loaded-name'>SVM model loaded</span>
            </div>""", unsafe_allow_html=True)

    # ── Model metadata panels ──
    if knn_bundle:
        st.markdown("<p class='sidebar-title' style='margin-top:1.5rem'>KNN metadata</p>",
                    unsafe_allow_html=True)
        meta = get_model_metadata(knn_bundle, 'KNN')
        rows = ''.join(
            f"<div class='meta-row'><span class='meta-key'>{k}</span>"
            f"<span class='meta-val'>{v}</span></div>"
            for k, v in meta.items()
        )
        st.markdown(f"<div class='meta-box'>{rows}</div>", unsafe_allow_html=True)

    if svm_bundle:
        st.markdown("<p class='sidebar-title' style='margin-top:1.5rem'>SVM metadata</p>",
                    unsafe_allow_html=True)
        meta = get_model_metadata(svm_bundle, 'SVM')
        rows = ''.join(
            f"<div class='meta-row'><span class='meta-key'>{k}</span>"
            f"<span class='meta-val'>{v}</span></div>"
            for k, v in meta.items()
        )
        st.markdown(f"<div class='meta-box'>{rows}</div>", unsafe_allow_html=True)

    if not knn_bundle and not svm_bundle:
        st.markdown(
            "<p style='color:#2a3050;font-size:0.78rem;margin-top:1rem;line-height:1.6'>"
            "Train models in Colab, download the .pkl files, then upload here.</p>",
            unsafe_allow_html=True
        )


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero-wrap'>
  <p class='hero-eyebrow'>ML · Classification · 3-class</p>
  <p class='hero-title'>Student Outcome Predictor</p>
  <p class='hero-sub'>Predict dropout risk using trained KNN &amp; SVM models — UCI dataset, 4,424 students</p>
</div>
""", unsafe_allow_html=True)

col_form, col_result = st.columns([1.15, 0.85], gap="large")

# ── Form ──────────────────────────────────────────────────────────────────────
with col_form:
    st.markdown("<p class='section-label'>Academic performance</p>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        units1_enrolled = st.number_input("Units enrolled — sem 1", 0, 10, 6, key="u1e")
        units1_approved = st.number_input("Units approved — sem 1", 0, 10, 5, key="u1a")
        grade1          = st.number_input("Grade sem 1 (0–20)",    0.0, 20.0, 13.0, step=0.1)
    with c2:
        units2_enrolled = st.number_input("Units enrolled — sem 2", 0, 10, 6, key="u2e")
        units2_approved = st.number_input("Units approved — sem 2", 0, 10, 5, key="u2a")
        grade2          = st.number_input("Grade sem 2 (0–20)",    0.0, 20.0, 13.0, step=0.1)

    st.markdown("<p class='section-label'>Student profile</p>", unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        age        = st.slider("Age at enrollment", 17, 60, 20)
        prev_grade = st.number_input("Previous qualification grade", 0.0, 200.0, 122.0, step=1.0)
    with c4:
        tuition_ok  = st.selectbox("Tuition fees up to date", [1, 0],
                                    format_func=lambda x: "Yes" if x else "No")
        scholarship = st.selectbox("Scholarship holder",      [0, 1],
                                    format_func=lambda x: "Yes" if x else "No")

    st.markdown("<p class='section-label'>Socioeconomic context</p>", unsafe_allow_html=True)
    c5, c6 = st.columns(2)
    with c5:
        unemployment = st.slider("Unemployment rate (%)", 5.0, 20.0, 10.8, step=0.1)
        gdp          = st.slider("GDP growth rate",      -5.0,  5.0,  1.0, step=0.1)
    with c6:
        inflation = st.slider("Inflation rate (%)", -2.0, 5.0, 1.4, step=0.1)
        debtor    = st.selectbox("Debtor", [0, 1], format_func=lambda x: "Yes" if x else "No")

    st.markdown("<p class='section-label'>Additional demographics</p>", unsafe_allow_html=True)
    c7, c8 = st.columns(2)
    with c7:
        gender       = st.selectbox("Gender", [1, 0], format_func=lambda x: "Male" if x else "Female")
        displaced    = st.selectbox("Displaced student", [0, 1], format_func=lambda x: "Yes" if x else "No")
    with c8:
        international = st.selectbox("International student", [0, 1], format_func=lambda x: "Yes" if x else "No")
        special_needs = st.selectbox("Educational special needs", [0, 1], format_func=lambda x: "Yes" if x else "No")

    # Input validation
    warnings = validate_inputs(units1_enrolled, units1_approved,
                               units2_enrolled, units2_approved,
                               grade1, grade2)
    if warnings:
        warn_items = ''.join(f"<p class='warn-item'>· {w}</p>" for w in warnings)
        st.markdown(f"""
        <div class='warn-box'>
            <p class='warn-title'>Input warnings</p>
            {warn_items}
        </div>""", unsafe_allow_html=True)

    predict_btn = st.button("Predict Outcome", use_container_width=True)


# ── Build full 36-feature row ─────────────────────────────────────────────────
def build_student_row():
    return pd.DataFrame([{
        'Marital Status':                              1,
        'Application mode':                           17,
        'Application order':                           1,
        'Course':                                    171,
        'Daytime/evening attendance':                  1,
        'Previous qualification':                      1,
        'Previous qualification (grade)':       prev_grade,
        'Nacionality':                                 1,
        "Mother's qualification":                     19,
        "Father's qualification":                     12,
        "Mother's occupation":                         5,
        "Father's occupation":                         5,
        'Admission grade':              prev_grade * 0.8,
        'Displaced':                             displaced,
        'Educational special needs':         special_needs,
        'Debtor':                                  debtor,
        'Tuition fees up to date':              tuition_ok,
        'Gender':                                  gender,
        'Scholarship holder':               scholarship,
        'Age at enrollment':                         age,
        'International':                   international,
        'Curricular units 1st sem (credited)':         0,
        'Curricular units 1st sem (enrolled)':  units1_enrolled,
        'Curricular units 1st sem (evaluations)': units1_enrolled,
        'Curricular units 1st sem (approved)':  units1_approved,
        'Curricular units 1st sem (grade)':          grade1,
        'Curricular units 1st sem (without evaluations)': 0,
        'Curricular units 2nd sem (credited)':         0,
        'Curricular units 2nd sem (enrolled)':  units2_enrolled,
        'Curricular units 2nd sem (evaluations)': units2_enrolled,
        'Curricular units 2nd sem (approved)':  units2_approved,
        'Curricular units 2nd sem (grade)':          grade2,
        'Curricular units 2nd sem (without evaluations)': 0,
        'Unemployment rate':                  unemployment,
        'Inflation rate':                       inflation,
        'GDP':                                       gdp,
    }])


# ── Probability bars ──────────────────────────────────────────────────────────
def render_proba_bars(proba_dict: dict, highlight: str):
    if not proba_dict:
        return
    st.markdown("<p class='section-label'>Confidence</p>", unsafe_allow_html=True)
    rows_html = ""
    for cls, prob in sorted(proba_dict.items(), key=lambda x: -x[1]):
        cfg = get_cfg(cls)
        pct = prob * 100
        bold = "font-weight:700;" if cls == highlight else ""
        rows_html += f"""
        <div class='prob-row'>
            <span class='prob-label'>{cls}</span>
            <div class='prob-bar-bg'>
                <div class='prob-bar-fill'
                     style='width:{pct:.1f}%;background:{cfg["bar_color"]};opacity:{"1" if cls==highlight else "0.35"}'></div>
            </div>
            <span class='prob-pct' style='color:{cfg["color"]};{bold}'>{pct:.1f}%</span>
        </div>"""
    st.markdown(rows_html, unsafe_allow_html=True)


# ── Result card ───────────────────────────────────────────────────────────────
def render_result_card(label: str, model_name: str, proba_dict):
    cfg = get_cfg(label)
    st.markdown(f"""
    <div class='result-card {cfg["css"]}'>
        <div class='result-emoji'>{cfg["emoji"]}</div>
        <div style='font-family:"DM Mono",monospace;font-size:0.65rem;color:#3a4060;
                    text-transform:uppercase;letter-spacing:0.15em;margin-bottom:0.4rem'>
            {model_name} prediction
        </div>
        <div class='result-label'>{label}</div>
        <div class='result-desc'>{cfg["desc"]}</div>
    </div>""", unsafe_allow_html=True)
    render_proba_bars(proba_dict, label)


# ── Results panel ─────────────────────────────────────────────────────────────
with col_result:
    if predict_btn:
        if not knn_bundle and not svm_bundle:
            st.warning("Upload at least one model from the sidebar first.")
        else:
            student_df = build_student_row()

            try:
                if knn_bundle and not svm_bundle:
                    pred, proba = predict(knn_bundle, student_df)
                    render_result_card(pred, "KNN", proba)

                elif svm_bundle and not knn_bundle:
                    pred, proba = predict(svm_bundle, student_df)
                    render_result_card(pred, "SVM", proba)

                else:
                    # Both models loaded — comparison view
                    pred_knn, proba_knn = predict(knn_bundle, student_df)
                    pred_svm, proba_svm = predict(svm_bundle, student_df)
                    agree = pred_knn == pred_svm

                    cfg_knn = get_cfg(pred_knn)
                    cfg_svm = get_cfg(pred_svm)

                    badge = ("<span class='badge badge-agree'>Models agree</span>"
                             if agree else
                             "<span class='badge badge-disagree'>Models disagree — review manually</span>")

                    st.markdown(f"""
                    <div class='compare-row'>
                      <div class='compare-card'>
                        <div class='compare-model'>KNN</div>
                        <div class='compare-pred' style='color:{cfg_knn["color"]}'>{pred_knn}</div>
                      </div>
                      <div class='compare-card'>
                        <div class='compare-model'>SVM</div>
                        <div class='compare-pred' style='color:{cfg_svm["color"]}'>{pred_svm}</div>
                      </div>
                    </div>
                    <div style='text-align:center'>{badge}</div>
                    """, unsafe_allow_html=True)

                    if agree:
                        cfg = get_cfg(pred_knn)
                        proba_combined = None
                        # Average probabilities if both support it
                        if proba_knn and proba_svm:
                            all_cls = set(proba_knn) | set(proba_svm)
                            proba_combined = {
                                c: (proba_knn.get(c, 0) + proba_svm.get(c, 0)) / 2
                                for c in all_cls
                            }
                        st.markdown(f"""
                        <div class='result-card {cfg["css"]}' style='margin-top:1rem'>
                            <div class='result-emoji'>{cfg["emoji"]}</div>
                            <div class='result-label'>{pred_knn}</div>
                            <div class='result-desc'>{cfg["desc"]}</div>
                        </div>""", unsafe_allow_html=True)
                        render_proba_bars(proba_combined or proba_knn, pred_knn)
                    else:
                        # Show individual confidence bars for each model
                        c_knn, c_svm = st.columns(2)
                        with c_knn:
                            st.markdown("<p class='section-label'>KNN confidence</p>",
                                        unsafe_allow_html=True)
                            if proba_knn:
                                for cls, p in sorted(proba_knn.items(), key=lambda x: -x[1]):
                                    cfg = get_cfg(cls)
                                    pct = p * 100
                                    st.markdown(f"""
                                    <div class='prob-row'>
                                        <span class='prob-label' style='min-width:60px'>{cls[:4]}</span>
                                        <div class='prob-bar-bg'>
                                            <div class='prob-bar-fill'
                                                 style='width:{pct:.1f}%;background:{cfg["bar_color"]};
                                                        opacity:{"1" if cls==pred_knn else "0.3"}'></div>
                                        </div>
                                        <span class='prob-pct' style='color:{cfg["color"]}'>{pct:.0f}%</span>
                                    </div>""", unsafe_allow_html=True)
                        with c_svm:
                            st.markdown("<p class='section-label'>SVM confidence</p>",
                                        unsafe_allow_html=True)
                            if proba_svm:
                                for cls, p in sorted(proba_svm.items(), key=lambda x: -x[1]):
                                    cfg = get_cfg(cls)
                                    pct = p * 100
                                    st.markdown(f"""
                                    <div class='prob-row'>
                                        <span class='prob-label' style='min-width:60px'>{cls[:4]}</span>
                                        <div class='prob-bar-bg'>
                                            <div class='prob-bar-fill'
                                                 style='width:{pct:.1f}%;background:{cfg["bar_color"]};
                                                        opacity:{"1" if cls==pred_svm else "0.3"}'></div>
                                        </div>
                                        <span class='prob-pct' style='color:{cfg["color"]}'>{pct:.0f}%</span>
                                    </div>""", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Prediction failed: {str(e)}\n\nMake sure the model was trained on the same feature set.")

            # ── Input summary ─────────────────────────────────────────
            st.markdown("<p class='section-label'>Input summary</p>", unsafe_allow_html=True)
            mc1, mc2, mc3, mc4 = st.columns(4)
            approve_rate1 = f"{int(units1_approved/units1_enrolled*100)}%" if units1_enrolled else "—"
            approve_rate2 = f"{int(units2_approved/units2_enrolled*100)}%" if units2_enrolled else "—"
            with mc1:
                st.markdown(f"""<div class='metric-box'>
                    <div class='metric-val'>{grade1:.1f}</div>
                    <div class='metric-lbl'>Sem 1 grade</div></div>""",
                    unsafe_allow_html=True)
            with mc2:
                st.markdown(f"""<div class='metric-box'>
                    <div class='metric-val'>{grade2:.1f}</div>
                    <div class='metric-lbl'>Sem 2 grade</div></div>""",
                    unsafe_allow_html=True)
            with mc3:
                st.markdown(f"""<div class='metric-box'>
                    <div class='metric-val'>{approve_rate1}</div>
                    <div class='metric-lbl'>Sem 1 pass rate</div></div>""",
                    unsafe_allow_html=True)
            with mc4:
                st.markdown(f"""<div class='metric-box'>
                    <div class='metric-val'>{approve_rate2}</div>
                    <div class='metric-lbl'>Sem 2 pass rate</div></div>""",
                    unsafe_allow_html=True)

            # ── Risk flag summary ──────────────────────────────────────
            flags = []
            if units1_approved < units1_enrolled // 2 and units1_enrolled > 0:
                flags.append("Low sem 1 pass rate")
            if units2_approved < units2_enrolled // 2 and units2_enrolled > 0:
                flags.append("Low sem 2 pass rate")
            if tuition_ok == 0:
                flags.append("Tuition fees not up to date")
            if debtor == 1:
                flags.append("Registered as debtor")
            if grade1 < 10 or grade2 < 10:
                flags.append("Grade below 10 in at least one semester")
            if unemployment > 15:
                flags.append("High unemployment rate in region")

            if flags:
                st.markdown("<p class='section-label'>Risk flags</p>", unsafe_allow_html=True)
                flag_items = ''.join(f"<p class='warn-item'>· {f}</p>" for f in flags)
                st.markdown(f"""
                <div class='warn-box'>
                    <p class='warn-title'>Factors associated with higher dropout risk</p>
                    {flag_items}
                </div>""", unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class='placeholder-box'>
            <div class='placeholder-icon'>🎓</div>
            <div class='placeholder-text'>
                Fill in the student details<br>and click <strong>Predict Outcome</strong>
            </div>
            <div style='color:#2a3050;font-size:0.75rem;margin-top:1rem;
                        font-family:"DM Mono",monospace;letter-spacing:0.05em'>
                Upload model pkl files in the sidebar first
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer note ───────────────────────────────────────────────────────────────
st.markdown("""
<div style='border-top:1px solid #111420;margin-top:3rem;padding-top:1rem;
            display:flex;justify-content:space-between;align-items:center'>
    <span style='font-family:"DM Mono",monospace;font-size:0.65rem;color:#2a3050'>
        UCI ML Repository · Predict Students Dropout and Academic Success · Dataset ID 697
    </span>
    <span style='font-family:"DM Mono",monospace;font-size:0.65rem;color:#2a3050'>
        KNN + SVM · sklearn · SMOTE
    </span>
</div>
""", unsafe_allow_html=True)
