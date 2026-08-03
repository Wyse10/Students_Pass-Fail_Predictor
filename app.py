import os

import joblib
import pandas as pd
import streamlit as st


BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "models", "pass_fail_voting_pipeline.pkl")

PROFILES = {
    "Balanced": {"wassce_grade": 74.0, "first_year_cgpa": 2.8, "gender": "F"},
    "High performer": {"wassce_grade": 91.0, "first_year_cgpa": 3.6, "gender": "M"},
    "Borderline": {"wassce_grade": 66.5, "first_year_cgpa": 2.0, "gender": "F"},
    "At risk": {"wassce_grade": 58.0, "first_year_cgpa": 1.5, "gender": "M"},
}


st.set_page_config(
    page_title="Student Pass/Fail Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(56, 189, 248, 0.18), transparent 28%),
            radial-gradient(circle at right 15%, rgba(16, 185, 129, 0.16), transparent 24%),
            linear-gradient(180deg, #f8fbff 0%, #eef4fb 100%);
    }
    .hero-card, .info-card, .result-card, .mini-card {
        background: rgba(255, 255, 255, 0.82);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(148, 163, 184, 0.24);
        border-radius: 22px;
        box-shadow: 0 18px 50px rgba(15, 23, 42, 0.08);
        padding: 1.2rem 1.35rem;
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        line-height: 1.05;
        margin-bottom: 0.35rem;
        color: #0f172a;
    }
    .hero-subtitle {
        color: #475569;
        font-size: 1.02rem;
        line-height: 1.55;
    }
    .eyebrow {
        display: inline-block;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        background: rgba(15, 23, 42, 0.06);
        color: #0f172a;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .result-pass {
        border-left: 6px solid #10b981;
    }
    .result-fail {
        border-left: 6px solid #ef4444;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.45rem 0.8rem;
        border-radius: 999px;
        font-weight: 700;
        letter-spacing: 0.01em;
    }
    .status-pass {
        background: rgba(16, 185, 129, 0.12);
        color: #047857;
    }
    .status-fail {
        background: rgba(239, 68, 68, 0.12);
        color: #b91c1c;
    }
    .small-note {
        color: #64748b;
        font-size: 0.9rem;
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.98), rgba(30, 41, 59, 0.97));
        color: white;
    }
    div[data-testid="stSidebar"] * {
        color: white;
    }
    div[data-testid="stSidebar"] .stSelectbox label,
    div[data-testid="stSidebar"] .stRadio label,
    div[data-testid="stSidebar"] .stCheckbox label,
    div[data-testid="stSidebar"] .stSlider label {
        color: rgba(255, 255, 255, 0.88) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def initialize_state():
    defaults = PROFILES["Balanced"].copy()
    defaults["auto_predict"] = True
    defaults["profile_name"] = "Balanced"
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def build_input_frame():
    return pd.DataFrame(
        {
            "Wassce Grade": [st.session_state["wassce_grade"]],
            "First_Year_CGPA": [st.session_state["first_year_cgpa"]],
            "Gender": [st.session_state["gender"]],
        }
    )


def predict_student():
    input_df = build_input_frame()
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    prob_fail = float(probabilities[0])
    prob_pass = float(probabilities[1])
    confidence = max(prob_pass, prob_fail)
    return prediction, prob_pass, prob_fail, confidence


try:
    model = load_model()
except Exception as exc:
    st.error(f"Could not load the model from the models folder.\n\n{exc}")
    st.stop()


initialize_state()


with st.sidebar:
    st.markdown("<span class='eyebrow'>Quick Controls</span>", unsafe_allow_html=True)
    st.markdown("### Student profile")
    st.caption("Use a preset to explore the model quickly, or keep the sliders and inputs live.")

    profile_name = st.selectbox(
        "Preset",
        options=list(PROFILES.keys()),
        index=list(PROFILES.keys()).index(st.session_state["profile_name"]),
    )

    if st.button("Load preset", use_container_width=True):
        st.session_state.update(PROFILES[profile_name])
        st.session_state["profile_name"] = profile_name
        st.rerun()

    if st.button("Reset to balanced", use_container_width=True):
        st.session_state.update(PROFILES["Balanced"])
        st.session_state["profile_name"] = "Balanced"
        st.rerun()

    st.toggle("Auto predict on change", key="auto_predict")

    st.markdown("---")
    st.markdown("**Model inputs**")
    st.write("• WASSCE Grade")
    st.write("• First_Year_CGPA")
    st.write("• Gender")
    st.markdown("<p class='small-note'>Prediction threshold: the pipeline predicts Pass/Fail using the trained ensemble.</p>", unsafe_allow_html=True)


st.markdown("<div class='hero-card'>", unsafe_allow_html=True)
st.markdown("<span class='eyebrow'>Student Pass/Fail Predictor</span>", unsafe_allow_html=True)
st.markdown("<div class='hero-title'>See pass/fail risk in a cleaner, more interactive dashboard.</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='hero-subtitle'>Adjust the inputs, load a preset profile, or let the result update automatically. The app shows the prediction, confidence, and a compact explanation of the student profile.</div>",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)


top1, top2, top3 = st.columns(3)
with top1:
    st.markdown("<div class='mini-card'><strong>Model</strong><br/>Voting Classifier ensemble</div>", unsafe_allow_html=True)
with top2:
    st.markdown("<div class='mini-card'><strong>Inputs</strong><br/>3 features used by the pipeline</div>", unsafe_allow_html=True)
with top3:
    st.markdown("<div class='mini-card'><strong>Outcome</strong><br/>Pass or Fail with probabilities</div>", unsafe_allow_html=True)


tab_predict, tab_model = st.tabs(["Predict", "Model notes"])

with tab_predict:
    left, right = st.columns([1.15, 0.95], gap="large")

    with left:
        st.markdown("<div class='info-card'>", unsafe_allow_html=True)
        st.subheader("Student details")
        input_col1, input_col2 = st.columns(2)

        with input_col1:
            st.number_input(
                "WASSCE Grade",
                min_value=1.0,
                max_value=100.0,
                value=float(st.session_state["wassce_grade"]),
                step=0.5,
                help="Student's WASSCE entry score",
                key="wassce_grade",
            )
            st.radio(
                "Gender",
                options=["F", "M"],
                format_func=lambda value: "Female" if value == "F" else "Male",
                horizontal=True,
                key="gender",
            )

        with input_col2:
            st.number_input(
                "First Year CGPA",
                min_value=0.0,
                max_value=4.0,
                value=float(st.session_state["first_year_cgpa"]),
                step=0.01,
                help="GPA at the end of Year 1 (scale: 0.0 - 4.0)",
                key="first_year_cgpa",
            )

        st.markdown(
            "<p class='small-note'>Tip: use the preset cards in the sidebar to preview common student profiles.</p>",
            unsafe_allow_html=True,
        )

        predict_requested = st.button("Predict now", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='info-card'>", unsafe_allow_html=True)
        st.subheader("Prediction result")

        should_predict = st.session_state["auto_predict"] or predict_requested
        if should_predict:
            prediction, prob_pass, prob_fail, confidence = predict_student()
            is_pass = int(prediction) == 1

            status_class = "result-pass" if is_pass else "result-fail"
            pill_class = "status-pass" if is_pass else "status-fail"
            status_text = "PASS" if is_pass else "FAIL"
            explanation = (
                "The model expects this student to pass." if is_pass else "The model flags this student as being at risk of failing."
            )

            st.markdown(
                f"""
                <div class="result-card {status_class}">
                    <div class="status-pill {pill_class}">{status_text}</div>
                    <h3 style="margin: 0.7rem 0 0.35rem 0;">{status_text}</h3>
                    <p style="margin: 0; color: #475569;">{explanation}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            prob_left, prob_right = st.columns(2)
            with prob_left:
                st.metric("Pass probability", f"{prob_pass * 100:.1f}%")
                st.progress(prob_pass)
            with prob_right:
                st.metric("Fail probability", f"{prob_fail * 100:.1f}%")
                st.progress(prob_fail)

            if confidence >= 0.8:
                confidence_label = "High confidence"
                confidence_color = "#10b981"
            elif confidence >= 0.6:
                confidence_label = "Moderate confidence"
                confidence_color = "#f59e0b"
            else:
                confidence_label = "Low confidence"
                confidence_color = "#ef4444"

            st.markdown(
                f"<p style='margin-top: 0.6rem; color: {confidence_color}; font-weight: 700;'>{confidence_label} ({confidence * 100:.1f}%)</p>",
                unsafe_allow_html=True,
            )

            summary = pd.DataFrame(
                {
                    "Feature": ["WASSCE Grade", "First Year CGPA", "Gender"],
                    "Value": [
                        f"{st.session_state['wassce_grade']:.1f}",
                        f"{st.session_state['first_year_cgpa']:.2f}",
                        "Female" if st.session_state["gender"] == "F" else "Male",
                    ],
                }
            )

            st.markdown("### Input snapshot")
            st.dataframe(summary, hide_index=True, use_container_width=True)
        else:
            st.info("Turn on auto predict or click Predict now to see the result.")

        st.markdown("</div>", unsafe_allow_html=True)


with tab_model:
    info_left, info_right = st.columns([1, 1], gap="large")

    with info_left:
        st.markdown("<div class='info-card'>", unsafe_allow_html=True)
        st.subheader("What the model uses")
        st.write("The pipeline predicts using these exact inputs:")
        st.write("• WASSCE Grade")
        st.write("• First_Year_CGPA")
        st.write("• Gender")
        st.markdown("<p class='small-note'>The model is a trained voting ensemble, and preprocessing happens inside the saved pipeline.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with info_right:
        st.markdown("<div class='info-card'>", unsafe_allow_html=True)
        st.subheader("Prediction guidance")
        st.write("• Higher CGPA generally pushes the prediction toward Pass.")
        st.write("• Lower CGPA or weaker entry scores may increase fail risk.")
        st.write("• Gender is included because it was part of the trained dataset.")
        st.markdown("<p class='small-note'>Use the presets to explore how confidence changes across different student profiles.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


st.markdown("---")
st.caption("Model: Voting Classifier (Random Forest · SVC · Gradient Boosting)  |  Prediction threshold: CGPA ≥ 2.0")