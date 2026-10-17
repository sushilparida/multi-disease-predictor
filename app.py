"""Multi-Disease Prediction System  -  Streamlit app.
Run:  streamlit run app.py
"""
import json
import os
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st

from config import DISEASES

st.set_page_config(page_title="Multi-Disease Prediction System", page_icon="🏥", layout="wide")


@st.cache_resource
def load_model(name):
    path = f"models/{name.replace(' ', '_').lower()}.joblib"
    return joblib.load(path) if os.path.exists(path) else None


def risk_level(p):
    if p < 0.30:
        return "LOW", "green"
    if p < 0.60:
        return "MODERATE", "orange"
    return "HIGH", "red"


def make_report(patient, disease, prob, level, top_feats, advice):
    """Plain-text report (always works). PDF is produced if fpdf2 is installed."""
    lines = [
        "MULTI-DISEASE PREDICTION SYSTEM - HEALTH SUMMARY",
        f"Generated: {datetime.now():%d %b %Y, %H:%M}",
        "",
        f"Patient: {patient or 'N/A'}",
        f"Screening for: {disease}",
        f"Predicted risk: {prob*100:.1f}%  ({level})",
        "",
        "Key factors in the model:",
    ] + [f" - {k}: {v*100:.1f}% importance" for k, v in top_feats] + [
        "",
        "Recommendations:",
    ] + [f" - {a}" for a in advice] + [
        "",
        "DISCLAIMER: This is a machine-learning screening aid, not a medical diagnosis.",
        "Please consult a qualified doctor.",
    ]
    return "\n".join(lines)


def to_pdf_bytes(text):
    try:
        from fpdf import FPDF
    except ImportError:
        return None
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    for line in text.split("\n"):
        safe = line.encode("latin-1", "replace").decode("latin-1")
        pdf.multi_cell(0, 7, safe if safe else " ")
    return bytes(pdf.output())


# ---------------- Sidebar ----------------
st.sidebar.title("🏥 Multi-Disease Predictor")
page = st.sidebar.radio("Go to", ["Predict", "Model Performance", "About"])

# ---------------- Predict ----------------
if page == "Predict":
    st.title("AI-Powered Multi-Disease Prediction System")
    st.caption("Enter the patient's details to estimate disease risk. Screening aid only - not a diagnosis.")

    disease = st.selectbox("Select disease", list(DISEASES.keys()))
    cfg = DISEASES[disease]
    bundle = load_model(disease)
    if bundle is None:
        st.error("Model not found. Run `python train.py` first (and put the CSVs in the data/ folder).")
        st.stop()

    patient = st.text_input("Patient name (optional)")
    values = {}
    cols = st.columns(3)
    for i, (feat, (label, lo, hi, default, step)) in enumerate(cfg["features"].items()):
        with cols[i % 3]:
            if isinstance(step, int) and isinstance(lo, int):
                values[feat] = st.number_input(label, int(lo), int(hi), int(default), int(step))
            else:
                values[feat] = st.number_input(label, float(lo), float(hi), float(default), float(step))

    if st.button("🔍 Predict", type="primary"):
        X = pd.DataFrame([values])[bundle["features"]]
        prob = float(bundle["model"].predict_proba(X)[0][1])
        level, color = risk_level(prob)

        st.subheader("Result")
        c1, c2 = st.columns(2)
        c1.metric("Risk probability", f"{prob*100:.1f}%")
        c2.markdown(f"### Risk level: :{color}[{level}]")
        st.progress(min(max(prob, 0.0), 1.0))

        imp = sorted(bundle["importance"].items(), key=lambda kv: kv[1], reverse=True)
        top = imp[:5]
        st.write("**Which inputs matter most to this model**")
        st.bar_chart(pd.DataFrame(top, columns=["Feature", "Importance"]).set_index("Feature"))

        advice = cfg["advice_high"] if prob >= 0.5 else cfg["advice_low"]
        st.write("**Recommendations**")
        for a in advice:
            st.write(f"- {a}")

        report = make_report(patient, disease, prob, level, top, advice)
        pdf = to_pdf_bytes(report)
        if pdf:
            st.download_button("📄 Download PDF report", pdf, "health_report.pdf", "application/pdf")
        else:
            st.download_button("📄 Download report (txt)", report, "health_report.txt")
        st.warning("This tool does not replace a doctor. Please seek professional medical advice.")

# ---------------- Performance ----------------
elif page == "Model Performance":
    st.title("Model Performance")
    if not os.path.exists("models/metrics.json"):
        st.info("Run `python train.py` to generate metrics.")
    else:
        metrics = json.load(open("models/metrics.json"))
        for disease, m in metrics.items():
            st.subheader(f"{disease}  (best: {m['best_model']}, {m['rows']} records)")
            st.dataframe(pd.DataFrame(m["results"]).T)

# ---------------- About ----------------
else:
    st.title("About this project")
    st.markdown(
        """
**Problem:** People often cannot quickly check their risk for common diseases.
**Solution:** A web app that predicts risk for **Diabetes, Heart Disease and Breast Cancer**
using separate machine-learning models for each disease.

**Tech stack:** Python, scikit-learn (Logistic Regression + Random Forest, best model chosen by ROC-AUC),
pandas, Streamlit, joblib, fpdf2.

**Datasets:** Pima Indians Diabetes (Kaggle/UCI), Heart Disease (UCI/Kaggle),
Breast Cancer Wisconsin (scikit-learn built-in).

**Features:** multi-disease prediction, risk level, feature-importance view,
lifestyle recommendations, downloadable PDF report, model performance page.
        """
    )
