#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 11:06:04 2026

@author: ekoderman
"""
import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
import base64

mcp = joblib.load("mondrian_cp.pkl")

model = mcp["model"]
q_dict = mcp["q_dict"]
alpha_per_class = mcp["alpha_per_class"]
class_mapping = mcp["class_mapping"]

def mondrian_cp_predict(proba, classes, q_dict, alpha_per_class):
    prediction_set = []

    for idx, cls in enumerate(classes):
        score = 1.0 - proba[idx] # nonconformity score
        q = q_dict[cls] # threshold for this patient
        alpha = alpha_per_class.get(cls, None)

        if alpha is None:
            continue

        if score <= q:
            prediction_set.append(cls)

    return prediction_set


#### APP ####
st.title("Functional Status Prediction for Contrast-Enhancing Glioma")

st.markdown(
            """
These predictions estimate the **one-year postoperative functional status** after the full
primary treatment of patients with radiologically confirmed contrast-enhancing gliomas.

Functional status is measured by **Karnofsky Performance Status (KPS)**:
- **Mortality:** KPS = 0
- **Functional dependence:** KPS = 10–60
- **Functional independence:** KPS = 70–100

Reference/DOI: *(add when available)*
"""
        )

# --- Disclaimer (prominent, always visible) ---
st.warning(
    "**Research use only.** This tool is intended for research purposes only "
    "and must not be used for clinical decision-making or patient care as it currently is.",
    icon=None
)

st.subheader("Patient characteristics")

# --- User inputs ---
age = st.slider(
    "Age at resection (years)",
    min_value=18, max_value=100, value=80
)
kps = st.slider(
    "Preoperative KPS",
    min_value=10, max_value=100, value=80, step=10
)
enh_vol = st.number_input(
    "Enhancing tumor component volume (mL)",
    min_value=0.005, value=20.0
)

# --- Build X exactly as in training ---
X = pd.DataFrame(
    [[age, kps, enh_vol]],
    columns=[
        "Age at resection",
        "Preoperative KPS",
        "Enhancing component volume"
    ]
)

class_mapping = {
    0: "Mortality",
    1: "Functional dependence",
    2: "Functional independence"
}

# --- Predict ---
if st.button("Predict"):
    proba = model.predict_proba(X)[0]
    pred_set = mondrian_cp_predict(
        proba,
        model.classes_,
        q_dict,
        alpha_per_class
    )

    alpha_val = list(alpha_per_class.values())[0]
    confidence_pct = 1 - alpha_val
    pred_labels = ', '.join([class_mapping[c] for c in pred_set])

    # Show prediction set
    st.subheader("Prediction set")
    st.info(
        f"With surgical resection, the patient's long-term functional status will fall within "
        f"the predicted set: **{pred_labels}.**"
    )

    # Pie chart — only classes in prediction set
    labels = [class_mapping[c] for c in pred_set]
    values = [round(proba[c] * 100, 2) for c in pred_set]

    df = pd.DataFrame({"Class": labels, "Probability": values})
    fig = px.pie(df, names="Class", values="Probability", hole=0.2)
    fig.update_traces(textinfo='label+value', insidetextorientation='radial')
    st.plotly_chart(fig)

    with st.popover("Want to see the full list of probabilities?"):
        st.subheader("Full set of probabilities")
        for c, p in zip(model.classes_, proba):
            st.write(f"**{class_mapping[c]}**: {p * 100:.2f}%")
            st.progress(p)

    # --- MCP (revealed after prediction) ---
    with st.expander("How to use prediction sets?", expanded=True):
        st.video("static/mcp_video_audio_final.mp4")
