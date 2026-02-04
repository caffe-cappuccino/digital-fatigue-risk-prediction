import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Digital Fatigue Intelligence Dashboard",
    page_icon="🧠",
    layout="wide"
)

# ================= SAFE MODEL LOADER =================
@st.cache_resource
def load_model_safe(path):
    try:
        return joblib.load(path), None
    except Exception as e:
        return None, str(e)

model, model_error = load_model_safe("model/fatigue_model.pkl")

# ================= GLOBAL STYLE =================
st.markdown("""
<style>
body {
    background-color: #0f1222;
}
h1, h2, h3, h4 {
    color: #ffffff;
}
p, span, div {
    color: #dcdcdc;
}
.section {
    background: linear-gradient(145deg, #1a1f3c, #11142a);
    padding: 25px;
    border-radius: 18px;
    margin-bottom: 25px;
    box-shadow: 0px 10px 30px rgba(0,0,0,0.4);
}
.stButton > button {
    background: linear-gradient(90deg, #7f7cff, #5a54e8);
    color: white;
    border-radius: 16px;
    height: 3.2em;
    width: 100%;
    font-size: 18px;
    border: none;
}
.stProgress > div > div {
    background: linear-gradient(90deg, #7f7cff, #ff6584);
}
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.title("🧠 Digital Fatigue Intelligence Dashboard")
st.markdown(
    "### Predict • Explain • Act  \n"
    "An **ML-powered decision-support system** for analyzing digital fatigue."
)

# ================= MODEL STATUS =================
if model_error:
    st.error("🚨 Model failed to load")
    st.code(model_error)
    st.info(
        "The UI is loaded safely, but predictions are disabled because the "
        "model file is corrupted or incomplete.\n\n"
        "**No changes to train.py were made.**\n"
        "Once a valid model file exists, predictions will automatically work."
    )

st.markdown("---")

# ================= INPUT PANEL =================
st.markdown("<div class='section'>", unsafe_allow_html=True)
st.subheader("📥 Behavioral Usage Inputs")

c1, c2, c3 = st.columns(3)

with c1:
    screen_time = st.slider("📱 Screen Time (hrs/day)", 1.0, 16.0, 6.5, 0.5)
    night_usage = st.slider("🌙 Night Usage (hrs)", 0.0, 8.0, 1.5, 0.5)

with c2:
    continuous_usage = st.slider("⏱ Continuous Usage (mins)", 10, 300, 90, 10)
    breaks = st.slider("☕ Breaks / Day", 0, 15, 4)

with c3:
    sleep = st.slider("😴 Sleep (hrs)", 3.0, 10.0, 7.0, 0.5)
    eye_strain = st.select_slider("👁 Eye Strain", [1, 2, 3, 4, 5], 3)

task_switch = st.slider("🔁 Task Switching Rate (per hour)", 1, 50, 18)

predict = st.button("🚀 Analyze Digital Fatigue", disabled=model is None)

st.markdown("</div>", unsafe_allow_html=True)

# ================= PREDICTION =================
if predict and model is not None:
    input_df = pd.DataFrame([[
        screen_time,
        continuous_usage,
        night_usage,
        breaks,
        sleep,
        eye_strain,
        task_switch
    ]], columns=[
        "screen_time_hours",
        "continuous_usage_minutes",
        "night_usage_hours",
        "breaks_per_day",
        "sleep_hours",
        "eye_strain_level",
        "task_switching_rate"
    ])

    fatigue_score = float(model.predict(input_df)[0])

    if fatigue_score < 35:
        risk, color = "LOW", "#4CAF50"
    elif fatigue_score < 65:
        risk, color = "MODERATE", "#FFC107"
    else:
        risk, color = "HIGH", "#FF5252"

    # ================= GAUGE =================
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.subheader("🎯 Fatigue Severity Gauge")

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh([0], [fatigue_score], color=color)
    ax.barh([0], [100 - fatigue_score], left=fatigue_score, color="#2a2f5a")
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_title(f"Predicted Fatigue Level: {risk}", color="white")
    ax.spines[:].set_visible(False)

    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

    # ================= RADAR =================
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.subheader("🧬 Factor Contribution Radar")

    labels = [
        "Screen Time",
        "Continuous Usage",
        "Night Usage",
        "Low Sleep",
        "Eye Strain",
        "Task Switching"
    ]

    values = [
        screen_time / 16,
        continuous_usage / 300,
        night_usage / 8,
        (10 - sleep) / 10,
        eye_strain / 5,
        task_switch / 50
    ]

    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(values))

    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, color="#7f7cff", linewidth=3)
    ax.fill(angles, values, color="#7f7cff", alpha=0.25)
    ax.set_yticklabels([])
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title("Relative Fatigue Contribution", color="white")

    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

# ================= FOOTER =================
st.markdown("---")
st.caption(
    "⚠️ This system provides **decision-support insights** only and is not a medical diagnostic tool."
)
