import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Digital Fatigue",
    page_icon="🌸",
    layout="centered"
)

# ================= LOAD MODEL (SIMPLE, AS BEFORE) =================
model = joblib.load("model/fatigue_model.pkl")

# ================= PASTEL UI (ONLY COSMETIC) =================
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #fdfbfb, #ebedee);
}
h1 {
    color: #6c63ff;
    font-weight: 700;
}
h2, h3 {
    color: #444;
}
.card {
    background: white;
    padding: 22px;
    border-radius: 22px;
    box-shadow: 0px 8px 20px rgba(0,0,0,0.08);
    margin-bottom: 24px;
}
.stButton > button {
    background: linear-gradient(90deg, #a18cd1, #fbc2eb);
    color: #222;
    border-radius: 30px;
    height: 3em;
    width: 100%;
    font-size: 16px;
    font-weight: 600;
    border: none;
}
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.title("🌸 Digital Fatigue Monitor")
st.write(
    "A calm, minimal interface to understand how daily digital habits "
    "affect fatigue levels."
)

# ================= INPUT CARD =================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("Daily Usage Overview")

c1, c2 = st.columns(2)

with c1:
    screen_time = st.slider("Screen time (hours)", 1.0, 16.0, 6.0, 0.5)
    night_usage = st.slider("Late-night usage (hours)", 0.0, 8.0, 1.5, 0.5)
    sleep = st.slider("Sleep duration (hours)", 3.0, 10.0, 7.0, 0.5)

with c2:
    continuous_usage = st.slider("Longest continuous usage (minutes)", 10, 300, 90, 10)
    eye_strain = st.select_slider("Eye strain level", [1, 2, 3, 4, 5], 3)
    task_switch = st.slider("Task switching frequency", 1, 50, 18)

predict = st.button("Analyze Fatigue")

st.markdown("</div>", unsafe_allow_html=True)

# ================= PREDICTION (UNCHANGED LOGIC) =================
if predict:
    input_df = pd.DataFrame([[
        screen_time,
        continuous_usage,
        night_usage,
        4,  # breaks_per_day
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

    fatigue = model.predict(input_df)[0]

    # ================= RESULT =================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Fatigue Score")
    st.metric("Predicted fatigue level", f"{fatigue:.1f} / 100")

    if fatigue < 35:
        st.success("Low fatigue detected. Habits look balanced.")
    elif fatigue < 65:
        st.warning("Moderate fatigue detected. Some adjustments may help.")
    else:
        st.error("High fatigue detected. Rest is recommended.")

    st.markdown("</div>", unsafe_allow_html=True)

    # ================= SIMPLE, SMALL GRAPH =================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Key Contributors")

    factors = ["Screen Time", "Night Usage", "Low Sleep", "Eye Strain", "Task Switching"]
    values = [
        screen_time,
        night_usage,
        10 - sleep,
        eye_strain,
        task_switch
    ]

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.barh(factors, values, color="#cdb4db")
    ax.set_xlabel("Relative impact")

    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

# ================= FOOTER =================
st.caption("Minimal • Calm • Aesthetic ✨")
