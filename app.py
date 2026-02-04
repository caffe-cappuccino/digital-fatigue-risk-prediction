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

# ================= LOAD MODEL (UNCHANGED) =================
model = joblib.load("model/fatigue_model.pkl")

# ================= SOFT PASTEL UI =================
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

# ================= PREDICTION (UNCHANGED) =================
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
        level = "Low"
        color = "#b5ead7"
    elif fatigue < 65:
        level = "Moderate"
        color = "#fff1ac"
    else:
        level = "High"
        color = "#ffb7b2"

    st.write(f"**Fatigue category:** {level}")
    st.markdown("</div>", unsafe_allow_html=True)

    # ================= LOLLIPOP CHART =================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("What contributes the most")

    factors = [
        "Screen time",
        "Night usage",
        "Low sleep",
        "Eye strain",
        "Task switching"
    ]

    values = np.array([
        screen_time,
        night_usage,
        10 - sleep,
        eye_strain,
        task_switch
    ])

    y_pos = np.arange(len(factors))

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hlines(y=y_pos, xmin=0, xmax=values, color="#cdb4db", linewidth=3)
    ax.plot(values, y_pos, "o", color="#6c63ff", markersize=8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(factors)
    ax.set_xlabel("Relative impact")
    ax.set_title("Fatigue contributors")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

    # ================= ADVICE CARD =================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Suggestions")

    tips = []

    if screen_time > 8:
        tips.append("Reduce overall daily screen exposure.")
    if night_usage > 2:
        tips.append("Avoid screens close to bedtime.")
    if sleep < 6:
        tips.append("Aim for longer, consistent sleep.")
    if eye_strain >= 4:
        tips.append("Take frequent eye breaks (20–20–20 rule).")
    if task_switch > 30:
        tips.append("Reduce frequent context switching.")

    if tips:
        for t in tips:
            st.write("•", t)
    else:
        st.write("Your habits look balanced. Keep maintaining them.")

    st.markdown("</div>", unsafe_allow_html=True)

# ================= FOOTER =================
st.caption("Minimal • Calm • Aesthetic ✨")
