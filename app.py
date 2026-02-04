import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Digital Fatigue Intelligence Dashboard",
    page_icon="🧠",
    layout="wide"
)

# ================= LOAD MODEL =================
model = joblib.load("model/fatigue_model.pkl")

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
    "A **machine-learning powered decision system** that quantifies digital fatigue, "
    "explains contributing factors, and recommends corrective actions."
)

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
    eye_strain = st.select_slider("👁 Eye Strain", [1,2,3,4,5], 3)

task_switch = st.slider("🔁 Task Switching Rate (per hour)", 1, 50, 18)

predict = st.button("🚀 Analyze Digital Fatigue")

st.markdown("</div>", unsafe_allow_html=True)

# ================= PREDICTION =================
if predict:
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

    fatigue_score = model.predict(input_df)[0]

    if fatigue_score < 35:
        risk = "LOW"
        color = "#4CAF50"
    elif fatigue_score < 65:
        risk = "MODERATE"
        color = "#FFC107"
    else:
        risk = "HIGH"
        color = "#FF5252"

    # ================= GAUGE VISUAL =================
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.subheader("🎯 Fatigue Severity Gauge")

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh([0], [fatigue_score], color=color)
    ax.barh([0], [100 - fatigue_score], left=fatigue_score, color="#2a2f5a")
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel("Fatigue Severity (0–100)")
    ax.set_title(f"Predicted Fatigue Level: {risk}", color="white")
    ax.spines[:].set_visible(False)

    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

    # ================= CONTRIBUTION RADAR =================
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
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_yticklabels([])
    ax.set_title("Relative Fatigue Contribution by Factor", color="white")

    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

    # ================= CONTRIBUTION TABLE =================
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.subheader("📊 Quantified Contribution Breakdown")

    contrib_df = pd.DataFrame({
        "Factor": labels,
        "Relative Impact (%)": [round(v * 100, 1) for v in values[:-1]]
    })

    st.dataframe(
        contrib_df.style
        .background_gradient(cmap="cool")
        .format({"Relative Impact (%)": "{:.1f}"})
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # ================= ADVICE =================
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.subheader("💡 Actionable Recommendations")

    advice = []

    if screen_time > 8:
        advice.append("📱 Reduce daily screen time below 8 hours.")
    if night_usage > 2:
        advice.append("🌙 Avoid screen exposure before bedtime.")
    if sleep < 6:
        advice.append("😴 Increase sleep duration to improve recovery.")
    if continuous_usage > 120:
        advice.append("⏱ Take short breaks every 60 minutes.")
    if eye_strain >= 4:
        advice.append("👁 Follow the 20-20-20 eye rule.")
    if task_switch > 30:
        advice.append("🔁 Reduce task switching to lower cognitive load.")

    if advice:
        for tip in advice:
            st.write("•", tip)
    else:
        st.success("✅ Your digital behavior is well balanced.")

    st.markdown("</div>", unsafe_allow_html=True)

# ================= FOOTER =================
st.markdown("---")
st.caption(
    "⚠️ This system provides **decision-support insights** only. "
    "It is not a medical diagnostic tool."
)
