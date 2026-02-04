import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Digital Fatigue Predictor",
    page_icon="🧠",
    layout="centered"
)

# ================= LOAD MODEL =================
model = joblib.load("model/fatigue_model.pkl")

# ================= CUSTOM CSS =================
st.markdown("""
<style>
h1, h2, h3 { color: #2c3e50; }
.stButton > button {
    background-color: #6C63FF;
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-size: 16px;
}
.card {
    padding: 20px;
    border-radius: 14px;
    background-color: white;
    box-shadow: 0px 4px 14px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# ================= TITLE =================
st.title("🧠 Digital Fatigue Prediction System")
st.write(
    "An **explainable machine-learning decision-support system** that predicts "
    "**how much digital fatigue may occur and why**."
)

st.markdown("---")

# ================= INPUTS =================
st.subheader("📥 Usage Pattern Inputs")

col1, col2 = st.columns(2)

with col1:
    screen_time = st.slider("📱 Screen Time (hrs/day)", 1.0, 16.0, 6.5, 0.5)
    night_usage = st.slider("🌙 Night Usage (hrs)", 0.0, 8.0, 1.5, 0.5)
    sleep = st.slider("😴 Sleep (hrs)", 3.0, 10.0, 7.0, 0.5)

with col2:
    continuous_usage = st.slider("⏱ Continuous Usage (mins)", 10, 300, 90, 10)
    breaks = st.slider("☕ Breaks/day", 0, 15, 4)
    eye_strain = st.select_slider("👁 Eye Strain", [1,2,3,4,5], 3)

task_switch = st.slider("🔁 Task Switching/hour", 1, 50, 18)

# ================= PREDICTION =================
st.markdown("---")
st.subheader("🔍 Prediction, Breakdown & Advice")

if st.button("Predict Digital Fatigue"):

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
        risk, emoji = "Low", "🟢"
    elif fatigue_score < 65:
        risk, emoji = "Medium", "🟡"
    else:
        risk, emoji = "High", "🔴"

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.metric("Predicted Fatigue Severity", f"{fatigue_score:.1f} / 100")
    st.write(f"{emoji} **Risk Level:** {risk}")
    st.markdown("</div>", unsafe_allow_html=True)

    # ================= CONTRIBUTION TABLE =================
    st.subheader("📊 Factor-wise Fatigue Contribution")

    contrib_df = pd.DataFrame({
        "Factor": [
            "Screen Time",
            "Continuous Usage",
            "Night Usage",
            "Low Sleep",
            "Eye Strain",
            "Task Switching"
        ],
        "User Value": [
            screen_time,
            continuous_usage,
            night_usage,
            sleep,
            eye_strain,
            task_switch
        ],
        "Estimated Contribution (%)": [
            (screen_time / 16) * 100,
            (continuous_usage / 300) * 100,
            (night_usage / 8) * 100,
            ((10 - sleep) / 10) * 100,
            (eye_strain / 5) * 100,
            (task_switch / 50) * 100
        ]
    })

    st.dataframe(
        contrib_df.style.format({"Estimated Contribution (%)": "{:.1f}"})
    )

    # ================= CONTRIBUTION GRAPH =================
    st.subheader("🎨 Contribution Visualization")

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(
        contrib_df["Factor"],
        contrib_df["Estimated Contribution (%)"]
    )

    for bar, val in zip(bars, contrib_df["Estimated Contribution (%)"]):
        bar.set_color(plt.cm.viridis(val / 100))

    ax.set_xlim(0, 100)
    ax.set_xlabel("Contribution (%)")
    ax.set_title("Estimated Contribution of Each Factor to Digital Fatigue")

    st.pyplot(fig)

    # ================= ADVICE =================
    st.subheader("💡 Personalized Recommendations")

    advice = []

    if screen_time > 8:
        advice.append("📱 Reduce total daily screen time below 8 hours.")
    if night_usage > 2:
        advice.append("🌙 Limit late-night screen usage.")
    if sleep < 6:
        advice.append("😴 Increase sleep duration to at least 7 hours.")
    if continuous_usage > 120:
        advice.append("⏱ Take breaks every 60 minutes.")
    if eye_strain >= 4:
        advice.append("👁 Follow the 20-20-20 eye rule.")
    if task_switch > 30:
        advice.append("🔁 Reduce frequent task switching.")

    if advice:
        for tip in advice:
            st.write("- ", tip)
    else:
        st.success("✅ Your digital usage pattern is well balanced.")

# ================= FOOTER =================
st.markdown("---")
st.caption(
    "⚠️ This system provides **decision-support insights** only and is not a medical diagnostic tool."
)
