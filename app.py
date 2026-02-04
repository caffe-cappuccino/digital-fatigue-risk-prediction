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
body {
    background-color: #f6f8fc;
}
h1, h2, h3 {
    color: #2c3e50;
}
.section-card {
    padding: 22px;
    border-radius: 16px;
    background-color: white;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}
.stButton > button {
    background: linear-gradient(90deg, #6C63FF, #5A54E8);
    color: white;
    border-radius: 14px;
    height: 3em;
    width: 100%;
    font-size: 16px;
}
.stProgress > div > div {
    background-color: #6C63FF;
}
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.title("🧠 Digital Fatigue Prediction Dashboard")
st.write(
    "An **explainable ML-based decision-support system** that predicts "
    "**how much digital fatigue may occur, why it occurs, and what to do next**."
)

st.markdown("---")

# ================= TABS =================
tab1, tab2, tab3 = st.tabs(["📥 Inputs", "📊 Analysis", "💡 Advice"])

# ================= TAB 1: INPUTS =================
with tab1:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Usage Pattern Inputs")

    col1, col2 = st.columns(2)

    with col1:
        screen_time = st.slider("📱 Screen Time (hrs/day)", 1.0, 16.0, 6.5, 0.5)
        night_usage = st.slider("🌙 Night Usage (hrs)", 0.0, 8.0, 1.5, 0.5)
        sleep = st.slider("😴 Sleep Duration (hrs)", 3.0, 10.0, 7.0, 0.5)

    with col2:
        continuous_usage = st.slider("⏱ Continuous Usage (mins)", 10, 300, 90, 10)
        breaks = st.slider("☕ Breaks per Day", 0, 15, 4)
        eye_strain = st.select_slider("👁 Eye Strain Level", [1,2,3,4,5], 3)

    task_switch = st.slider("🔁 Task Switching Rate (per hour)", 1, 50, 18)

    st.markdown("</div>", unsafe_allow_html=True)

    predict_clicked = st.button("🚀 Predict Digital Fatigue")

# ================= SHARED PREDICTION LOGIC =================
if predict_clicked:
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
        risk, color = "Low", "🟢"
    elif fatigue_score < 65:
        risk, color = "Medium", "🟡"
    else:
        risk, color = "High", "🔴"

# ================= TAB 2: ANALYSIS =================
with tab2:
    if not predict_clicked:
        st.info("ℹ️ Enter inputs and click **Predict Digital Fatigue** first.")
    else:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("Fatigue Severity Analysis")

        st.metric(
            label="Predicted Fatigue Severity Score",
            value=f"{fatigue_score:.1f} / 100",
            delta=risk
        )

        st.progress(int(fatigue_score))

        st.write(f"{color} **Risk Level:** {risk}")
        st.markdown("</div>", unsafe_allow_html=True)

        # -------- Contribution Table --------
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("Factor-wise Contribution Breakdown")

        contrib_df = pd.DataFrame({
            "Factor": [
                "Screen Time",
                "Continuous Usage",
                "Night Usage",
                "Low Sleep",
                "Eye Strain",
                "Task Switching"
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
            contrib_df.style
            .format({"Estimated Contribution (%)": "{:.1f}"})
            .background_gradient(cmap="Purples")
        )

        # -------- Contribution Graph --------
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.barh(
            contrib_df["Factor"],
            contrib_df["Estimated Contribution (%)"]
        )

        for bar, val in zip(bars, contrib_df["Estimated Contribution (%)"]):
            bar.set_color(plt.cm.plasma(val / 100))

        ax.set_xlim(0, 100)
        ax.set_xlabel("Contribution (%)")
        ax.set_title("Relative Contribution to Digital Fatigue")

        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

# ================= TAB 3: ADVICE =================
with tab3:
    if not predict_clicked:
        st.info("ℹ️ Prediction required to generate advice.")
    else:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("Personalized Actionable Advice")

        advice = []

        if screen_time > 8:
            advice.append("📱 Reduce total screen time to under 8 hours/day.")
        if night_usage > 2:
            advice.append("🌙 Avoid screen usage at least 1 hour before sleep.")
        if sleep < 6:
            advice.append("😴 Increase sleep duration to improve recovery.")
        if continuous_usage > 120:
            advice.append("⏱ Take short breaks every 60 minutes.")
        if eye_strain >= 4:
            advice.append("👁 Use the 20-20-20 eye relaxation rule.")
        if task_switch > 30:
            advice.append("🔁 Reduce frequent task switching to lower mental load.")

        if advice:
            for tip in advice:
                st.write("•", tip)
        else:
            st.success("✅ Your digital habits are well balanced. Keep it up!")

        st.markdown("</div>", unsafe_allow_html=True)

# ================= FOOTER =================
st.markdown("---")
st.caption(
    "⚠️ This system is a **decision-support tool**, not a medical diagnostic system. "
    "Predictions are based on behavioral usage analytics."
)
