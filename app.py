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

# ================= LOAD MODEL =================
model = joblib.load("model/fatigue_model.pkl")

# ================= CLEAN PREMIUM UI =================
st.markdown("""
<style>

/* Global background */
body {
    background-color: #f7f8fc;
}

/* Headings */
h1 {
    color: #2d2f7f;
    font-weight: 700;
}
h2, h3 {
    color: #2f2f2f;
}

/* Card */
.card {
    background: white;
    padding: 28px;
    border-radius: 18px;
    box-shadow: 0px 12px 30px rgba(0,0,0,0.08);
    margin-bottom: 36px;
}

/* Button */
.stButton > button {
    background: linear-gradient(90deg, #6c63ff, #8f88ff);
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-size: 16px;
    font-weight: 600;
    border: none;
}

/* Progress bar */
.progress-wrapper {
    background: #ecebff;
    border-radius: 10px;
    height: 12px;
    width: 100%;
    overflow: hidden;
    margin-top: 10px;
}
.progress-fill {
    height: 100%;
    border-radius: 10px;
    background: linear-gradient(90deg, #6c63ff, #8f88ff);
}

/* Section spacing */
.section {
    margin-bottom: 40px;
}

</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.title("Digital Fatigue Monitor")
st.write(
    "A premium analytics interface to understand how your digital habits "
    "impact cognitive and physical fatigue."
)

# ================= INPUT =================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("Daily Usage Overview")

c1, c2 = st.columns(2)

with c1:
    screen_time = st.slider("Screen time (hours)", 1.0, 16.0, 6.0, 0.5)
    night_usage = st.slider("Late-night usage (hours)", 0.0, 8.0, 1.5, 0.5)
    sleep = st.slider("Sleep duration (hours)", 3.0, 10.0, 7.0, 0.5)

with c2:
    continuous_usage = st.slider("Longest continuous usage (minutes)", 10, 300, 90, 10)
    eye_strain = st.select_slider("Eye strain level", [1,2,3,4,5], 3)
    task_switch = st.slider("Task switching frequency", 1, 50, 18)

predict = st.button("Analyze Fatigue")
st.markdown("</div>", unsafe_allow_html=True)

# ================= PREDICTION =================
if predict:
    input_df = pd.DataFrame([[
        screen_time,
        continuous_usage,
        night_usage,
        4,
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
    fatigue_pct = min(max(fatigue, 0), 100)

    # ================= RESULT =================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Fatigue Score")
    st.write(f"**{fatigue_pct:.1f} / 100**")

    st.markdown(
        f"""
        <div class="progress-wrapper">
            <div class="progress-fill" style="width:{fatigue_pct}%;"></div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if fatigue < 35:
        st.success("Low fatigue detected. Your habits appear well balanced.")
    elif fatigue < 65:
        st.warning("Moderate fatigue detected. Small adjustments may help.")
    else:
        st.error("High fatigue detected. Recovery is strongly recommended.")

    st.markdown("</div>", unsafe_allow_html=True)

    # ================= LOLLIPOP CHART =================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Key Contributors")

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

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.hlines(y=y_pos, xmin=0, xmax=values, color="#c7c5ff", linewidth=4)
    ax.plot(values, y_pos, "o", color="#6c63ff", markersize=8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(factors)
    ax.set_xlabel("Relative impact")
    ax.set_title("Fatigue contribution by factor")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

    # ================= ADVICE =================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Actionable Suggestions")

    tips = []
    if screen_time > 8:
        tips.append("Reduce total daily screen exposure.")
    if night_usage > 2:
        tips.append("Avoid screen use close to bedtime.")
    if sleep < 6:
        tips.append("Increase sleep duration for recovery.")
    if eye_strain >= 4:
        tips.append("Follow the 20–20–20 eye care rule.")
    if task_switch > 30:
        tips.append("Limit frequent task switching to reduce cognitive load.")

    if tips:
        for t in tips:
            st.write("•", t)
    else:
        st.write("No major risk factors detected. Keep maintaining your routine.")

    st.markdown("</div>", unsafe_allow_html=True)

# ================= FOOTER =================
st.caption("Clean • Premium • Insightful")
