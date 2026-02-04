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

# ================= SAFE MODEL LOAD =================
model = None
model_error = None

try:
    model = joblib.load("model/fatigue_model.pkl")
except Exception as e:
    model_error = str(e)

# ================= GLOBAL CSS (FIXES WHITE BARS) =================
st.markdown("""
<style>

/* 🌑 Background */
body {
    background: linear-gradient(135deg, #0f0f14, #151522);
}

/* 🔥 REMOVE STREAMLIT DEFAULT WHITE BLOCKS */
section.main > div {
    background: transparent !important;
}

.block-container {
    padding-top: 2.5rem !important;
    padding-bottom: 2.5rem !important;
}

/* Remove empty Streamlit containers */
div[data-testid="stMarkdownContainer"]:empty {
    display: none !important;
}

/* Headings */
h1 {
    color: #f3f1ff;
    font-weight: 700;
}

h2, h3 {
    color: #e6e4ff;
}

/* 🌸 Cards (ONLY white elements now) */
.card {
    background: #ffffff;
    padding: 26px;
    border-radius: 26px;
    box-shadow: 0px 14px 30px rgba(0,0,0,0.35);
    margin-bottom: 42px;
}

/* 🌸 Button */
.stButton > button {
    background: linear-gradient(90deg, #a18cd1, #fbc2eb);
    color: #222;
    border-radius: 30px;
    height: 3.2em;
    width: 100%;
    font-size: 16px;
    font-weight: 600;
    border: none;
    transition: all 0.25s ease;
}

.stButton > button:hover {
    transform: scale(1.04);
    box-shadow: 0px 10px 24px rgba(161, 140, 209, 0.55);
}

/* 🌸 Progress bar */
.progress-wrapper {
    background: #ecebff;
    border-radius: 14px;
    height: 14px;
    width: 100%;
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    border-radius: 14px;
    background: linear-gradient(90deg, #a18cd1, #fbc2eb);
}

</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.title("🌸 Digital Fatigue Monitor")
st.write(
    "A calm, minimal interface to understand how daily digital habits "
    "affect fatigue levels."
)

# ================= MODEL ERROR =================
if model_error:
    st.error("Model could not be loaded")
    st.code(model_error)
    st.stop()

# ================= INPUT CARD =================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("Daily Usage Overview")

c1, c2 = st.columns(2)

with c1:
    screen_time = st.slider("Screen time (hours)", 1.0, 16.0, 6.0, 0.5)
    night_usage = st.slider("Late-night usage (hours)", 0.0, 8.0, 1.5, 0.5)
    sleep = st.slider("Sleep duration (hours)", 3.0, 10.0, 7.0, 0.5)

with c2:
    continuous_usage = st.slider(
        "Longest continuous usage (minutes)", 10, 300, 90, 10
    )
    eye_strain = st.select_slider(
        "Eye strain level", [1, 2, 3, 4, 5], 3
    )
    task_switch = st.slider(
        "Task switching frequency", 1, 50, 18
    )

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

    # ================= RESULT CARD =================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Fatigue Level")
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
        st.success("Low fatigue detected. Your habits look balanced.")
    elif fatigue < 65:
        st.warning("Moderate fatigue detected. Some adjustments may help.")
    else:
        st.error("High fatigue detected. Rest and recovery are recommended.")

    st.markdown("</div>", unsafe_allow_html=True)

    # ================= LOLLIPOP CHART =================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Main Contributors")

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
        tips.append("Limit screen usage close to bedtime.")
    if sleep < 6:
        tips.append("Aim for longer and more consistent sleep.")
    if eye_strain >= 4:
        tips.append("Take regular eye breaks (20–20–20 rule).")
    if task_switch > 30:
        tips.append("Reduce frequent context switching.")

    if tips:
        for tip in tips:
            st.write("•", tip)
    else:
        st.write("Your habits look well balanced. Keep maintaining them.")

    st.markdown("</div>", unsafe_allow_html=True)

# ================= FOOTER =================
st.caption("Minimal • Calm • Aesthetic ✨")
