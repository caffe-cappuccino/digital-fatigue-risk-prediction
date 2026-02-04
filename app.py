import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Digital Fatigue",
    page_icon="🌸",
    layout="centered"
)

# ================= SAFE MODEL LOAD =================
@st.cache_resource
def load_model():
    try:
        obj = joblib.load("model/fatigue_model.pkl")
        # hard validation
        if not hasattr(obj, "predict"):
            return None, "Loaded object is not a valid ML model."
        return obj, None
    except Exception as e:
        return None, str(e)

model, model_error = load_model()

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
p {
    color: #555;
}
.card {
    background: #ffffff;
    padding: 22px;
    border-radius: 24px;
    box-shadow: 0px 10px 24px rgba(0,0,0,0.08);
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
    "impact fatigue."
)

# ================= MODEL STATUS =================
if model_error:
    st.error("Model is not ready")
    st.write(model_error)
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
    continuous_usage = st.slider("Longest continuous usage (minutes)", 10, 300, 90, 10)
    eye_strain = st.select_slider("Eye strain level", [1, 2, 3, 4, 5], 3)
    task_switch = st.slider("Task switching frequency", 1, 50, 18)

predict = st.button("Analyze Fatigue")
st.markdown("</div>", unsafe_allow_html=True)

# ================= PREDICTION =================
if predict:

    # Build input safely
    input_df = pd.DataFrame([{
        "screen_time_hours": screen_time,
        "continuous_usage_minutes": continuous_usage,
        "night_usage_hours": night_usage,
        "breaks_per_day": 4,
        "sleep_hours": sleep,
        "eye_strain_level": eye_strain,
        "task_switching_rate": task_switch
    }])

    try:
        pred = model.predict(input_df)

        # universal safe cast
        fatigue = float(np.asarray(pred).reshape(-1)[0])

    except Exception as e:
        st.error("Prediction failed safely")
        st.write(e)
        st.stop()

    # ================= RESULT =================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Fatigue Score")
    st.metric("Predicted fatigue level", f"{fatigue:.1f} / 100")

    if fatigue < 35:
        st.success("Low fatigue detected. Habits look balanced.")
    elif fatigue < 65:
        st.warning("Moderate fatigue detected. Minor adjustments may help.")
    else:
        st.error("High fatigue detected. Rest is strongly recommended.")

    st.markdown("</div>", unsafe_allow_html=True)

    # ================= SMALL 3D GRAPH =================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Key Contributing Factors")

    factors = ["Screen", "Night", "Low Sleep", "Eye Strain", "Switching"]
    contrib = np.array([
        screen_time / 16,
        night_usage / 8,
        (10 - sleep) / 10,
        eye_strain / 5,
        task_switch / 50
    ]) * 100

    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection="3d")
    xs = np.arange(len(factors))

    ax.bar3d(xs, np.zeros(len(xs)), np.zeros(len(xs)),
             0.5, 0.5, contrib,
             color="#cdb4db", shade=True)

    ax.set_xticks(xs)
    ax.set_xticklabels(factors, rotation=20)
    ax.set_zlabel("Impact (%)")
    ax.set_title("Estimated Contribution")

    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

    # ================= ADVICE =================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Suggested Adjustments")

    tips = []
    if screen_time > 8:
        tips.append("Reduce total daily screen exposure.")
    if night_usage > 2:
        tips.append("Limit screen usage close to bedtime.")
    if sleep < 6:
        tips.append("Aim for longer, consistent sleep.")
    if eye_strain >= 4:
        tips.append("Take regular visual breaks during screen use.")
    if task_switch > 30:
        tips.append("Reduce frequent task switching.")

    if tips:
        for t in tips:
            st.write("•", t)
    else:
        st.write("No major issues detected. Maintain current habits.")

    st.markdown("</div>", unsafe_allow_html=True)

# ================= FOOTER =================
st.caption("Minimal • Calm • Aesthetic ✨")
