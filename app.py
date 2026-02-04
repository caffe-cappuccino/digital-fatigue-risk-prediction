import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Digital Fatigue Predictor",
    page_icon="🧠",
    layout="centered"
)

# ================= SAFE MODEL LOAD =================
@st.cache_resource
def load_model():
    try:
        return joblib.load("model/fatigue_model.pkl"), None
    except Exception as e:
        return None, str(e)

model, model_error = load_model()

# ================= BASIC CLEAN STYLE =================
st.markdown("""
<style>
body { background-color: #f7f8fc; }
h1, h2, h3 { color: #2c3e50; }
.card {
    background: white;
    padding: 18px;
    border-radius: 14px;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}
.stButton > button {
    background-color: #6C63FF;
    color: white;
    border-radius: 10px;
    width: 100%;
    height: 3em;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.title("🧠 Digital Fatigue Prediction")
st.write(
    "A clean, explainable ML system that predicts **how much digital fatigue may occur** "
    "and visually explains **why**."
)

# ================= INPUTS =================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("Usage Inputs")

c1, c2 = st.columns(2)

with c1:
    screen_time = st.slider("Screen Time (hrs/day)", 1.0, 16.0, 6.0, 0.5)
    night_usage = st.slider("Night Usage (hrs)", 0.0, 8.0, 1.5, 0.5)
    sleep = st.slider("Sleep (hrs)", 3.0, 10.0, 7.0, 0.5)

with c2:
    continuous_usage = st.slider("Continuous Usage (mins)", 10, 300, 90, 10)
    eye_strain = st.select_slider("Eye Strain", [1,2,3,4,5], 3)
    task_switch = st.slider("Task Switching/hour", 1, 50, 18)

predict = st.button("Predict Fatigue")
st.markdown("</div>", unsafe_allow_html=True)

# ================= PREDICTION =================
if model_error:
    st.error("Model failed to load")
    st.code(model_error)

elif predict:
    input_df = pd.DataFrame([[
        screen_time,
        continuous_usage,
        night_usage,
        4,  # fixed breaks (kept stable)
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

    fatigue = float(model.predict(input_df)[0])

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.metric("Fatigue Severity Score", f"{fatigue:.1f} / 100")
    st.markdown("</div>", unsafe_allow_html=True)

    # ================= 3D GRAPH 1: CONTRIBUTION =================
    st.subheader("3D Contribution View (Compact)")

    factors = ["Screen", "Night", "Low Sleep", "Eye", "Switching"]
    contrib = np.array([
        screen_time / 16,
        night_usage / 8,
        (10 - sleep) / 10,
        eye_strain / 5,
        task_switch / 50
    ]) * 100

    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(111, projection='3d')

    xs = np.arange(len(factors))
    ys = np.zeros(len(factors))
    zs = np.zeros(len(factors))

    dx = np.ones(len(factors)) * 0.5
    dy = np.ones(len(factors)) * 0.5
    dz = contrib

    ax.bar3d(xs, ys, zs, dx, dy, dz, shade=True)
    ax.set_xticks(xs)
    ax.set_xticklabels(factors, rotation=20)
    ax.set_zlabel("Contribution (%)")
    ax.set_title("Factor Contribution (3D)")

    st.pyplot(fig)

    # ================= 3D GRAPH 2: FATIGUE LANDSCAPE =================
    st.subheader("Fatigue Landscape (User Position)")

    x = np.linspace(2, 14, 20)
    y = np.linspace(4, 9, 20)
    X, Y = np.meshgrid(x, y)
    Z = (X * 5) + ((10 - Y) * 8)

    fig2 = plt.figure(figsize=(6,4))
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.plot_surface(X, Y, Z, alpha=0.4)
    ax2.scatter(screen_time, sleep, fatigue, color='red', s=50)

    ax2.set_xlabel("Screen Time")
    ax2.set_ylabel("Sleep")
    ax2.set_zlabel("Fatigue")
    ax2.set_title("User Position in Fatigue Space")

    st.pyplot(fig2)

    # ================= ADVICE =================
    st.subheader("Quick Advice")
    if fatigue > 65:
        st.warning("High fatigue risk: reduce screen time and improve sleep.")
    elif fatigue > 35:
        st.info("Moderate fatigue: balance usage and take regular breaks.")
    else:
        st.success("Low fatigue: current habits are healthy.")
