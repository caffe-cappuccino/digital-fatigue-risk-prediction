import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Digital Fatigue ✨",
    page_icon="🌸",
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

# ================= GEN-Z PASTEL STYLE =================
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #fdfbfb, #ebedee);
}
h1 {
    color: #5f5cff;
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
    padding: 20px;
    border-radius: 22px;
    box-shadow: 0px 8px 20px rgba(0,0,0,0.08);
    margin-bottom: 22px;
}
.stButton > button {
    background: linear-gradient(90deg, #ff9a9e, #fad0c4);
    color: #222;
    border-radius: 30px;
    height: 3em;
    width: 100%;
    font-size: 16px;
    font-weight: 600;
    border: none;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #fbc2eb, #a6c1ee);
}
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.title("🌸 Digital Fatigue Check")
st.write(
    "hey bestie 👋  
    let’s see how tired your **digital life** is making you 💻🧠  
    no judgement, just vibes ✨"
)

# ================= MODEL ERROR =================
if model_error:
    st.error("oops 😭 something went wrong loading the brain")
    st.code(model_error)
    st.stop()

# ================= INPUT CARD =================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🧩 tell me about your day")

c1, c2 = st.columns(2)

with c1:
    screen_time = st.slider("📱 screen time (hrs)", 1.0, 16.0, 6.0, 0.5)
    night_usage = st.slider("🌙 late night scrolling (hrs)", 0.0, 8.0, 1.5, 0.5)
    sleep = st.slider("😴 sleep (hrs)", 3.0, 10.0, 7.0, 0.5)

with c2:
    continuous_usage = st.slider("⏱ binge usage (mins)", 10, 300, 90, 10)
    eye_strain = st.select_slider("👀 eye strain", [1,2,3,4,5], 3)
    task_switch = st.slider("🔁 app switching", 1, 50, 18)

predict = st.button("✨ check my fatigue")

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

    raw_pred = model.predict(input_df)
    fatigue = float(np.array(raw_pred).reshape(-1)[0])

    # ================= RESULT CARD =================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🧠 your fatigue score")

    st.metric("fatigue level", f"{fatigue:.1f} / 100")

    if fatigue < 35:
        st.success("💖 you’re doing okay bestie, keep it up!")
    elif fatigue < 65:
        st.warning("💛 a little tired — maybe slow down today")
    else:
        st.error("💔 you’re exhausted, pls rest 😭")

    st.markdown("</div>", unsafe_allow_html=True)

    # ================= CUTE 3D CONTRIBUTION GRAPH =================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🎨 what’s draining you the most")

    factors = ["screen", "night", "sleep", "eyes", "switching"]
    contrib = np.array([
        screen_time / 16,
        night_usage / 8,
        (10 - sleep) / 10,
        eye_strain / 5,
        task_switch / 50
    ]) * 100

    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection='3d')
    xs = np.arange(len(factors))

    ax.bar3d(xs, np.zeros(len(xs)), np.zeros(len(xs)),
             0.5, 0.5, contrib, color="#ffb6c1", shade=True)

    ax.set_xticks(xs)
    ax.set_xticklabels(factors, rotation=20)
    ax.set_zlabel("impact %")
    ax.set_title("digital energy drain")

    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

    # ================= ADVICE =================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🌱 gentle reminders")

    tips = []
    if screen_time > 8:
        tips.append("📱 log off a little earlier today")
    if night_usage > 2:
        tips.append("🌙 doomscrolling isn’t self-care bestie")
    if sleep < 6:
        tips.append("😴 pls sleep, everything feels worse when tired")
    if eye_strain >= 4:
        tips.append("👀 blink. hydrate. look outside.")
    if task_switch > 30:
        tips.append("🧠 do one thing at a time, it’s okay")

    if tips:
        for t in tips:
            st.write("•", t)
    else:
        st.write("✨ honestly? you’re balanced rn, proud of you")

    st.markdown("</div>", unsafe_allow_html=True)

# ================= FOOTER =================
st.caption("made with 💖, vibes & a tiny bit of ML ✨")
