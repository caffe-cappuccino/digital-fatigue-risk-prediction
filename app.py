import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import time

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Digital Fatigue Intelligence",
    page_icon="🧠",
    layout="wide"
)

# ================= GLOBAL STYLE =================
st.markdown("""
<style>
@keyframes fadeIn {
  from {opacity: 0; transform: translateY(10px);}
  to {opacity: 1; transform: translateY(0);}
}

body {
    background: linear-gradient(135deg, #eef2ff, #fdf2f8);
    color: #0f172a;
}

h1 {
    font-weight: 800;
    letter-spacing: -1px;
}

.subtitle {
    color: #475569;
    margin-top: -10px;
    font-size: 16px;
}

.glass {
    background: rgba(255,255,255,0.75);
    backdrop-filter: blur(10px);
    padding: 30px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.4);
    box-shadow: 0 20px 40px rgba(0,0,0,0.05);
    animation: fadeIn 0.6s ease;
    margin-bottom: 28px;
}

.metric {
    font-size: 42px;
    font-weight: 800;
}

.metric-label {
    color: #64748b;
    font-size: 13px;
}

.stButton > button {
    background: linear-gradient(90deg, #6366f1, #ec4899);
    color: white;
    border-radius: 14px;
    height: 3.2em;
    font-size: 16px;
    font-weight: 600;
    border: none;
}

.stButton > button:hover {
    transform: scale(1.02);
}

.divider {
    height: 1px;
    background: #e5e7eb;
    margin: 22px 0;
}
</style>
""", unsafe_allow_html=True)

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    return pd.read_csv("data/digital_fatigue.csv")

df = load_data()

FEATURES = [
    "screen_time_hours",
    "continuous_usage_minutes",
    "night_usage_hours",
    "breaks_per_day",
    "sleep_hours",
    "eye_strain_level",
    "task_switching_rate"
]

# ================= TRAIN MODEL =================
@st.cache_resource
def train_model(data):
    X = data[FEATURES]
    raw = (
        X["screen_time_hours"] * 6
        + X["continuous_usage_minutes"] * 0.08
        + X["night_usage_hours"] * 8
        - X["breaks_per_day"] * 4
        - X["sleep_hours"] * 7
        + X["eye_strain_level"] * 10
        + X["task_switching_rate"] * 1.5
    )
    y = (raw - raw.min()) / (raw.max() - raw.min()) * 100
    Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        random_state=42
    )
    model.fit(Xtr, ytr)
    return model, y.mean()

model, avg_fatigue = train_model(df)

# ================= HEADER =================
st.markdown("<h1>🧠 Digital Fatigue Intelligence</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Smart behavioral insights to understand your digital burnout risk</div>", unsafe_allow_html=True)
st.write("")

# ================= INPUT =================
st.markdown("<div class='glass'>", unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)

with c1:
    screen = st.slider("📱 Screen time (hours)", 1.0, 16.0, 6.0, 0.5)
    night = st.slider("🌙 Late-night usage (hours)", 0.0, 8.0, 1.5, 0.5)

with c2:
    cont = st.slider("⏱ Longest continuous usage (min)", 10, 300, 90, 10)
    switch = st.slider("🔀 Task switching frequency", 1, 50, 18)

with c3:
    sleep = st.slider("😴 Sleep duration (hours)", 3.0, 10.0, 7.0, 0.5)
    eye = st.select_slider("👁 Eye strain level", [1,2,3,4,5], 3)

run = st.button("🚀 Run Smart Assessment")
st.markdown("</div>", unsafe_allow_html=True)

# ================= RUN PIPELINE =================
if run:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    with st.spinner("Analyzing behavioral patterns…"):
        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress.progress(i + 1)

    user_df = pd.DataFrame([[screen, cont, night, 4, sleep, eye, switch]], columns=FEATURES)
    fatigue = float(model.predict(user_df)[0])
    fatigue = np.clip(fatigue, 0, 100)

    color = "#22c55e" if fatigue < 40 else "#f59e0b" if fatigue < 70 else "#ef4444"
    emoji = "😌" if fatigue < 40 else "😵" if fatigue < 70 else "🔥"

    c1, c2, c3 = st.columns(3)
    c1.markdown(f"<div class='metric' style='color:{color}'>{fatigue:.1f}</div><div class='metric-label'>Your fatigue score {emoji}</div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric'>{avg_fatigue:.1f}</div><div class='metric-label'>Population average</div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric'>{fatigue-avg_fatigue:+.1f}</div><div class='metric-label'>Difference</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ================= ANALYSIS =================
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("📊 Key Fatigue Drivers")

    factors = ["Screen time", "Night usage", "Low sleep", "Eye strain", "Task switching"]
    values = [screen, night, 10-sleep, eye, switch]

    fig = go.Figure(go.Bar(
        x=values,
        y=factors,
        orientation="h",
        marker_color=color
    ))
    fig.update_layout(
        height=420,
        margin=dict(l=40, r=20, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ================= RECOMMENDATIONS =================
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("💡 Smart Recommendations")

    advice = {
        "Screen time": "Introduce offline recovery windows and cap non-essential usage.",
        "Night usage": "Implement a digital sunset one hour before sleep.",
        "Low sleep": "Optimize sleep hygiene to restore cognitive resilience.",
        "Eye strain": "Apply the 20–20–20 rule consistently.",
        "Task switching": "Batch tasks to reduce mental context switching."
    }

    for f, v in sorted(zip(factors, values), key=lambda x: x[1], reverse=True):
        st.markdown(f"**{f}**")
        st.write(advice[f])
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ================= FOOTER =================
st.caption("Bright • Intelligent • Human-Centered Analytics")
