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

/* ---------- ANIMATIONS ---------- */
@keyframes float {
  0% { transform: translateY(0px); }
  50% { transform: translateY(-10px); }
  100% { transform: translateY(0px); }
}

@keyframes fadeUp {
  from { opacity: 0; transform: translateY(20px); }
  to { opacity: 1; transform: translateY(0); }
}

/* ---------- BACKGROUND ---------- */
body {
    background:
        radial-gradient(circle at 15% 20%, #e0e7ff, transparent 40%),
        radial-gradient(circle at 85% 30%, #fce7f3, transparent 45%),
        radial-gradient(circle at 50% 85%, #ecfeff, transparent 50%),
        linear-gradient(135deg, #f8fafc, #f1f5f9);
    color: #020617;
}

/* ---------- HEADINGS ---------- */
h1 {
    font-weight: 900;
    letter-spacing: -1.4px;
}

.subtitle {
    color: #475569;
    margin-top: -12px;
    font-size: 15px;
}

/* ---------- GLASS CARD ---------- */
.glass {
    background: rgba(255, 255, 255, 0.72);
    backdrop-filter: blur(14px);
    padding: 36px;
    border-radius: 24px;
    border: 1px solid rgba(255,255,255,0.4);
    box-shadow:
        0 30px 60px rgba(15,23,42,0.08),
        inset 0 1px 0 rgba(255,255,255,0.6);
    animation: fadeUp 0.6s ease;
    margin-bottom: 36px;
}

/* ---------- CONTROL PANEL ---------- */
.control-panel {
    background: linear-gradient(145deg, rgba(255,255,255,0.88), rgba(255,255,255,0.68));
    border-radius: 28px;
    padding: 38px 42px;
    box-shadow:
        0 45px 90px rgba(15,23,42,0.1),
        inset 0 1px 0 rgba(255,255,255,0.75);
    border: 1px solid rgba(255,255,255,0.45);
    margin-bottom: 42px;
}

.panel-title {
    font-weight: 800;
    letter-spacing: -0.4px;
    margin-bottom: 6px;
}

.panel-subtitle {
    color: #64748b;
    font-size: 14px;
    margin-bottom: 30px;
}

/* ---------- INPUT GROUP ---------- */
.input-group {
    background: rgba(255,255,255,0.7);
    border-radius: 22px;
    padding: 26px 28px;
    box-shadow: 0 18px 38px rgba(15,23,42,0.06);
    transition: all 0.25s ease;
}

.input-group:hover {
    transform: translateY(-4px);
    box-shadow: 0 28px 60px rgba(15,23,42,0.12);
}

.group-title {
    font-weight: 700;
    font-size: 14px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 18px;
    color: #020617;
}

/* ---------- METRICS ---------- */
.metric {
    font-size: 46px;
    font-weight: 900;
    letter-spacing: -1px;
}

.metric-label {
    color: #64748b;
    font-size: 13px;
}

/* ---------- BUTTON ---------- */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #db2777);
    color: white;
    border-radius: 18px;
    height: 3.5em;
    font-size: 16px;
    font-weight: 700;
    border: none;
    box-shadow: 0 16px 36px rgba(79,70,229,0.4);
    transition: all 0.25s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 24px 50px rgba(79,70,229,0.55);
}

/* ---------- ADVICE CARDS ---------- */
.advice-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.9), rgba(255,255,255,0.7));
    border-radius: 20px;
    padding: 24px 28px;
    box-shadow: 0 20px 45px rgba(15,23,42,0.1);
    border-left: 5px solid #6366f1;
    animation: float 7s ease-in-out infinite;
    margin-bottom: 20px;
}

.advice-title {
    font-weight: 700;
    margin-bottom: 6px;
    color: #020617;
}

.advice-text {
    color: #475569;
    font-size: 14px;
}

/* ---------- DIVIDER ---------- */
.divider {
    height: 1px;
    background: linear-gradient(to right, transparent, #e5e7eb, transparent);
    margin: 28px 0;
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
st.markdown("<h1>Digital Fatigue Intelligence</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Behavioral analytics engine for cognitive load and burnout risk</div>", unsafe_allow_html=True)
st.write("")

# ================= INPUT CONTROL PANEL =================
st.markdown("""
<div class="control-panel">
    <div class="panel-title">Behavioral Parameters</div>
    <div class="panel-subtitle">
        Tune system-level variables to simulate digital fatigue impact
    </div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("<div class='input-group'><div class='group-title'>Exposure</div>", unsafe_allow_html=True)
    screen = st.slider("Screen Time (hours)", 1.0, 16.0, 6.0, 0.5)
    night = st.slider("Late-Night Usage (hours)", 0.0, 8.0, 1.5, 0.5)
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown("<div class='input-group'><div class='group-title'>Cognitive Load</div>", unsafe_allow_html=True)
    cont = st.slider("Longest Continuous Session (minutes)", 10, 300, 90, 10)
    switch = st.slider("Task Switching Rate", 1, 50, 18)
    st.markdown("</div>", unsafe_allow_html=True)

with c3:
    st.markdown("<div class='input-group'><div class='group-title'>Recovery</div>", unsafe_allow_html=True)
    sleep = st.slider("Sleep Duration (hours)", 3.0, 10.0, 7.0, 0.5)
    eye = st.select_slider("Eye Strain Severity", [1,2,3,4,5], 3)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='display:flex;justify-content:center;margin-top:38px;'>", unsafe_allow_html=True)
run = st.button("Run Fatigue Assessment")
st.markdown("</div></div>", unsafe_allow_html=True)

# ================= RUN PIPELINE =================
if run:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    with st.spinner("Executing behavioral inference pipeline"):
        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.008)
            progress.progress(i + 1)

    user_df = pd.DataFrame([[screen, cont, night, 4, sleep, eye, switch]], columns=FEATURES)
    fatigue = float(model.predict(user_df)[0])
    fatigue = np.clip(fatigue, 0, 100)

    color = "#22c55e" if fatigue < 40 else "#f59e0b" if fatigue < 70 else "#ef4444"

    c1, c2, c3 = st.columns(3)
    c1.markdown(f"<div class='metric' style='color:{color}'>{fatigue:.1f}</div><div class='metric-label'>Your fatigue score</div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric'>{avg_fatigue:.1f}</div><div class='metric-label'>Population baseline</div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric'>{fatigue-avg_fatigue:+.1f}</div><div class='metric-label'>Deviation</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ================= ANALYSIS =================
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("Primary Fatigue Contributors")

    factors = ["Screen Exposure", "Night Usage", "Sleep Deficit", "Visual Strain", "Context Switching"]
    values = [screen, night, 10-sleep, eye, switch]

    fig = go.Figure(go.Bar(
        x=values,
        y=factors,
        orientation="h",
        marker=dict(color=color)
    ))

    fig.update_layout(
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#020617"),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False)
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ================= RECOMMENDATIONS =================
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("Actionable Optimization Strategies")

    advice = {
        "Screen Exposure": "Introduce structured offline intervals and cap non-essential usage windows.",
        "Night Usage": "Enforce a digital sunset protocol to protect circadian rhythm stability.",
        "Sleep Deficit": "Optimize sleep hygiene to restore executive function and resilience.",
        "Visual Strain": "Apply controlled visual recovery cycles during extended sessions.",
        "Context Switching": "Batch cognitively similar tasks to minimize attention fragmentation."
    }

    for f, v in sorted(zip(factors, values), key=lambda x: x[1], reverse=True):
        st.markdown(f"""
        <div class="advice-card">
            <div class="advice-title">{f}</div>
            <div class="advice-text">{advice[f]}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
