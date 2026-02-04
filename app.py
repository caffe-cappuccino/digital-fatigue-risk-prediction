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
    layout="wide"
)

# ================= GLOBAL STYLE =================
st.markdown("""
<style>

body {
    background: #F8FAFC;
    color: #020617;
}

h1 {
    font-weight: 900;
    letter-spacing: -1.2px;
}

.subtitle {
    color: #64748B;
    margin-top: -10px;
    font-size: 15px;
}

/* ---------- CARDS ---------- */
.card {
    background: #FFFFFF;
    padding: 32px;
    border-radius: 20px;
    box-shadow: 0 20px 40px rgba(2,6,23,0.08);
    margin-bottom: 36px;
}

/* ---------- CONTROL PANEL ---------- */
.control-panel {
    background: #FFFFFF;
    padding: 36px;
    border-radius: 24px;
    box-shadow: 0 25px 60px rgba(2,6,23,0.1);
    margin-bottom: 42px;
}

.panel-title {
    font-weight: 800;
}

.panel-subtitle {
    color: #64748B;
    font-size: 14px;
    margin-bottom: 28px;
}

/* ---------- INPUT GROUP ---------- */
.input-group {
    background: #F1F5F9;
    padding: 24px;
    border-radius: 18px;
}

.group-title {
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 16px;
    color: #020617;
}

/* ---------- METRICS ---------- */
.metric {
    font-size: 44px;
    font-weight: 900;
}

.metric-label {
    font-size: 13px;
    color: #64748B;
}

/* ---------- BUTTON ---------- */
.stButton > button {
    background: #1E40AF;
    color: white;
    font-weight: 700;
    height: 3.4em;
    border-radius: 14px;
    border: none;
}

.stButton > button:hover {
    background: #1D4ED8;
}

/* ---------- ADVICE ---------- */
.advice-card {
    background: #FFFFFF;
    border-left: 5px solid #0EA5E9;
    padding: 22px 26px;
    border-radius: 16px;
    box-shadow: 0 16px 30px rgba(2,6,23,0.08);
    margin-bottom: 18px;
}

.advice-title {
    font-weight: 700;
    margin-bottom: 6px;
}

.advice-text {
    color: #475569;
    font-size: 14px;
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

# ================= MODEL =================
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
    model = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42)
    model.fit(Xtr, ytr)
    return model, y.mean()

model, avg_fatigue = train_model(df)

# ================= HEADER =================
st.markdown("<h1>Digital Fatigue Intelligence</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Behavioral analytics for cognitive load and burnout risk</div>", unsafe_allow_html=True)

# ================= INPUT =================
st.markdown("<div class='control-panel'>", unsafe_allow_html=True)
st.markdown("<div class='panel-title'>Behavioral Parameters</div>", unsafe_allow_html=True)
st.markdown("<div class='panel-subtitle'>Tune variables to simulate fatigue impact</div>", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("<div class='input-group'><div class='group-title'>Exposure</div>", unsafe_allow_html=True)
    screen = st.slider("Screen Time (hours)", 1.0, 16.0, 6.0, 0.5)
    night = st.slider("Late-Night Usage (hours)", 0.0, 8.0, 1.5, 0.5)
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown("<div class='input-group'><div class='group-title'>Cognitive Load</div>", unsafe_allow_html=True)
    cont = st.slider("Continuous Session (minutes)", 10, 300, 90, 10)
    switch = st.slider("Task Switching Rate", 1, 50, 18)
    st.markdown("</div>", unsafe_allow_html=True)

with c3:
    st.markdown("<div class='input-group'><div class='group-title'>Recovery</div>", unsafe_allow_html=True)
    sleep = st.slider("Sleep Duration (hours)", 3.0, 10.0, 7.0, 0.5)
    eye = st.select_slider("Eye Strain Severity", [1,2,3,4,5], 3)
    st.markdown("</div>", unsafe_allow_html=True)

run = st.button("Run Fatigue Assessment")
st.markdown("</div>", unsafe_allow_html=True)

# ================= RUN =================
if run:
    user_df = pd.DataFrame([[screen, cont, night, 4, sleep, eye, switch]], columns=FEATURES)
    fatigue = float(model.predict(user_df)[0])
    fatigue = np.clip(fatigue, 0, 100)

    color = "#16A34A" if fatigue < 40 else "#D97706" if fatigue < 70 else "#DC2626"

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"<div class='metric' style='color:{color}'>{fatigue:.1f}</div><div class='metric-label'>Fatigue Score</div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric'>{avg_fatigue:.1f}</div><div class='metric-label'>Population Baseline</div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric'>{fatigue-avg_fatigue:+.1f}</div><div class='metric-label'>Deviation</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ================= REAL ANIMATION =================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Primary Fatigue Contributors")

    factors = ["Screen Exposure", "Night Usage", "Sleep Deficit", "Visual Strain", "Context Switching"]
    target = np.array([screen, night, 10-sleep, eye, switch])

    placeholder = st.empty()

    for step in range(1, 21):
        values = target * (step / 20)

        fig = go.Figure(go.Bar(
            x=values,
            y=factors,
            orientation="h",
            marker=dict(color="#1E40AF"),
        ))

        fig.update_layout(
            height=420,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
            bargap=0.3
        )

        placeholder.plotly_chart(fig, use_container_width=True)
        time.sleep(0.03)

    st.markdown("</div>", unsafe_allow_html=True)

    # ================= ADVICE =================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Actionable Optimization Strategies")

    advice = {
        "Screen Exposure": "Introduce structured offline intervals and cap non-essential usage windows.",
        "Night Usage": "Implement a digital sunset to protect circadian rhythm stability.",
        "Sleep Deficit": "Optimize sleep hygiene to restore executive function.",
        "Visual Strain": "Apply controlled visual recovery cycles during extended sessions.",
        "Context Switching": "Batch cognitively similar tasks to reduce attention fragmentation."
    }

    for f in factors:
        st.markdown(f"""
        <div class="advice-card">
            <div class="advice-title">{f}</div>
            <div class="advice-text">{advice[f]}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
