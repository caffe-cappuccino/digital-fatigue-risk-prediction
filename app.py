import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import time

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Digital Fatigue Evaluation",
    page_icon="📊",
    layout="wide"
)

# ================= STYLE =================
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: #e6e6e6;
}
h1, h2, h3 {
    color: #ffffff;
}
.card {
    background: #161b22;
    padding: 24px;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 24px;
}
.metric {
    font-size: 30px;
    font-weight: 700;
}
.label {
    opacity: 0.7;
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

    model = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42)
    model.fit(Xtr, ytr)
    return model, y.mean()

model, avg_fatigue = train_model(df)

# ================= HEADER =================
st.title("Digital Fatigue Evaluation System")
st.write("Interactive 3D evaluation of digital behaviour and fatigue risk")

# ================= INPUT =================
st.markdown("<div class='card'>", unsafe_allow_html=True)
c1, c2 = st.columns(2)

with c1:
    screen = st.slider("Screen Time (hours)", 1.0, 16.0, 6.0, 0.5)
    night = st.slider("Late-night Usage (hours)", 0.0, 8.0, 1.5, 0.5)
    sleep = st.slider("Sleep Duration (hours)", 3.0, 10.0, 7.0, 0.5)

with c2:
    cont = st.slider("Longest Continuous Usage (minutes)", 10, 300, 90, 10)
    eye = st.select_slider("Eye Strain Level", [1,2,3,4,5], 3)
    switch = st.slider("Task Switching Frequency", 1, 50, 18)

run = st.button("Run Evaluation")
st.markdown("</div>", unsafe_allow_html=True)

# ================= EVALUATION =================
if run:
    with st.spinner("Evaluating digital fatigue..."):
        time.sleep(1.2)

    user_df = pd.DataFrame([[screen, cont, night, 4, sleep, eye, switch]], columns=FEATURES)
    fatigue = float(model.predict(user_df)[0])
    fatigue = np.clip(fatigue, 0, 100)

    # ================= KPIs =================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"<div class='metric'>{fatigue:.1f}</div><div class='label'>Your Score</div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric'>{avg_fatigue:.1f}</div><div class='label'>Average User</div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric'>{fatigue-avg_fatigue:+.1f}</div><div class='label'>Difference</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ================= CONTRIBUTIONS =================
    factors = ["Screen Time", "Night Usage", "Low Sleep", "Eye Strain", "Task Switching"]
    values = np.array([screen, night, 10-sleep, eye, switch])

    # ================= 3D INTERACTIVE GRAPH =================
    fig3d = go.Figure(
        data=[go.Scatter3d(
            x=values,
            y=np.arange(len(values)),
            z=[fatigue]*len(values),
            mode='markers+lines',
            marker=dict(
                size=8,
                color=values,
                colorscale='Blues',
                opacity=0.85
            )
        )]
    )

    fig3d.update_layout(
        title="3D Fatigue Contribution Space",
        scene=dict(
            xaxis_title="Impact",
            yaxis_title="Factor Index",
            zaxis_title="Fatigue Score"
        ),
        height=500,
        transition_duration=900
    )

    st.plotly_chart(fig3d, use_container_width=True)

    # ================= FACTOR + ADVICE =================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Key Contributors & Advice")

    advice_map = {
        "Screen Time": "Reduce overall daily screen exposure.",
        "Night Usage": "Avoid screens at least 1 hour before sleep.",
        "Low Sleep": "Increase sleep duration for recovery.",
        "Eye Strain": "Apply the 20–20–20 eye care rule.",
        "Task Switching": "Reduce frequent context switching."
    }

    for f, v in sorted(zip(factors, values), key=lambda x: x[1], reverse=True):
        st.write(f"**{f}** → Impact: {v:.2f}")
        st.write(f"• {advice_map[f]}")

    st.markdown("</div>", unsafe_allow_html=True)

# ================= FOOTER =================
st.caption("3D • Interactive • Animated • Evaluation-ready")
