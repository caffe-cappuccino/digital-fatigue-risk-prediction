import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import time

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Digital Fatigue Assessment",
    page_icon="",
    layout="wide"
)

# ================= STYLE =================
st.markdown("""
<style>
body {
    background-color: #f9fafb;
    color: #111827;
}
h1, h2, h3 {
    color: #111827;
    font-weight: 600;
}
.subtitle {
    color: #6b7280;
    margin-top: -8px;
}
.card {
    background: #ffffff;
    padding: 28px;
    border-radius: 14px;
    border: 1px solid #e5e7eb;
    margin-bottom: 28px;
}
.metric {
    font-size: 34px;
    font-weight: 600;
}
.metric-label {
    color: #6b7280;
    font-size: 13px;
}
.divider {
    height: 1px;
    background: #e5e7eb;
    margin: 24px 0;
}
.stButton > button {
    background: #2563eb;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 15px;
    font-weight: 500;
    border: none;
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
st.title("Digital Fatigue Assessment")
st.markdown("<div class='subtitle'>A behavioral analysis of digital usage and fatigue risk</div>", unsafe_allow_html=True)

tab_input, tab_results, tab_analysis, tab_advice = st.tabs(
    ["Input", "Summary", "Analysis", "Recommendations"]
)

# ================= INPUT =================
with tab_input:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        screen = st.slider("Screen time (hours)", 1.0, 16.0, 6.0, 0.5)
        night = st.slider("Late-night usage (hours)", 0.0, 8.0, 1.5, 0.5)
        sleep = st.slider("Sleep duration (hours)", 3.0, 10.0, 7.0, 0.5)

    with c2:
        cont = st.slider("Longest continuous usage (minutes)", 10, 300, 90, 10)
        eye = st.select_slider("Eye strain level", [1,2,3,4,5], 3)
        switch = st.slider("Task switching frequency", 1, 50, 18)

    run = st.button("Run assessment")
    st.markdown("</div>", unsafe_allow_html=True)

# ================= RUN =================
if run:
    with st.spinner("Evaluating..."):
        time.sleep(0.8)

    user_df = pd.DataFrame([[screen, cont, night, 4, sleep, eye, switch]], columns=FEATURES)
    fatigue = float(model.predict(user_df)[0])
    fatigue = np.clip(fatigue, 0, 100)

    # ================= SUMMARY =================
    with tab_results:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"<div class='metric'>{fatigue:.1f}</div><div class='metric-label'>Your score</div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric'>{avg_fatigue:.1f}</div><div class='metric-label'>Population average</div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='metric'>{fatigue-avg_fatigue:+.1f}</div><div class='metric-label'>Difference</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ================= ANALYSIS =================
    with tab_analysis:
        factors = ["Screen time", "Night usage", "Low sleep", "Eye strain", "Task switching"]
        values = [screen, night, 10-sleep, eye, switch]

        fig = go.Figure(go.Bar(
            x=values,
            y=factors,
            orientation="h",
            marker_color="#2563eb"
        ))
        fig.update_layout(
            title="Relative contribution to fatigue",
            height=420,
            margin=dict(l=40, r=20, t=50, b=40),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ================= ADVICE =================
    with tab_advice:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Targeted recommendations")

        advice = {
            "Screen time": "Reduce total daily screen exposure and introduce offline breaks.",
            "Night usage": "Avoid screens at least one hour before sleep.",
            "Low sleep": "Increase sleep duration to support recovery.",
            "Eye strain": "Use the 20–20–20 rule to reduce eye fatigue.",
            "Task switching": "Batch tasks to minimize cognitive context switching."
        }

        for f, v in sorted(zip(factors, values), key=lambda x: x[1], reverse=True):
            st.markdown(f"**{f}**")
            st.write(advice[f])
            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

# ================= FOOTER =================
st.caption("Minimal • Analytical • Professional")
