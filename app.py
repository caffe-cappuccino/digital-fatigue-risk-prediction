import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Digital Fatigue ML",
    layout="wide"
)

# ---------------- STYLE ----------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #fde7f3, #e0f2fe);
    color: #020617;
}

.container {
    background: white;
    padding: 48px;
    border-radius: 30px;
    box-shadow: 0 35px 70px rgba(0,0,0,0.12);
    margin-bottom: 50px;
}

.title {
    font-size: 44px;
    font-weight: 900;
    background: linear-gradient(90deg, #ec4899, #6366f1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    font-size: 18px;
    color: #475569;
    margin-bottom: 40px;
}

.metric {
    font-size: 64px;
    font-weight: 900;
}

.metric-label {
    color: #64748b;
    font-size: 14px;
}

.section-title {
    font-size: 22px;
    font-weight: 800;
    margin-bottom: 20px;
}

.stButton > button {
    background: linear-gradient(90deg, #6366f1, #ec4899);
    color: white;
    border-radius: 20px;
    height: 3.8em;
    font-size: 18px;
    font-weight: 700;
    border: none;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# ---------------- DATA (STABLE SYNTHETIC) ----------------
np.random.seed(42)
data = pd.DataFrame({
    "screen_time": np.random.uniform(2, 14, 300),
    "night_usage": np.random.uniform(0, 5, 300),
    "sleep": np.random.uniform(4, 9, 300),
    "eye_strain": np.random.randint(1, 6, 300),
    "task_switch": np.random.randint(5, 40, 300)
})

fatigue_raw = (
    data["screen_time"] * 6 +
    data["night_usage"] * 8 -
    data["sleep"] * 7 +
    data["eye_strain"] * 10 +
    data["task_switch"] * 1.2
)

data["fatigue"] = (fatigue_raw - fatigue_raw.min()) / (fatigue_raw.max() - fatigue_raw.min()) * 100

X = data.drop("fatigue", axis=1)
y = data["fatigue"]

X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2)
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ---------------- UI ----------------
st.markdown("<div class='container'>", unsafe_allow_html=True)

st.markdown("<div class='title'>Digital Fatigue Intelligence</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>ML-powered estimation of digital burnout risk</div>", unsafe_allow_html=True)

st.markdown("<div class='section-title'>Input Parameters</div>", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    screen = st.slider("Screen Time (hrs)", 1.0, 16.0, 6.0)
    night = st.slider("Night Usage (hrs)", 0.0, 8.0, 2.0)

with c2:
    sleep = st.slider("Sleep (hrs)", 3.0, 10.0, 7.0)
    eye = st.slider("Eye Strain", 1, 5, 3)

with c3:
    switch = st.slider("Task Switching Rate", 1, 50, 20)

run = st.button("Run Fatigue Prediction")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- RESULTS ----------------
if run:
    user = pd.DataFrame([[screen, night, sleep, eye, switch]], columns=X.columns)
    fatigue = float(model.predict(user)[0])

    color = "#22c55e" if fatigue < 40 else "#f59e0b" if fatigue < 70 else "#ef4444"

    st.markdown("<div class='container'>", unsafe_allow_html=True)

    c1, c2 = st.columns([1,2])
    with c1:
        st.markdown(f"<div class='metric' style='color:{color}'>{fatigue:.1f}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Predicted Fatigue Score</div>", unsafe_allow_html=True)

    with c2:
        fig = go.Figure(go.Bar(
            x=user.iloc[0],
            y=user.columns,
            orientation="h",
            marker_color="#6366f1"
        ))

        fig.update_layout(
            height=320,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False)
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)
