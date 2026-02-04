import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go

# ================= PAGE CONFIG =================
st.set_page_config(layout="wide")

# ================= CSS (HARD OVERRIDE) =================
st.markdown("""
<style>

/* FULL APP BACKGROUND */
.stApp {
    background: linear-gradient(135deg, #fde7f3, #e0f2fe);
}

/* REMOVE ALL DEFAULT PADDING + GAPS */
.block-container {
    padding: 1.5rem !important;
}

/* REMOVE COLUMN GAPS COMPLETELY */
div[data-testid="column"] {
    padding: 0 !important;
    margin: 0 !important;
}

/* SINGLE CARD */
.card {
    background: white;
    border-radius: 28px;
    padding: 42px;
    box-shadow: 0 30px 60px rgba(0,0,0,0.12);
}

/* TEXT */
.title {
    font-size: 44px;
    font-weight: 900;
    background: linear-gradient(90deg, #ec4899, #6366f1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    color: #475569;
    font-size: 18px;
    margin-bottom: 30px;
}

.metric {
    font-size: 64px;
    font-weight: 900;
}

.metric-label {
    color: #64748b;
}

/* BUTTON */
.stButton > button {
    background: linear-gradient(90deg, #6366f1, #ec4899);
    color: white;
    border-radius: 18px;
    height: 3.6em;
    font-size: 18px;
    font-weight: 700;
    border: none;
    width: 100%;
}

</style>
""", unsafe_allow_html=True)

# ================= ML (REAL + STABLE) =================
np.random.seed(42)
df = pd.DataFrame({
    "screen": np.random.uniform(2, 14, 300),
    "night": np.random.uniform(0, 5, 300),
    "sleep": np.random.uniform(4, 9, 300),
    "strain": np.random.randint(1, 6, 300),
    "switch": np.random.randint(5, 40, 300),
})

y = (
    df["screen"] * 6 +
    df["night"] * 8 -
    df["sleep"] * 7 +
    df["strain"] * 10 +
    df["switch"] * 1.2
)

X_train, _, y_train, _ = train_test_split(df, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ================= UI (ONE CONTAINER ONLY) =================
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.markdown("<div class='title'>Digital Fatigue Intelligence</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>ML-powered digital fatigue prediction</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        screen = st.slider("Screen Time (hrs)", 1.0, 16.0, 6.0)
        night = st.slider("Night Usage (hrs)", 0.0, 8.0, 2.0)

    with c2:
        sleep = st.slider("Sleep (hrs)", 3.0, 10.0, 7.0)
        strain = st.slider("Eye Strain", 1, 5, 3)

    with c3:
        switch = st.slider("Task Switching", 1, 50, 20)

    run = st.button("Predict Fatigue")

    if run:
        user = pd.DataFrame([[screen, night, sleep, strain, switch]], columns=df.columns)
        fatigue = model.predict(user)[0]

        color = "#22c55e" if fatigue < 40 else "#f59e0b" if fatigue < 70 else "#ef4444"

        st.markdown(f"<div class='metric' style='color:{color}'>{fatigue:.1f}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Predicted Fatigue Score</div>", unsafe_allow_html=True)

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
