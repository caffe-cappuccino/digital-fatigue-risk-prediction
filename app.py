import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Digital Fatigue Monitor",
    page_icon="🌸",
    layout="centered"
)

# ================= PREMIUM UI =================
st.markdown("""
<style>
body { background-color: #f6f7fb; }
h1 { color: #2d2f7f; font-weight: 700; }
h2, h3 { color: #2f2f2f; }
.card {
    background: white;
    padding: 26px;
    border-radius: 18px;
    box-shadow: 0 12px 30px rgba(0,0,0,0.08);
    margin-bottom: 32px;
}
.kpi {
    background: #f0f1ff;
    padding: 18px;
    border-radius: 14px;
    text-align: center;
}
.kpi h2 { margin: 0; }
.stButton > button {
    background: linear-gradient(90deg,#6c63ff,#8f88ff);
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-size: 16px;
    font-weight: 600;
    border: none;
}
.progress {
    background: #ecebff;
    border-radius: 10px;
    height: 12px;
    overflow: hidden;
}
.progress > div {
    background: linear-gradient(90deg,#6c63ff,#8f88ff);
    height: 100%;
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

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42)
    model.fit(Xtr, ytr)

    return model, y.mean()

model, avg_fatigue = train_model(df)

# ================= HEADER =================
st.title("Digital Fatigue Monitor")
st.write("A high-end analytics system to understand how digital habits impact fatigue.")

# ================= INPUT =================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("Daily Usage Input")

c1, c2 = st.columns(2)
with c1:
    screen = st.slider("Screen time (hrs)", 1.0, 16.0, 6.0, 0.5)
    night = st.slider("Late-night usage (hrs)", 0.0, 8.0, 1.5, 0.5)
    sleep = st.slider("Sleep (hrs)", 3.0, 10.0, 7.0, 0.5)
with c2:
    cont = st.slider("Longest continuous usage (min)", 10, 300, 90, 10)
    eye = st.select_slider("Eye strain", [1,2,3,4,5], 3)
    switch = st.slider("Task switching", 1, 50, 18)

predict = st.button("Analyze Fatigue")
st.markdown("</div>", unsafe_allow_html=True)

# ================= ANALYSIS =================
if predict:
    user_df = pd.DataFrame([[screen, cont, night, 4, sleep, eye, switch]], columns=FEATURES)
    fatigue = float(model.predict(user_df)[0])
    fatigue = np.clip(fatigue, 0, 100)

    # ================= KPIs =================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"<div class='kpi'><h2>{fatigue:.1f}</h2><p>Your score</p></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='kpi'><h2>{avg_fatigue:.1f}</h2><p>Avg user</p></div>", unsafe_allow_html=True)
    diff = fatigue - avg_fatigue
    c3.markdown(f"<div class='kpi'><h2>{diff:+.1f}</h2><p>Difference</p></div>", unsafe_allow_html=True)

    st.markdown(f"<div class='progress'><div style='width:{fatigue}%;'></div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ================= TABS =================
    tab1, tab2, tab3 = st.tabs(["📊 Contributors", "🧠 Profile", "📈 Population"])

    # -------- LOLLIPOP --------
    with tab1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        factors = ["Screen", "Night", "Low Sleep", "Eye", "Switching"]
        values = np.array([screen, night, 10-sleep, eye, switch])
        y = np.arange(len(factors))

        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        ax.hlines(y, 0, values, color="#c7c5ff", linewidth=4)
        ax.plot(values, y, "o", color="#6c63ff")
        ax.set_yticks(y)
        ax.set_yticklabels(factors)
        ax.set_xlabel("Impact")
        ax.set_title("Contribution Breakdown")
        ax.spines[['top','right','left']].set_visible(False)
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    # -------- RADAR --------
    with tab2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        radar_vals = values / values.max()
        radar_vals = np.append(radar_vals, radar_vals[0])
        angles = np.linspace(0, 2*np.pi, len(radar_vals))

        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(111, polar=True)
        ax.plot(angles, radar_vals, color="#6c63ff", linewidth=2)
        ax.fill(angles, radar_vals, color="#c7c5ff", alpha=0.5)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(factors)
        ax.set_yticklabels([])
        ax.set_title("Your Digital Profile")
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    # -------- DISTRIBUTION --------
    with tab3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5.5,3))
        ax.hist(df["screen_time_hours"], bins=20, color="#e6e5ff")
        ax.axvline(screen, color="#6c63ff", linewidth=3)
        ax.set_title("Screen Time vs Population")
        ax.set_xlabel("Hours")
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    # ================= INSIGHTS =================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Actionable Insights")
    if fatigue > 65:
        st.error("High fatigue risk. Strong recovery needed.")
    elif fatigue > 35:
        st.warning("Moderate fatigue. Small optimizations recommended.")
    else:
        st.success("Low fatigue. Habits look balanced.")

    st.markdown("</div>", unsafe_allow_html=True)

# ================= FOOTER =================
st.caption("One file • High-end • No crashes • Done.")
