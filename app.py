import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Digital Fatigue",
    page_icon="🌸",
    layout="centered"
)

# ================= LOAD MODEL =================
model = None
model_error = None
try:
    model = joblib.load("model/fatigue_model.pkl")
except Exception as e:
    model_error = str(e)

# ================= ADVANCED UI =================
st.markdown("""
<style>

/* ===== PARALLAX-LIKE BACKGROUND ===== */
body {
    background:
        radial-gradient(circle at 20% 10%, #2a1f4d, transparent 40%),
        radial-gradient(circle at 80% 30%, #1b2a4d, transparent 45%),
        linear-gradient(180deg, #0c0c14, #141425);
    background-attachment: fixed;
}

/* Remove Streamlit defaults */
section.main > div,
.block-container,
div[data-testid="stVerticalBlock"],
div[data-testid="stHorizontalBlock"],
div[data-testid="stMarkdownContainer"] {
    background: transparent !important;
}

/* ===== GLASS DEPTH VARIANTS ===== */
.glass {
    background: rgba(255,255,255,0.07);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border-radius: 26px;
    padding: 28px;
    margin-bottom: 48px;
    border: 1px solid rgba(255,255,255,0.12);
    transition: box-shadow 0.3s ease, transform 0.3s ease;
}

.glass.deep {
    background: rgba(255,255,255,0.10);
    backdrop-filter: blur(18px);
}

/* ===== HOVER GLOW ===== */
.glass:hover {
    box-shadow:
        0 0 22px rgba(160,140,255,0.35),
        inset 0 0 20px rgba(255,255,255,0.04);
    transform: translateY(-2px);
}

/* ===== TEXT ===== */
h1 { color: #f3f1ff; font-weight: 700; }
h2, h3 { color: #e6e4ff; }
p, span, label { color: #d6d4ff !important; }

/* ===== BUTTON ===== */
.stButton > button {
    background: linear-gradient(90deg, #a18cd1, #fbc2eb);
    color: #1a1a1a;
    border-radius: 28px;
    height: 3.2em;
    width: 100%;
    font-size: 16px;
    font-weight: 600;
    border: none;
    transition: all 0.3s ease;
    box-shadow: 0 0 14px rgba(251,194,235,0.4);
}
.stButton > button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 26px rgba(161,140,209,0.75);
}

/* ===== ANIMATED PROGRESS ===== */
.progress-wrapper {
    background: rgba(255,255,255,0.18);
    border-radius: 12px;
    height: 14px;
    width: 100%;
    overflow: hidden;
    margin-top: 10px;
}

.progress-fill {
    height: 100%;
    width: 0%;
    border-radius: 12px;
    background: linear-gradient(90deg, #a18cd1, #fbc2eb);
    animation: fillBar 1.2s ease forwards;
}

@keyframes fillBar {
    from { width: 0%; }
    to { width: var(--fill); }
}

</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.title("🌸 Digital Fatigue Monitor")
st.write(
    "A layered, depth-based interface to explore how digital habits "
    "affect fatigue."
)

# ================= MODEL ERROR =================
if model_error:
    st.error("Model could not be loaded")
    st.code(model_error)
    st.stop()

# ================= INPUT =================
st.markdown("<div class='glass'>", unsafe_allow_html=True)
st.subheader("Daily Usage Overview")

c1, c2 = st.columns(2)

with c1:
    screen_time = st.slider("Screen time (hours)", 1.0, 16.0, 6.0, 0.5)
    night_usage = st.slider("Late-night usage (hours)", 0.0, 8.0, 1.5, 0.5)
    sleep = st.slider("Sleep duration (hours)", 3.0, 10.0, 7.0, 0.5)

with c2:
    continuous_usage = st.slider("Longest continuous usage (minutes)", 10, 300, 90, 10)
    eye_strain = st.select_slider("Eye strain level", [1,2,3,4,5], 3)
    task_switch = st.slider("Task switching frequency", 1, 50, 18)

predict = st.button("Analyze Fatigue")
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

    fatigue = model.predict(input_df)[0]
    fatigue_pct = min(max(fatigue, 0), 100)

    # ================= RESULT =================
    st.markdown("<div class='glass deep'>", unsafe_allow_html=True)
    st.subheader("Fatigue Level")
    st.write(f"**{fatigue_pct:.1f} / 100**")

    st.markdown(
        f"""
        <div class="progress-wrapper">
            <div class="progress-fill" style="--fill:{fatigue_pct}%;"></div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if fatigue < 35:
        st.success("Low fatigue detected.")
    elif fatigue < 65:
        st.warning("Moderate fatigue detected.")
    else:
        st.error("High fatigue detected.")

    st.markdown("</div>", unsafe_allow_html=True)

    # ================= LOLLIPOP =================
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("Main Contributors")

    factors = ["Screen time", "Night usage", "Low sleep", "Eye strain", "Task switching"]
    values = np.array([
        screen_time,
        night_usage,
        10 - sleep,
        eye_strain,
        task_switch
    ])

    y_pos = np.arange(len(factors))

    fig, ax = plt.subplots(figsize=(5, 3.5))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    ax.hlines(y=y_pos, xmin=0, xmax=values, color="#b8a9ff", linewidth=3)
    ax.plot(values, y_pos, "o", color="#fbc2eb", markersize=8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(factors, color="#e6e4ff")
    ax.set_xlabel("Relative impact", color="#e6e4ff")
    ax.set_title("Fatigue contributors", color="#f3f1ff")

    for spine in ax.spines.values():
        spine.set_visible(False)

    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

# ================= FOOTER =================
st.caption("Depth • Motion • Calm ✨")
