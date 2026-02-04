import streamlit as st
import plotly.graph_objects as go
import numpy as np
import time

st.set_page_config(layout="wide")

st.title("Animated Fatigue Contributors")

# Sample data (replace with your real values)
factors = [
    "Screen Exposure",
    "Night Usage",
    "Sleep Deficit",
    "Visual Strain",
    "Context Switching"
]

target_values = np.array([6.0, 1.5, 3.0, 3.0, 18.0])

# Placeholder for animation
chart_placeholder = st.empty()

# Animation loop
steps = 25
for step in range(1, steps + 1):
    current_values = target_values * (step / steps)

    fig = go.Figure(
        data=[
            go.Bar(
                x=current_values,
                y=factors,
                orientation="h",
                marker=dict(
                    color="#1E40AF",   # Deep blue
                    line=dict(width=0)
                )
            )
        ]
    )

    fig.update_layout(
        height=420,
        margin=dict(l=40, r=20, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            showgrid=False
        ),
        bargap=0.35,
        font=dict(
            color="#020617",
            size=13
        )
    )

    chart_placeholder.plotly_chart(fig, use_container_width=True)
    time.sleep(0.03)
