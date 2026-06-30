"""
Smart Cart Abandonment Predictor — Live Demo
Streams synthetic sessions through the trained XGBoost model in real-time.

Run locally:
    streamlit run demo/app.py

Deploy: HuggingFace Spaces (SDK: Streamlit)
"""

import random
import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.generate import generate
from src.features.engineering import extract_features, features_to_array
from src.model.predict import Incentive, recommend_incentive
from src.model.train import MODEL_PATH, build_training_data, train_model

st.set_page_config(
    page_title="Cart Abandonment Predictor",
    page_icon="🛒",
    layout="wide",
)


@st.cache_resource(show_spinner="Training model on synthetic data...")
def load_or_train_model() -> xgb.XGBClassifier:
    if MODEL_PATH.exists():
        model = xgb.XGBClassifier()
        model.load_model(str(MODEL_PATH))
        return model
    X, y = build_training_data(volume=20_000, seed=42)
    return train_model(X, y)


INCENTIVE_LABELS = {
    Incentive.NONE: ("No action needed", "green"),
    Incentive.URGENCY: ("Show urgency message", "orange"),
    Incentive.FREE_SHIPPING: ("Offer free shipping", "orange"),
    Incentive.DISCOUNT_10: ("Apply 10% discount", "red"),
}

SCENARIO_OPTIONS = {
    "Mixed (realistic ~70% abandonment)": "mixed",
    "High Risk (85% abandonment)": "high_risk",
    "Happy Path (25% abandonment)": "happy_path",
    "Edge Cases": "edge_cases",
}


@st.fragment
def stream_sessions(model: xgb.XGBClassifier, scenario: str, num_sessions: int, speed: float) -> None:
    """Runs the streaming loop in an isolated fragment to avoid session rerun conflicts."""
    sessions = generate(scenario, num_sessions, seed=random.randint(1, 9999))
    delay = 1.0 / speed

    col1, col2, col3, col4 = st.columns(4)
    metric_total = col1.empty()
    metric_high_risk = col2.empty()
    metric_avg_risk = col3.empty()
    col4.metric("Model AUC-ROC", "~0.80", help="Measured on held-out synthetic test set")

    st.divider()
    col_feed, col_detail = st.columns([2, 1])
    with col_feed:
        st.subheader("Live Session Feed")
        feed_placeholder = st.empty()
    with col_detail:
        st.subheader("Last Prediction Detail")
        detail_placeholder = st.empty()
    chart_placeholder = st.empty()

    history: list[dict] = []
    feed_rows: list[dict] = []

    for i, session in enumerate(sessions):
        features = extract_features(session.__dict__)
        arr = features_to_array(features).reshape(1, -1)
        risk_score = float(model.predict_proba(arr)[0][1])
        incentive = recommend_incentive(risk_score)
        high_risk = risk_score >= 0.6
        label, color = INCENTIVE_LABELS[incentive]

        history.append({
            "risk_score": risk_score,
            "high_risk": high_risk,
        })
        feed_rows.insert(0, {
            "#": i + 1,
            "Risk Score": f"{risk_score:.2f}",
            "Device": session.device_type,
            "Cart $": f"${session.cart_value:.0f}",
            "Incentive": incentive.value,
            "Ground Truth": "Abandoned" if session.abandoned else "Converted",
        })

        total = len(history)
        high_risk_count = sum(1 for h in history if h["high_risk"])

        metric_total.metric("Sessions Scored", total)
        metric_high_risk.metric(
            "High Risk (>=0.6)", f"{high_risk_count} ({high_risk_count / total:.0%})"
        )
        metric_avg_risk.metric(
            "Avg Risk Score", f"{np.mean([h['risk_score'] for h in history]):.2f}"
        )
        feed_placeholder.dataframe(feed_rows[:20], use_container_width=True, hide_index=True)
        detail_placeholder.markdown(f"""
**Session #{i + 1}**
- Risk Score: `{risk_score:.3f}`
- :{color}[{label}]
- Device: `{session.device_type}`
- Cart Value: `${session.cart_value:.2f}`
- Items: `{session.item_count}`
- Session Duration: `{session.session_duration_seconds:.0f}s`
- Checkout Attempts: `{session.checkout_start_attempts}`
""")
        if len(history) >= 5:
            import pandas as pd
            chart_data = pd.DataFrame({"Risk Score": [h["risk_score"] for h in history[-50:]]})
            chart_placeholder.line_chart(chart_data, y="Risk Score", use_container_width=True)

        time.sleep(delay)

    st.success(f"Demo complete! Scored {len(history)} sessions.")


def main() -> None:
    st.title("Smart Cart Abandonment Predictor")
    st.caption("Real-time ML engine · XGBoost · Trained on synthetic e-commerce sessions")

    model = load_or_train_model()

    with st.sidebar:
        st.header("Demo Controls")
        scenario_label = st.selectbox("Scenario", list(SCENARIO_OPTIONS.keys()))
        scenario = SCENARIO_OPTIONS[scenario_label]
        speed = st.slider("Sessions per second", 0.5, 3.0, 1.0, 0.5)
        num_sessions = st.number_input("Sessions to run", 20, 500, 100, 10)
        run = st.button("Run Demo", use_container_width=True)

    if not run:
        _, _, _, col4 = st.columns(4)
        col4.metric("Model AUC-ROC", "~0.80", help="Measured on held-out synthetic test set")
        st.divider()
        st.info("Configure a scenario in the sidebar and click **Run Demo** to start.")
        return

    stream_sessions(model, scenario, int(num_sessions), speed)


main()
