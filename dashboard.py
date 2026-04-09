from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from sniper_tok.config import get_settings
from sniper_tok.services.trends import get_top_trends, get_trend_history


st.set_page_config(page_title="SniperTok Dashboard", layout="wide")
st.title("SniperTok — AI Trend Intelligence")

settings = get_settings()

col1, col2, col3 = st.columns(3)
col1.metric("Database", "Ready" if settings.database_path.exists() else "Missing")
col2.metric("Model Artifact", "Ready" if settings.model_path.exists() else "Missing")
col3.metric("Metrics File", "Ready" if settings.metadata_path.exists() else "Missing")

st.subheader("Top Trends")
limit = st.slider("Rows", min_value=5, max_value=50, value=10, step=5)
trends = get_top_trends(limit=limit)

if trends.empty:
    st.info("No trend data found yet. Run the pipeline first.")
else:
    st.dataframe(trends, use_container_width=True)

    product_options = trends["product_name"].dropna().unique().tolist()
    if product_options:
        selected = st.selectbox("View product history", product_options)
        history = get_trend_history(selected)
        if not history.empty:
            st.line_chart(history.set_index("metric_date")["trend_score"])
            st.dataframe(history, use_container_width=True)

st.subheader("Latest Model Metrics")
if settings.metadata_path.exists():
    with settings.metadata_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    preview = {
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "rows_used": metrics["rows_used"],
        "classes": metrics["classes"],
    }
    st.json(preview)
else:
    st.info("Train the model to populate metrics.")
