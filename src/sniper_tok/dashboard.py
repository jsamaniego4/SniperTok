from __future__ import annotations

import sys
from pathlib import Path
import json
from typing import Any

import pandas as pd
import streamlit as st

# Ensure `src` is on sys.path when running the dashboard directly.
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from sniper_tok.config import get_settings
from sniper_tok.services.ml import predict_category
from sniper_tok.services.trends import get_top_trends, get_trend_history

settings = get_settings()

st.set_page_config(page_title="SniperTok Dashboard", layout="wide")
st.title("SniperTok — AI Trend Intelligence")
st.markdown(
    "Explore emerging ecommerce product trends from short-form video data, compare category performance, "
    "and make live category predictions from candy-coated content signals."
)

@st.cache_data
def load_trends(limit: int) -> pd.DataFrame:
    return get_top_trends(limit=limit)

@st.cache_data
def load_history(product_name: str) -> pd.DataFrame:
    return get_trend_history(product_name)

@st.cache_data
def load_metrics() -> dict[str, Any] | None:
    if not settings.metadata_path.exists():
        return None
    with settings.metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)

col1, col2, col3 = st.columns(3)
col1.metric("SQLite warehouse", "Ready" if settings.database_path.exists() else "Missing")
col2.metric("Model artifact", "Ready" if settings.model_path.exists() else "Missing")
col3.metric("Metrics file", "Ready" if settings.metadata_path.exists() else "Missing")

st.subheader("Top trends")
limit = st.slider("Top results", min_value=5, max_value=50, value=10, step=5)
trends = load_trends(limit)

if trends.empty:
    st.warning("No trend data found yet. Run the ingestion and feature pipeline first.")
    st.info("Generate sample data with `python tools/generate_sample_data.py`, then ingest and build features.")
else:
    stats_col1, stats_col2, stats_col3 = st.columns(3)
    stats_col1.metric("Products", f"{trends['product_name'].nunique():,}")
    stats_col2.metric("Categories", f"{trends['product_category'].nunique():,}")
    stats_col3.metric("Latest date", trends["metric_date"].max())

    category_counts = trends["product_category"].value_counts().reset_index()
    category_counts.columns = ["category", "count"]

    st.markdown("### Trend snapshot")
    chart_col, table_col = st.columns((2, 1))
    with chart_col:
        st.bar_chart(category_counts.set_index("category")["count"])
    with table_col:
        st.table(category_counts.head(10))

    st.markdown("### Trend details")
    st.dataframe(trends, use_container_width=True)

    product_options = ["Choose a product..."] + trends["product_name"].dropna().unique().tolist()
    selected = st.selectbox("View product history", product_options)
    if selected != "Choose a product...":
        history = load_history(selected)
        if history.empty:
            st.info("No trend history is available for the selected product.")
        else:
            st.markdown(f"### {selected} — Trend history")
            st.line_chart(history.set_index("metric_date")["trend_score"])
            st.dataframe(history, use_container_width=True)

st.subheader("Live category prediction")
with st.sidebar:
    st.header("Predict category")
    st.write("Enter a TikTok-style post sample and get a predicted product category with confidence.")

    caption = st.text_area("Caption", "Amazon favorite mini car vacuum that cleans crumbs fast")
    hashtags = st.text_input("Hashtags", "#car #cleaning #musthave")
    product_name = st.text_input("Product name", "Mini Car Vacuum")
    views = st.number_input("Views", value=220000, min_value=0)
    likes = st.number_input("Likes", value=18000, min_value=0)
    comments = st.number_input("Comments", value=340, min_value=0)
    shares = st.number_input("Shares", value=1500, min_value=0)
    saves = st.number_input("Saves", value=1200, min_value=0)
    watch_time_avg = st.number_input("Avg watch time", value=13.5, min_value=0.0, step=0.1)
    video_length_sec = st.number_input("Video length sec", value=19.0, min_value=0.1, step=0.1)
    creator_followers = st.number_input("Creator followers", value=86000, min_value=0)
    predict_button = st.button("Predict category")

    if predict_button:
        try:
            label, confidence = predict_category(
                {
                    "caption": caption,
                    "hashtags": hashtags,
                    "product_name": product_name,
                    "views": int(views),
                    "likes": int(likes),
                    "comments": int(comments),
                    "shares": int(shares),
                    "saves": int(saves),
                    "watch_time_avg": float(watch_time_avg),
                    "video_length_sec": float(video_length_sec),
                    "creator_followers": int(creator_followers),
                }
            )
            st.success(f"Predicted category: {label} ({confidence:.2%} confidence)")
        except FileNotFoundError as exc:
            st.error(str(exc))

st.subheader("Model metrics")
metrics = load_metrics()
if metrics is None:
    st.info("No model metrics found. Train the classifier to populate this section.")
else:
    metrics_preview = {
        "accuracy": round(metrics.get("accuracy", 0.0), 4),
        "macro_f1": round(metrics.get("macro_f1", 0.0), 4),
        "rows_used": metrics.get("rows_used", 0),
        "classes": metrics.get("classes", []),
    }
    st.json(metrics_preview)
    with st.expander("View full metrics JSON"):
        st.json(metrics)
