from __future__ import annotations

import numpy as np
import pandas as pd

from ..db import get_connection, initialize_database


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    den = den.replace(0, np.nan)
    return (num / den).fillna(0.0)


def build_daily_product_features() -> pd.DataFrame:
    initialize_database()
    with get_connection() as conn:
        raw = pd.read_sql_query("SELECT * FROM posts", conn)

    if raw.empty:
        raise ValueError("No posts found. Run ingestion first.")

    raw["metric_date"] = pd.to_datetime(raw["created_at"]).dt.date.astype(str)

    grouped = (
        raw.groupby(["metric_date", "product_name", "product_category"], as_index=False)
        .agg(
            posts_count=("post_id", "count"),
            total_views=("views", "sum"),
            total_likes=("likes", "sum"),
            total_comments=("comments", "sum"),
            total_shares=("shares", "sum"),
            total_saves=("saves", "sum"),
            avg_watch_time=("watch_time_avg", "mean"),
            avg_video_length=("video_length_sec", "mean"),
            avg_creator_followers=("creator_followers", "mean"),
        )
        .sort_values(["product_name", "metric_date"])
    )

    grouped["engagement_rate"] = _safe_div(
        grouped["total_likes"] + grouped["total_comments"] + grouped["total_shares"] + grouped["total_saves"],
        grouped["total_views"],
    )
    grouped["share_rate"] = _safe_div(grouped["total_shares"], grouped["total_views"])
    grouped["save_rate"] = _safe_div(grouped["total_saves"], grouped["total_views"])
    grouped["watch_through_rate"] = _safe_div(grouped["avg_watch_time"], grouped["avg_video_length"])

    grouped["prev_views"] = grouped.groupby("product_name")["total_views"].shift(1).fillna(0)
    grouped["prev_engagement_rate"] = grouped.groupby("product_name")["engagement_rate"].shift(1).fillna(0)

    grouped["views_growth"] = _safe_div(grouped["total_views"] - grouped["prev_views"], grouped["prev_views"] + 1)
    grouped["engagement_growth"] = grouped["engagement_rate"] - grouped["prev_engagement_rate"]

    grouped["velocity_score"] = (
        np.log1p(grouped["total_views"]) * 0.50
        + np.log1p(grouped["total_shares"] + grouped["total_saves"]) * 0.30
        + grouped["engagement_rate"] * 100 * 0.20
    )

    grouped["momentum_score"] = (
        grouped["views_growth"].clip(-1, 5) * 0.70
        + grouped["engagement_growth"].clip(-1, 1) * 8.0 * 0.30
    )

    grouped["creator_quality_score"] = (
        np.log1p(grouped["avg_creator_followers"]) * 0.40
        + grouped["watch_through_rate"].clip(0, 1.5) * 3.5 * 0.35
        + grouped["save_rate"].clip(0, 1) * 100 * 0.25
    )

    def _normalize(series: pd.Series) -> pd.Series:
        if series.nunique() <= 1:
            return pd.Series(np.ones(len(series)) * 0.5, index=series.index)
        return (series - series.min()) / (series.max() - series.min())

    grouped["trend_score"] = (
        _normalize(grouped["velocity_score"]) * 0.35
        + _normalize(grouped["momentum_score"]) * 0.30
        + _normalize(grouped["watch_through_rate"]) * 0.15
        + _normalize(grouped["save_rate"]) * 0.10
        + _normalize(grouped["creator_quality_score"]) * 0.10
    ) * 100

    output_cols = [
        "metric_date", "product_name", "product_category", "posts_count",
        "total_views", "total_likes", "total_comments", "total_shares", "total_saves",
        "avg_watch_time", "avg_video_length", "avg_creator_followers",
        "engagement_rate", "share_rate", "save_rate", "watch_through_rate",
        "velocity_score", "momentum_score", "creator_quality_score", "trend_score",
    ]
    result = grouped[output_cols].copy()

    with get_connection() as conn:
        conn.execute("DELETE FROM product_daily_metrics")
        result.to_sql("product_daily_metrics", conn, if_exists="append", index=False)

    return result
