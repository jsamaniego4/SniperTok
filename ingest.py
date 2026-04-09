from __future__ import annotations

import pandas as pd

from ..db import get_connection, initialize_database


REQUIRED_COLUMNS = [
    "post_id",
    "created_at",
    "platform",
    "creator_id",
    "creator_followers",
    "caption",
    "hashtags",
    "product_name",
    "product_category",
    "views",
    "likes",
    "comments",
    "shares",
    "saves",
    "watch_time_avg",
    "video_length_sec",
]


def validate_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def ingest_csv(csv_path: str) -> int:
    initialize_database()
    df = pd.read_csv(csv_path)
    validate_columns(df)

    df["created_at"] = pd.to_datetime(df["created_at"], utc=False).dt.strftime("%Y-%m-%d %H:%M:%S")
    numeric_cols = [
        "creator_followers", "views", "likes", "comments",
        "shares", "saves", "watch_time_avg", "video_length_sec"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    inserted = 0
    with get_connection() as conn:
        for _, row in df.iterrows():
            conn.execute(
                """
                INSERT OR REPLACE INTO posts (
                    post_id, created_at, platform, creator_id, creator_followers,
                    caption, hashtags, product_name, product_category, views,
                    likes, comments, shares, saves, watch_time_avg, video_length_sec
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                tuple(row[col] for col in REQUIRED_COLUMNS),
            )
            inserted += 1
    return inserted
