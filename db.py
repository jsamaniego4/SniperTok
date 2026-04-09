from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .config import get_settings


POSTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS posts (
    post_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    platform TEXT NOT NULL,
    creator_id TEXT NOT NULL,
    creator_followers INTEGER NOT NULL,
    caption TEXT NOT NULL,
    hashtags TEXT NOT NULL,
    product_name TEXT NOT NULL,
    product_category TEXT NOT NULL,
    views INTEGER NOT NULL,
    likes INTEGER NOT NULL,
    comments INTEGER NOT NULL,
    shares INTEGER NOT NULL,
    saves INTEGER NOT NULL,
    watch_time_avg REAL NOT NULL,
    video_length_sec REAL NOT NULL
);
"""

PRODUCT_DAILY_SCHEMA = """
CREATE TABLE IF NOT EXISTS product_daily_metrics (
    metric_date TEXT NOT NULL,
    product_name TEXT NOT NULL,
    product_category TEXT NOT NULL,
    posts_count INTEGER NOT NULL,
    total_views INTEGER NOT NULL,
    total_likes INTEGER NOT NULL,
    total_comments INTEGER NOT NULL,
    total_shares INTEGER NOT NULL,
    total_saves INTEGER NOT NULL,
    avg_watch_time REAL NOT NULL,
    avg_video_length REAL NOT NULL,
    avg_creator_followers REAL NOT NULL,
    engagement_rate REAL NOT NULL,
    share_rate REAL NOT NULL,
    save_rate REAL NOT NULL,
    watch_through_rate REAL NOT NULL,
    velocity_score REAL NOT NULL,
    momentum_score REAL NOT NULL,
    creator_quality_score REAL NOT NULL,
    trend_score REAL NOT NULL,
    PRIMARY KEY(metric_date, product_name)
);
"""

MODEL_REGISTRY_SCHEMA = """
CREATE TABLE IF NOT EXISTS model_registry (
    model_name TEXT NOT NULL,
    trained_at TEXT NOT NULL,
    accuracy REAL NOT NULL,
    macro_f1 REAL NOT NULL,
    rows_used INTEGER NOT NULL,
    PRIMARY KEY(model_name, trained_at)
);
"""

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_posts_created_at ON posts(created_at);",
    "CREATE INDEX IF NOT EXISTS idx_posts_product_name ON posts(product_name);",
    "CREATE INDEX IF NOT EXISTS idx_posts_product_category ON posts(product_category);",
    "CREATE INDEX IF NOT EXISTS idx_product_daily_trend_score ON product_daily_metrics(trend_score DESC);",
    "CREATE INDEX IF NOT EXISTS idx_product_daily_metric_date ON product_daily_metrics(metric_date);",
]


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    settings = get_settings()
    conn = sqlite3.connect(settings.database_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def initialize_database() -> Path:
    settings = get_settings()
    with get_connection() as conn:
        conn.execute(POSTS_SCHEMA)
        conn.execute(PRODUCT_DAILY_SCHEMA)
        conn.execute(MODEL_REGISTRY_SCHEMA)
        for stmt in INDEXES:
            conn.execute(stmt)
    return settings.database_path
