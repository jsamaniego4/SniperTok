from __future__ import annotations

import json
from datetime import datetime
from typing import Tuple

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import get_settings
from ..db import get_connection, initialize_database
from ..utils.text import merge_text_fields


TEXT_COL = "combined_text"
NUMERIC_COLS = [
    "views", "likes", "comments", "shares", "saves",
    "watch_time_avg", "video_length_sec", "creator_followers",
    "engagement_rate", "watch_through_rate",
]


def _load_training_frame() -> pd.DataFrame:
    initialize_database()
    with get_connection() as conn:
        df = pd.read_sql_query("SELECT * FROM posts", conn)

    if df.empty:
        raise ValueError("No training data found. Run ingestion first.")

    df["engagement_rate"] = (
        (df["likes"] + df["comments"] + df["shares"] + df["saves"]) / df["views"].replace(0, 1)
    )
    df["watch_through_rate"] = df["watch_time_avg"] / df["video_length_sec"].replace(0, 1)
    df[TEXT_COL] = df.apply(
        lambda row: merge_text_fields(row["caption"], row["hashtags"], row["product_name"]),
        axis=1,
    )
    return df


def train_category_model(test_size: float = 0.2, random_state: int = 42) -> dict:
    df = _load_training_frame()

    X = df[[TEXT_COL] + NUMERIC_COLS].copy()
    y = df["product_category"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(max_features=3000, ngram_range=(1, 2)), TEXT_COL),
            ("num", Pipeline([("scale", StandardScaler())]), NUMERIC_COLS),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1200)),
        ]
    )

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    accuracy = float(accuracy_score(y_test, preds))
    macro_f1 = float(f1_score(y_test, preds, average="macro"))

    settings = get_settings()
    joblib.dump(pipeline, settings.model_path)

    metrics = {
        "trained_at": datetime.utcnow().isoformat(),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "rows_used": int(len(df)),
        "classes": sorted(y.unique().tolist()),
        "classification_report": classification_report(y_test, preds, output_dict=True),
    }

    with settings.metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO model_registry (model_name, trained_at, accuracy, macro_f1, rows_used)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("category_model", metrics["trained_at"], accuracy, macro_f1, len(df)),
        )

    return metrics


def load_model():
    settings = get_settings()
    if not settings.model_path.exists():
        raise FileNotFoundError("Model artifact not found. Train the model first.")
    return joblib.load(settings.model_path)


def predict_category(payload: dict) -> Tuple[str, float]:
    model = load_model()

    views = max(int(payload["views"]), 1)
    engagement_rate = (
        payload["likes"] + payload["comments"] + payload["shares"] + payload["saves"]
    ) / views
    watch_through_rate = payload["watch_time_avg"] / max(float(payload["video_length_sec"]), 1.0)

    row = pd.DataFrame([{
        TEXT_COL: merge_text_fields(payload["caption"], payload["hashtags"], payload["product_name"]),
        "views": payload["views"],
        "likes": payload["likes"],
        "comments": payload["comments"],
        "shares": payload["shares"],
        "saves": payload["saves"],
        "watch_time_avg": payload["watch_time_avg"],
        "video_length_sec": payload["video_length_sec"],
        "creator_followers": payload["creator_followers"],
        "engagement_rate": engagement_rate,
        "watch_through_rate": watch_through_rate,
    }])

    label = model.predict(row)[0]
    probabilities = model.predict_proba(row)[0]
    confidence = float(probabilities.max())
    return label, confidence
