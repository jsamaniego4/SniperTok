from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sniper_tok.services.ingest import ingest_csv
from sniper_tok.services.features import build_daily_product_features
from sniper_tok.services.ml import train_category_model
from tools.generate_sample_data import generate_dataset


def test_end_to_end_pipeline(tmp_path):
    csv_path = tmp_path / "sample.csv"
    df = generate_dataset(rows=500, seed=7)
    df.to_csv(csv_path, index=False)

    inserted = ingest_csv(str(csv_path))
    assert inserted == 500

    features = build_daily_product_features()
    assert not features.empty
    assert "trend_score" in features.columns

    metrics = train_category_model(test_size=0.25, random_state=7)
    assert metrics["accuracy"] >= 0.70
    assert metrics["rows_used"] == 500
