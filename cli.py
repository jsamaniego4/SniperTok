from __future__ import annotations

import argparse
import json

from .db import initialize_database
from .services.features import build_daily_product_features
from .services.ingest import ingest_csv
from .services.ml import train_category_model
from .services.trends import get_top_trends


def handle_ingest(args: argparse.Namespace) -> None:
    rows = ingest_csv(args.csv)
    print(f"Ingested {rows} rows into SQLite.")


def handle_build_features(args: argparse.Namespace) -> None:
    df = build_daily_product_features()
    print(f"Built {len(df)} product-day feature rows.")


def handle_top_trends(args: argparse.Namespace) -> None:
    df = get_top_trends(limit=args.limit)
    if df.empty:
        print("No trend data found. Run build-features first.")
        return
    display_cols = [
        "metric_date", "product_name", "product_category",
        "trend_score", "total_views", "engagement_rate", "watch_through_rate"
    ]
    print(df[display_cols].to_string(index=False))


def handle_train_model(args: argparse.Namespace) -> None:
    metrics = train_category_model()
    print(json.dumps({
        "accuracy": round(metrics["accuracy"], 4),
        "macro_f1": round(metrics["macro_f1"], 4),
        "rows_used": metrics["rows_used"],
        "classes": metrics["classes"],
    }, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SniperTok CLI")
    subparsers = parser.add_subparsers(required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest a CSV into SQLite")
    ingest_parser.add_argument("--csv", required=True, help="Path to source CSV")
    ingest_parser.set_defaults(func=handle_ingest)

    feature_parser = subparsers.add_parser("build-features", help="Build daily trend features")
    feature_parser.set_defaults(func=handle_build_features)

    trends_parser = subparsers.add_parser("top-trends", help="List top product trends")
    trends_parser.add_argument("--limit", type=int, default=10)
    trends_parser.set_defaults(func=handle_top_trends)

    train_parser = subparsers.add_parser("train-model", help="Train product category classifier")
    train_parser.set_defaults(func=handle_train_model)

    init_parser = subparsers.add_parser("init-db", help="Initialize the SQLite warehouse")
    init_parser.set_defaults(func=lambda _: print(f"Database ready at {initialize_database()}"))

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
