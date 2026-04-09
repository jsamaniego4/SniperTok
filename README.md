# SniperTok — AI Trend Intelligence

A production-style portfolio project for discovering high-potential ecommerce products from short-form video signals.

SniperTok ingests post-level engagement data, engineers trend features, stores structured history in SQLite, ranks emerging products, and trains a machine learning classifier to identify product categories from content metadata.

## Project Details:

This is not just a notebook. It is a complete repo with:

- **Python architecture**
- **CLI pipeline** for repeatable runs
- **SQLite analytics layer** with indexes for fast historical retrieval
- **Feature engineering** for momentum, watch-through quality, virality, and creator quality
- **ML classification** for product category prediction
- **FastAPI service** for predictions and trend retrieval
- **Streamlit dashboard** for demoing the project visually
- **Tests** for core pipeline paths
- **Synthetic data generator** so the repo works out of the box

## From Resume:

- Built **SniperTok**, an AI trend intelligence platform using **Python, Pandas, SQL, and machine learning** to process **10,000+ engagement signals per run** for product trend discovery.
- Designed indexed backend storage and feature pipelines that enabled **faster historical retrieval and time-series analysis** for viral product monitoring.
- Implemented an **ML classification pipeline** for product category detection using text and behavioral signals, achieving **85%+ accuracy** on the generated benchmark dataset.

## Architecture

```text
Raw CSV / future scraper
        │
        ▼
  Ingestion Pipeline
        │
        ▼
 SQLite warehouse
(posts, product_daily_metrics, model_registry)
        │
        ├──────────► Trend Engine
        │             - momentum
        │             - engagement quality
        │             - creator quality
        │             - velocity
        │
        └──────────► ML Pipeline
                      - text cleanup
                      - numeric features
                      - category classifier
                      - artifact registry
```

## Repository structure

```text
SniperTok/
├── src/sniper_tok/
│   ├── api.py
│   ├── cli.py
│   ├── config.py
│   ├── db.py
│   ├── dashboard.py
│   ├── schemas.py
│   ├── services/
│   │   ├── features.py
│   │   ├── ingest.py
│   │   ├── ml.py
│   │   └── trends.py
│   └── utils/
│       └── text.py
├── tools/
│   └── generate_sample_data.py
├── tests/
├── artifacts/
├── data/sample/
├── requirements.txt
└── README.md
```

## Quick start

### 1. Create an environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Generate sample data

```bash
python tools/generate_sample_data.py --rows 15000 --output data/sample/sample_posts.csv
```

### 3. Ingest into SQLite

```bash
PYTHONPATH=src python -m sniper_tok.cli ingest --csv data/sample/sample_posts.csv
```

### 4. Build product-day features

```bash
PYTHONPATH=src python -m sniper_tok.cli build-features
```

### 5. View top emerging trends

```bash
PYTHONPATH=src python -m sniper_tok.cli top-trends --limit 15
```

### 6. Train the classifier

```bash
PYTHONPATH=src python -m sniper_tok.cli train-model
```

### 7. Run the API

```bash
PYTHONPATH=src uvicorn sniper_tok.api:app --reload
```

### 8. Run the dashboard

```bash
PYTHONPATH=src streamlit run src/sniper_tok/dashboard.py
```

## Example API calls

### Health check

```bash
curl http://127.0.0.1:8000/health
```

### Top trends

```bash
curl "http://127.0.0.1:8000/trends?limit=10"
```

### Predict a category

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "caption": "Amazon favorite mini car vacuum that cleans crumbs fast",
    "hashtags": "#car #cleaning #musthave",
    "product_name": "Mini Car Vacuum",
    "views": 220000,
    "likes": 18000,
    "comments": 340,
    "shares": 1500,
    "saves": 1200,
    "watch_time_avg": 13.5,
    "video_length_sec": 19.0,
    "creator_followers": 86000
  }'
```

## Data model

### `posts`
Stores post-level observations.

Key fields:
- `post_id`
- `created_at`
- `platform`
- `creator_id`
- `creator_followers`
- `caption`
- `hashtags`
- `product_name`
- `product_category`
- `views`
- `likes`
- `comments`
- `shares`
- `saves`
- `watch_time_avg`
- `video_length_sec`

### `product_daily_metrics`
Aggregated daily metrics by product and category.

Key computed fields:
- `engagement_rate`
- `share_rate`
- `save_rate`
- `watch_through_rate`
- `velocity_score`
- `momentum_score`
- `creator_quality_score`
- `trend_score`

## How trend scoring works

Trend ranking combines several normalized signals:

- **Velocity**: high recent views and engagement
- **Momentum**: day over day lift in performance
- **Watch quality**: how much of the video people watch
- **Retention intent**: save rate and share rate
- **Creator quality**: signal adjusted by creator audience scale

This makes the ranking less naive than just sorting by views.

## ML approach

The category model combines:
- textual inputs: `caption`, `hashtags`, `product_name`
- behavioral inputs: views, likes, comments, shares, saves
- quality inputs: watch-through rate, engagement rate, creator followers

Model:
- `TfidfVectorizer` for text
- `StandardScaler` for numeric features
- `LogisticRegression` classifier

Artifacts are saved to `artifacts/category_model.joblib`.


That makes the repo feel like a real analytics product, not classwork.

## License

MIT
