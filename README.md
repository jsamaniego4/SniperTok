# SniperTok — AI Trend Intelligence

A production-style portfolio project for discovering high-potential ecommerce products from short-form video signals.

SniperTok ingests post-level engagement data, engineers trend features, stores structured history in SQLite, ranks emerging products, and trains a machine learning classifier to identify product categories from content metadata.

## What’s improved

This repo now includes:

- A clean package layout under `src/sniper_tok`
- A reusable CLI pipeline for ingestion, feature engineering, trend ranking, and model training
- A Streamlit dashboard that combines trend exploration, historical product tracking, and live category prediction
- A local FastAPI service for health, trend retrieval, and category inference
- A synthetic data generator to bootstrap the pipeline immediately
- A development-ready `requirements.txt` for reproducible installs

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
        └──────────► ML Pipeline
                      - text cleanup
                      - numeric features
                      - category classifier
                      - artifact registry
```

## Repository structure

```text
src/sniper_tok/
├── api.py
├── cli.py
├── config.py
├── dashboard.py
├── db.py
├── schemas.py
├── services/
│   ├── features.py
│   ├── ingest.py
│   ├── ml.py
│   └── trends.py
└── utils/
    └── text.py

tools/
└── generate_sample_data.py

tests/
└── test_pipeline.py

artifacts/
└── (generated model, metrics, and SQLite warehouse)

requirements.txt
README.md
```

## Quick start

### 1. Create an environment

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
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

## Dashboard features

The Streamlit dashboard now includes:

- Status cards for local artifacts
- Top trend snapshot and category distribution
- Product-level trend history charts
- Live prediction form for inference samples
- Model metrics preview with full JSON expansion

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

Artifacts are saved to `artifacts/category_model.joblib`
