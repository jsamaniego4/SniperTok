from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException

# Ensure `src` is on sys.path when running the API directly.
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from sniper_tok.config import get_settings
from sniper_tok.db import initialize_database
from sniper_tok.schemas import HealthResponse, PredictionRequest, PredictionResponse
from sniper_tok.services.ml import predict_category
from sniper_tok.services.trends import get_top_trends

app = FastAPI(
    title="SniperTok API",
    version="1.0.0",
    description="API for social commerce trend intelligence and product category classification.",
)


@app.on_event("startup")
def startup_event() -> None:
    initialize_database()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        database_ready=settings.database_path.exists(),
        model_ready=settings.model_path.exists(),
    )


@app.get("/trends")
def trends(limit: int = 10):
    df = get_top_trends(limit=limit)
    return df.to_dict(orient="records")


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        label, confidence = predict_category(request.model_dump())
        return PredictionResponse(predicted_category=label, confidence=confidence)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
