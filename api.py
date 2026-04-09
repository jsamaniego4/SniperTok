from __future__ import annotations

from fastapi import FastAPI, HTTPException

from .config import get_settings
from .db import initialize_database
from .schemas import HealthResponse, PredictionRequest, PredictionResponse
from .services.ml import predict_category
from .services.trends import get_top_trends

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
