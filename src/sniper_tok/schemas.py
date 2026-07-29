from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    caption: str = Field(..., description="Video caption")
    hashtags: str = Field(default="", description="Hashtags string")
    product_name: str = Field(..., description="Detected product name")
    views: int = Field(..., ge=0)
    likes: int = Field(..., ge=0)
    comments: int = Field(..., ge=0)
    shares: int = Field(..., ge=0)
    saves: int = Field(..., ge=0)
    watch_time_avg: float = Field(..., ge=0)
    video_length_sec: float = Field(..., gt=0)
    creator_followers: int = Field(..., ge=0)


class PredictionResponse(BaseModel):
    predicted_category: str
    confidence: float


class HealthResponse(BaseModel):
    status: str
    database_ready: bool
    model_ready: bool
