from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class Settings:
    project_root: Path = Path(__file__).resolve().parents[2]
    database_path: Path = project_root / "artifacts" / "sniper_tok.db"
    model_path: Path = project_root / "artifacts" / "category_model.joblib"
    metadata_path: Path = project_root / "artifacts" / "model_metrics.json"


def get_settings() -> Settings:
    settings = Settings()
    settings.database_path.parent.mkdir(parents=True, exist_ok=True)
    settings.model_path.parent.mkdir(parents=True, exist_ok=True)
    return settings
