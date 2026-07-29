from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

@dataclass(frozen=True)
class Settings:
    project_root: Path
    database_path: Path
    model_path: Path
    metadata_path: Path


def get_settings() -> Settings:
    project_root = Path(__file__).resolve().parents[2]
    artifacts_dir = project_root / "artifacts"

    database_path = Path(
        os.environ.get("SNIPER_TOK_DATABASE_PATH", str(artifacts_dir / "sniper_tok.db"))
    )
    model_path = Path(
        os.environ.get("SNIPER_TOK_MODEL_PATH", str(artifacts_dir / "category_model.joblib"))
    )
    metadata_path = Path(
        os.environ.get("SNIPER_TOK_METADATA_PATH", str(artifacts_dir / "model_metrics.json"))
    )

    database_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    return Settings(
        project_root=project_root,
        database_path=database_path,
        model_path=model_path,
        metadata_path=metadata_path,
    )