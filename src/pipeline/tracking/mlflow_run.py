from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import mlflow

from pipeline import config


def start_run(run_name: str, params: dict[str, Any], metrics: dict[str, float], tags: dict[str, str] | None = None):
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENT_NAME)
    with mlflow.start_run(run_name=run_name) as run:
        for k, v in params.items():
            mlflow.log_param(k, v)
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))
        if tags:
            mlflow.set_tags(tags)
        return run.info.run_id


def log_side_by_side_artifact(rows: list[dict[str, Any]], filename: str = "per_query_scores.jsonl"):
    lines = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / filename
        p.write_text(lines, encoding="utf-8")
        mlflow.log_artifact(str(p))
