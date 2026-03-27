from __future__ import annotations

import os
from pathlib import Path

# two dirs up from here = repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = REPO_ROOT / "billsum_v4_1" / "us_test_data_final_OFFICIAL.jsonl"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
USE_MOCK_LLM = os.environ.get("USE_MOCK_LLM", "").lower() in ("1", "true", "yes")
if not OPENAI_API_KEY:
    USE_MOCK_LLM = True

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "bill_summarization_agents")

MAX_DOC_CHARS = int(os.environ.get("MAX_DOC_CHARS", "12000"))
