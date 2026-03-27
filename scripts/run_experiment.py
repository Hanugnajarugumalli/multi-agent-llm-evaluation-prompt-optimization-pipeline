#!/usr/bin/env python3
# run from repo root: PYTHONPATH=src python scripts/run_experiment.py [--limit N]

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from pipeline import config
from pipeline.baselines.single_agent import run_single_agent_summary
from pipeline.evaluation.metrics import mean_scores, score_summary
from pipeline.graph.workflow import build_multi_agent_graph
from pipeline.tracking.mlflow_run import log_side_by_side_artifact


def load_jsonl(path: Path, limit: int | None):
    rows = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=config.DEFAULT_DATA)
    ap.add_argument("--limit", type=int, default=None, help="subset size; default = all lines in file")
    ap.add_argument("--run-name", type=str, default="billsum_eval")
    ap.add_argument("--skip-mlflow", action="store_true")
    args = ap.parse_args()

    data = load_jsonl(args.data, args.limit)
    n = len(data)
    if n == 0:
        print("No rows loaded; check --data path.", file=sys.stderr)
        sys.exit(1)

    graph = build_multi_agent_graph()
    multi_scores = []
    single_scores = []
    artifact_rows = []
    t0 = time.perf_counter()
    for row in data:
        bill_id = row.get("bill_id", "")
        title = row.get("title", "")
        text = row.get("clean_text") or row.get("text") or ""
        text = text[: config.MAX_DOC_CHARS]
        ref = row.get("summary", "")

        s_multi = graph.invoke(
            {
                "bill_id": bill_id,
                "title": title,
                "bill_text": text,
                "reference_summary": ref,
            }
        )
        draft = (s_multi.get("draft_summary") or "").strip()

        pred_single = run_single_agent_summary(title=title, bill_text=text)

        sm = score_summary(draft, ref)
        ss = score_summary(pred_single, ref)
        multi_scores.append(sm)
        single_scores.append(ss)
        artifact_rows.append(
            {
                "bill_id": bill_id,
                "rougeL_multi": sm.rougeL,
                "rougeL_single": ss.rougeL,
                "relevance_multi": sm.relevance_f1,
                "relevance_single": ss.relevance_f1,
            }
        )

    elapsed = time.perf_counter() - t0
    mm = mean_scores(multi_scores)
    ms = mean_scores(single_scores)

    params = {
        "num_queries": n,
        "openai_model": config.OPENAI_MODEL,
        "mock_llm": config.USE_MOCK_LLM or not config.OPENAI_API_KEY,
        "max_doc_chars": config.MAX_DOC_CHARS,
        "data_path": str(args.data),
    }
    metrics = {
        "mean_rougeL_multi": mm["rougeL"],
        "mean_rougeL_single": ms["rougeL"],
        "mean_relevance_f1_multi": mm["relevance_f1"],
        "mean_relevance_f1_single": ms["relevance_f1"],
        "mean_rouge1_multi": mm["rouge1"],
        "mean_rouge1_single": ms["rouge1"],
        "seconds_total": elapsed,
        "seconds_per_query": elapsed / max(n, 1),
    }

    print(json.dumps({"params": params, "metrics": metrics}, indent=2))

    if not args.skip_mlflow:
        import mlflow

        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(config.EXPERIMENT_NAME)
        with mlflow.start_run(run_name=args.run_name):
            for k, v in params.items():
                mlflow.log_param(k, v)
            for k, v in metrics.items():
                mlflow.log_metric(k, float(v))
            mlflow.set_tags({"setup": "multi_vs_single"})
            log_side_by_side_artifact(artifact_rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
