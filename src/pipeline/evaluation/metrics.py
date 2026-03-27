from __future__ import annotations

import re
from dataclasses import dataclass

from rouge_score import rouge_scorer


def _normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def token_f1(prediction: str, reference: str) -> float:
    # word overlap f1, not fancy
    pt = set(_normalize(prediction).split())
    rt = set(_normalize(reference).split())
    if not pt or not rt:
        return 0.0
    inter = len(pt & rt)
    precision = inter / len(pt)
    recall = inter / len(rt)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


@dataclass
class SummaryScores:
    rouge1: float
    rouge2: float
    rougeL: float
    relevance_f1: float


_scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def score_summary(prediction: str, reference: str) -> SummaryScores:
    ref = reference.strip()
    pred = prediction.strip()
    if not pred:
        return SummaryScores(0.0, 0.0, 0.0, 0.0)
    r = _scorer.score(ref, pred)
    return SummaryScores(
        rouge1=r["rouge1"].fmeasure,
        rouge2=r["rouge2"].fmeasure,
        rougeL=r["rougeL"].fmeasure,
        relevance_f1=token_f1(pred, ref),
    )


def mean_scores(rows: list[SummaryScores]) -> dict[str, float]:
    if not rows:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "relevance_f1": 0.0}
    n = len(rows)
    return {
        "rouge1": sum(x.rouge1 for x in rows) / n,
        "rouge2": sum(x.rouge2 for x in rows) / n,
        "rougeL": sum(x.rougeL for x in rows) / n,
        "relevance_f1": sum(x.relevance_f1 for x in rows) / n,
    }
