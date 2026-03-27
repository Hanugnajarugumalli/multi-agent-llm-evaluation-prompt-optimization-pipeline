from __future__ import annotations

from typing import TypedDict


class AgentState(TypedDict, total=False):
    bill_id: str
    title: str
    bill_text: str
    reference_summary: str
    reasoning: str
    draft_summary: str
    final_summary: str
    decision_support: str
