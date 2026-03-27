from __future__ import annotations

import re
from typing import Any

from pipeline import config


def _heuristic_reasoning(title: str, text: str) -> str:
    snippet = re.sub(r"\s+", " ", text)[:900]
    return (
        f"Key focus (from title): {title.strip()[:200]}\n"
        f"Early sections mention: {snippet[:400]}...\n"
        f"Notes: identify main statutory change and who it affects."
    )


def _heuristic_summary(title: str, text: str) -> str:
    clean = re.sub(r"\s+", " ", text)
    head = clean[:1200]
    tail = clean[-400:] if len(clean) > 1600 else ""
    parts = [p for p in (head, tail) if p]
    body = " ... ".join(parts)
    return f"{title.strip()} — {body[:1800]}".strip()


class MockChat:
    # fake responses so imports work without paying for API
    def invoke(self, messages: list[Any]) -> Any:
        last = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])
        low = last.lower()
        text = _extract_doc(last)
        title = _extract_title(last)
        if "reasoning notes" in low and "1-paragraph" in low:
            out = _heuristic_summary(title, text)
        elif "decision support" in low or "staffer should verify" in low:
            out = (
                "Decision support: prioritize checking fiscal impact and implementation date. "
                f"Document length ~{len(text)} chars; skim sections titled SEC. for operative text."
            )
        elif "main problem" in low or "bullet-style" in low:
            out = _heuristic_reasoning(title, text)
        else:
            out = _heuristic_summary(title, text)

        class R:
            content = out

        return R()


def _extract_doc(prompt: str) -> str:
    m = re.search(r"DOCUMENT:\s*(.*)", prompt, re.DOTALL | re.IGNORECASE)
    return (m.group(1) if m else prompt).strip()


def _extract_title(prompt: str) -> str:
    m = re.search(r"TITLE:\s*([^\n]+)", prompt, re.IGNORECASE)
    return m.group(1).strip() if m else ""


def get_chat_model():
    if config.USE_MOCK_LLM or not config.OPENAI_API_KEY:
        return MockChat()
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=config.OPENAI_MODEL,
        temperature=0.2,
        api_key=config.OPENAI_API_KEY,
    )
