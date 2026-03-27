from __future__ import annotations

from langchain_core.messages import HumanMessage

from pipeline.llm_factory import get_chat_model


def run_single_agent_summary(*, title: str, bill_text: str) -> str:
    llm = get_chat_model()
    prompt = (
        "Summarize the following legislative bill in one paragraph for a policy audience. "
        "Cover main purpose, key mechanisms, and who is affected.\n\n"
        f"TITLE:\n{title}\n\nBILL TEXT:\n{bill_text}"
    )
    out = llm.invoke([HumanMessage(content=prompt)])
    return getattr(out, "content", str(out)).strip()
