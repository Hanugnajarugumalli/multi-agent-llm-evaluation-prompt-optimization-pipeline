from __future__ import annotations

from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

from pipeline.llm_factory import get_chat_model
from pipeline.state import AgentState

SYSTEM_STYLE = (
    "You write clear, factual summaries of U.S. legislative text for policy readers. "
    "Do not invent citations; stick to the document."
)


def _doc_block(state: AgentState) -> str:
    text = state.get("bill_text") or ""
    title = state.get("title") or ""
    return f"TITLE:\n{title}\n\nDOCUMENT:\n{text}"


def build_multi_agent_graph():
    llm = get_chat_model()

    def node_reason(state: AgentState) -> AgentState:
        prompt = (
            f"{SYSTEM_STYLE}\n\n"
            "Task: extract the main problem the bill addresses and 3–5 bullet-style facts "
            "a reader must know before summarizing. Be concise.\n\n"
            + _doc_block(state)
        )
        out = llm.invoke([HumanMessage(content=prompt)])
        reasoning = getattr(out, "content", str(out))
        return {"reasoning": reasoning}

    def node_summarize(state: AgentState) -> AgentState:
        prompt = (
            f"{SYSTEM_STYLE}\n\n"
            "Task: write a 1-paragraph summary of the bill. Use the reasoning notes to stay on-topic.\n\n"
            f"REASONING NOTES:\n{state.get('reasoning', '')}\n\n"
            + _doc_block(state)
        )
        out = llm.invoke([HumanMessage(content=prompt)])
        draft = getattr(out, "content", str(out))
        return {"draft_summary": draft}

    def node_decide(state: AgentState) -> AgentState:
        prompt = (
            f"{SYSTEM_STYLE}\n\n"
            "Task: decision support — in 3-5 sentences, say what a staffer should verify next "
            "(e.g., funding, effective dates, affected agencies) and whether the draft summary "
            "is safe to circulate internally.\n\n"
            f"DRAFT SUMMARY:\n{state.get('draft_summary', '')}\n\n"
            + _doc_block(state)
        )
        out = llm.invoke([HumanMessage(content=prompt)])
        note = getattr(out, "content", str(out))
        # glued together for logging; metrics use draft_summary only
        final = (state.get("draft_summary") or "").strip()
        if note:
            final = f"{final}\n\n[Decision support]: {note.strip()}"
        return {"decision_support": note, "final_summary": final}

    g = StateGraph(AgentState)
    g.add_node("reason", node_reason)
    g.add_node("summarize", node_summarize)
    g.add_node("decide", node_decide)
    g.set_entry_point("reason")
    g.add_edge("reason", "summarize")
    g.add_edge("summarize", "decide")
    g.add_edge("decide", END)
    return g.compile()
