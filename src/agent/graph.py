from typing import TypedDict, List
from langgraph.graph import StateGraph, END

from src.agent.config import Settings
from src.agent.llm_client import LLMClient
from src.agent.controller import Controller


class AgentState(TypedDict, total=False):
    input: str
    final_answer: str
    trace: List[str]


def build_graph(settings: Settings):
    # LLM client (Groq via OpenAI-compatible endpoint if OPENAI_BASE_URL is set)
    llm = LLMClient(
        api_key=settings.openai_api_key,  # type: ignore
        base_url=settings.openai_base_url,
        model=settings.model_name,
    )
    controller = Controller(llm, settings.tavily_api_key, settings.index_dir)

    # ---------------- Nodes ----------------
    def node_orchestrate(state: AgentState) -> AgentState:
        result = controller.orchestrate(state["input"])  # type: ignore
        trace = state.get("trace", []) + result.get("trace", [])
        return {**state, "final_answer": result["final_answer"], "trace": trace}

    # ---------------- Graph ----------------
    graph = StateGraph(AgentState)
    graph.add_node("orchestrate", node_orchestrate)
    graph.set_entry_point("orchestrate")
    graph.add_edge("orchestrate", END)

    return graph.compile()
