from typing import Tuple, Dict, Any, Optional
from decimal import Decimal
import re
from dotenv import load_dotenv

from src.agent.config import Settings
from src.agent.controller import Controller
from src.agent.llm_client import LLMClient
from .memory import MemoryManager

load_dotenv()

# ---- Base settings & LLM (singletons) ----
SETTINGS_BASE = Settings.from_env()
LLM = LLMClient(
    api_key=SETTINGS_BASE.openai_api_key,
    base_url=SETTINGS_BASE.openai_base_url,
    model=SETTINGS_BASE.model_name,
)

# Small cache so we don't rebuild a Controller for the same index path repeatedly
_CTRL_CACHE: Dict[str, Controller] = {}

def _get_controller(index_dir: Optional[str] = None) -> Controller:
    """
    Return a Controller configured for the given index_dir (or the default one from env).
    Controllers are cached per index path to avoid reloading prompt files, etc.
    """
    idx = (index_dir or SETTINGS_BASE.index_dir) or ""
    if idx in _CTRL_CACHE:
        return _CTRL_CACHE[idx]
    ctrl = Controller(LLM, SETTINGS_BASE.tavily_api_key, idx)
    _CTRL_CACHE[idx] = ctrl
    # keep cache bounded (tiny LRU-ish)
    if len(_CTRL_CACHE) > 8:
        _CTRL_CACHE.pop(next(iter(_CTRL_CACHE)))
    return ctrl


PURE_NUM_RX = re.compile(r"^\s*-?\d+(?:\.\d+)?\s*$", re.M)

# --- memory-intent detection (regex-first; no LLM needed) ---

def _detect_memory_intent(text: str) -> Tuple[str, Dict[str, Any]] | None:
    t = text.strip().lower()
    # history of this thread
    if re.search(r"\b(history|what.*(we|i).*ask(ed)?|what did i say|what have i asked|conversation so far|recent messages)\b", t):
        return ("history", {})
    # just user questions recently
    if re.search(r"\b(what.*i.*asked recently|recent questions|recent queries)\b", t):
        return ("recent_user", {})
    # list stored facts / memory dump
    if re.search(r"\b(what.*data.*stored|what.*do.*you.*remember|show.*memory|list.*memory|what.*have.*you.*saved)\b", t):
        return ("list_memory", {})
    # clear thread memory
    if re.search(r"\b(clear|reset|forget).*(memory|context)\b", t):
        return ("clear_thread", {})
    # last number/result
    if re.search(r"\b(last (number|result)|what.*(was|is).*the result)\b", t):
        return ("last_number", {})
    return None


def _format_table(rows: list[list[str]], headers: list[str]) -> str:
    # simple fixed-width text table
    colw = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            colw[i] = max(colw[i], len(cell))
    def fmt_row(r): return " | ".join(cell.ljust(colw[i]) for i, cell in enumerate(r))
    sep = "-+-".join("-" * w for w in colw)
    out = [fmt_row(headers), sep]
    out += [fmt_row(r) for r in rows] if rows else ["(none)"]
    return "\n".join(out)


def _handle_memory_intent(intent: str, mem: MemoryManager) -> str:
    if intent == "history":
        rows = mem.recent_messages(limit=12)
        table = _format_table([[r, c, ts] for (r, c, ts) in rows], ["role", "content", "time"])
        return table
    if intent == "recent_user":
        rows = mem.recent_user_questions(limit=10)
        table = _format_table([[q, ts] for (q, ts) in rows], ["question", "time"])
        return table
    if intent == "list_memory":
        facts = mem.list_stored_facts()
        if not facts:
            return "I haven't stored any durable facts yet."
        table = _format_table([[ns, key, val] for (ns, key, val) in facts], ["namespace", "key", "value"])
        return table
    if intent == "clear_thread":
        n = mem.clear_thread_memory()
        return f"Cleared {n} thread-scoped memory item(s)."
    if intent == "last_number":
        val = (mem.get_last_number() or mem.get_best_followup_number())
        return str(val) if val is not None else "I don't have a recent numeric result saved."
    return "(unknown memory intent)"


def handle_chat(
    user_text: str,
    mem: MemoryManager,
    src_msg=None,
    override_index_dir: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Main bridge used by the Django view.
    - Detects quick memory intents (history, list_memory, etc.)
    - Applies memory-powered rewrites for numeric/fact follow-ups
    - Routes to the Controller (optionally using a per-session index_dir for RAG)
    - Captures numeric facts and durable facts when appropriate
    """
    # 0) Pure memory-management requests (no LLM calls)
    mi = _detect_memory_intent(user_text)
    if mi:
        intent, _ = mi
        resp = _handle_memory_intent(intent, mem)
        trace = [f"memory -> {intent}"]
        return resp, {"trace": trace, "rewritten": None}

    # 1) Track entities/topics from the raw query
    mem.capture_topic_from_query(user_text)

    # 2) Memory-powered rewrites (numeric + fact followups)
    rewritten = (
        mem.rewrite_numeric_followup(user_text, prefer_tool="calculator")
        or mem.rewrite_numeric_followup(user_text, prefer_tool="gsm8k")
        or mem.rewrite_numeric_followup(user_text)
        or mem.rewrite_fact_followup(user_text)
    )
    query = rewritten or user_text

    # 3) Controller with optional per-session index
    controller = _get_controller(override_index_dir)
    result = controller.orchestrate(query)
    final = result.get("final_answer", "") or ""
    trace = result.get("trace", [])

    # 4) Capture numeric memory only when final is a pure number
    if PURE_NUM_RX.match(final):
        tool_hint = "calculator" if any("calculator" in t for t in trace) else "gsm8k" if any("gsm8k" in t for t in trace) else None
        try:
            mem.set_last_number(Decimal(final), tool=tool_hint, src_msg=src_msg)
        except Exception:
            pass

    # 5) Store durable facts based on the query (web/rag)
    tool_hint_fact = "web" if any("web" in t for t in trace) else "rag" if any("rag" in t for t in trace) else None
    if tool_hint_fact:
        mem.maybe_store_fact_from_qa(query, final, tool=tool_hint_fact, src_msg=src_msg)

    return final, {"trace": trace, "rewritten": rewritten}
