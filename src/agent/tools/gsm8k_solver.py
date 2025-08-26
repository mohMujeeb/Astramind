from typing import Dict, Any
from src.agent.llm_client import LLMClient
import re

SYSTEM = """You are a careful math tutor. Solve grade-school math word problems.
Show minimal reasoning in 2-5 short steps. At the end, output only the final numeric answer on a single line prefixed by 'ANSWER: '.

Disambiguation rules:
- Interpret phrases like 'N times more than X' as 'N times as many as X' (multiply N * X), NOT N*X + X.
- For 'empty seats now' type questions: if total seats S and initially occupied O are given, and 'get on' (+A) and 'get off' (-B) events occur,
  then occupied_after = O + A - B and empty = S - occupied_after (bounded between 0 and S).
- For 'how long at the same speed' problems: if speed is given implicitly by U distance in V time, then time for distance D is D / (U/V).
"""

PROMPT = """Problem:
{question}

Solve step-by-step, but be concise. Then write the final numeric answer like:
ANSWER: 1234
"""

RETRY_SYSTEM = "You must output EXACTLY one line in the form 'ANSWER: <number>' with no other text."
RETRY_PROMPT = """Problem:
{question}

Return only one line: ANSWER: <number>"""

# --- normalization helpers ---

_TIMES_MORE_RX = re.compile(r"\b(\d+)\s+times\s+more\s+than\b", re.IGNORECASE)
_TWICE_MORE_RX = re.compile(r"\btwice\s+more\s+than\b", re.IGNORECASE)
_THRICE_MORE_RX = re.compile(r"\bthrice\s+more\s+than\b", re.IGNORECASE)

def _normalize_phrasing(q: str) -> str:
    """Normalize ambiguous 'times more than' phrasing to 'times as many as' (N * X)."""
    s = q
    s = _TIMES_MORE_RX.sub(r"\1 times as many as", s)
    s = _TWICE_MORE_RX.sub("2 times as many as", s)
    s = _THRICE_MORE_RX.sub("3 times as many as", s)
    return s

def _extract_answer_line(text: str) -> str | None:
    for line in (text or "").splitlines():
        if line.strip().upper().startswith("ANSWER:"):
            return line.split(":", 1)[1].strip()
    return None

def _last_number(text: str) -> str | None:
    nums = re.findall(r"-?\d+(?:\.\d+)?", text or "")
    return nums[-1] if nums else None

def _fmt_num(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else str(x)

# --- deterministic solvers for common patterns ---

def _bus_empty_seats_solver(q: str) -> str | None:
    """
    If the problem describes a bus with S seats, O initially occupied,
    A people get on, and B get off, compute empty = S - (O + A - B).
    """
    text = q.lower()

    def grab(pat: str) -> int | None:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
        return None

    seats   = grab(r"\b(\d+)\s+seats?\b")
    occupied= grab(r"\b(\d+)\s+(?:are\s+)?occupied(?:\s+at\s+the\s+start)?\b")
    got_on  = grab(r"\b(\d+)\s+(?:people\s+)?get\s+on\b")
    got_off = grab(r"\b(\d+)\s+(?:people\s+)?get\s+off\b")

    if seats is None or occupied is None or got_on is None or got_off is None:
        return None

    occ_after = occupied + got_on - got_off
    empty = seats - occ_after
    empty = max(0, min(seats, empty))
    return str(empty)

def _uniform_rate_time_solver(q: str) -> str | None:
    """
    Detect: "[agent] travels U (km/miles) in V hour(s/min)", then "how long ... D (km/miles)?" at same speed.
    Compute time = D / (U/V). Units are assumed consistent; we don't convert km<->miles.
    """
    text = q.lower()

    # base: travels U ... in V hour(s/minute[s])
    m1 = re.search(
        r"travels?\s+(\d+(?:\.\d+)?)\s*(km|kilometer|kilometers|mile|miles)\s+in\s+(\d+(?:\.\d+)?)\s*(hour|hours|hr|h|minute|minutes|min|m)\b",
        text, re.IGNORECASE)
    # target distance: how long ... D (km/miles)
    m2 = re.search(
        r"how\s+long.*?\b(\d+(?:\.\d+)?)\s*(km|kilometer|kilometers|mile|miles)\b",
        text, re.IGNORECASE)

    if not (m1 and m2):
        return None

    U = float(m1.group(1))
    V = float(m1.group(3))
    # convert minutes to hours if needed for V
    unit_time = m1.group(4)
    if unit_time.startswith("min") or unit_time.startswith("m"):
        V = V / 60.0

    D = float(m2.group(1))
    # assume same distance units, so time in hours
    speed = U / V  # distance per hour
    t_hours = D / speed
    return _fmt_num(t_hours)

def solve_with_llm(client: LLMClient, question: str) -> Dict[str, Any]:
    # 1) Deterministic fixes first
    heur = _bus_empty_seats_solver(question)
    if heur is not None:
        return {"reasoning": "Applied deterministic bus-seats rule.", "final": heur}

    heur = _uniform_rate_time_solver(question)
    if heur is not None:
        return {"reasoning": "Applied deterministic uniform-rate time rule.", "final": heur}

    # 2) Otherwise, normalize ambiguity and use the LLM
    normalized_q = _normalize_phrasing(question)

    # First attempt: standard prompt with short CoT
    msg = client.chat([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": PROMPT.format(question=normalized_q)},
    ], temperature=0.0)

    final = _extract_answer_line(msg)

    # If missing or malformed, do a strict retry to force `ANSWER: <number>`
    if not final or not re.fullmatch(r"-?\d+(?:\.\d+)?", final):
        retry = client.chat([
            {"role": "system", "content": RETRY_SYSTEM},
            {"role": "user", "content": RETRY_PROMPT.format(question=normalized_q)},
        ], temperature=0.0)
        forced = _extract_answer_line(retry)
        if forced and re.fullmatch(r"-?\d+(?:\.\d+)?", forced):
            final = forced
        else:
            # Last resort: scrape the last numeric token from the first pass
            final = final or _last_number(msg)

    return {"reasoning": msg, "final": final}
