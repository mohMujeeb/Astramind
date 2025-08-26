from typing import Dict, Any, Tuple, List
from pathlib import Path
import re

from src.agent.llm_client import LLMClient
from src.agent.tools import calculator as calc
from src.agent.tools import gsm8k_solver
from src.agent.tools.web_search import web_search
from src.agent.tools.rag import retrieve, answer_with_contexts


class Controller:
    """
    JSON-planning controller with guardrails and a strong rule-based fallback.
    """
    def __init__(self, llm: LLMClient, tavily_key: str | None, index_dir: str) -> None:
        self.llm = llm
        self.tavily_key = tavily_key
        self.index_dir = index_dir
        self.controller_prompt = self._load_controller_prompt()

    # ---------- Public ----------

    def orchestrate(self, question: str) -> Dict[str, Any]:
        plan = self.make_plan(question)
        if not self.validate_plan(question, plan):
            plan = self.fallback_plan(question)
        step_results, trace = self.execute_plan(plan)
        final = self.render_final(plan, step_results)
        return {"final_answer": final, "trace": trace}

    # ---------- Planning ----------

    def _load_controller_prompt(self) -> str:
        p = Path(__file__).parent / "prompts" / "controller_prompt.md"
        return p.read_text(encoding="utf-8")

    def make_plan(self, user_query: str) -> Dict[str, Any]:
        sys = "You are a planning controller that outputs ONLY JSON per the instructions."
        prompt = self.controller_prompt.replace("{{USER_QUERY}}", user_query)
        # If your LLMClient has chat_json, use it; otherwise fallback to deterministic plan below.
        try:
            return self.llm.chat_json(
                [
                    {"role": "system", "content": sys},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
        except Exception:
            return {"plan": [], "final_response_instructions": ""}

    def validate_plan(self, user_query: str, plan_obj: Dict[str, Any]) -> bool:
        if "plan" not in plan_obj or not isinstance(plan_obj["plan"], list):
            return False
        if len(plan_obj["plan"]) == 0 or len(plan_obj["plan"]) > 4:
            return False

        q = user_query.lower()
        query_keywords = set(re.findall(r"[a-zA-Z]{3,}", q))

        def looks_related(text: str) -> bool:
            text_l = (text or "").lower()
            words = set(re.findall(r"[a-zA-Z]{3,}", text_l))
            return len(query_keywords.intersection(words)) > 0

        allowed_tools = {"calculator", "gsm8k", "web_search", "web", "rag"}
        banned_inputs = [
            "current president of the united states",
            "practice problem",
            "tom has",
            "the word \"also\"",
        ]

        for step in plan_obj["plan"]:
            tool = (step.get("tool") or "").lower().strip()
            if tool not in allowed_tools:
                return False
            inp = (step.get("input") or "").strip()
            if not looks_related(inp):
                return False
            if any(b in inp.lower() for b in banned_inputs):
                return False

        if not isinstance(plan_obj.get("final_response_instructions", ""), str):
            return False
        return True

    # ---------- Fallback (deterministic) ----------

    def fallback_plan(self, user_query: str) -> Dict[str, Any]:
        parts = self._split_query(user_query)
        steps: List[Dict[str, Any]] = []
        for i, p in enumerate(parts, start=1):
            tool, normalized_input = self._route_part(p)
            steps.append({
                "id": f"step-{i}",
                "tool": tool,
                "input": normalized_input,
                "depends_on": []
            })
        template = " | ".join([f"{{{{{s['id']}}}}}" for s in steps])
        return {"plan": steps, "final_response_instructions": template, "notes_on_false_premises": ""}

    # ---------- Splitting ----------

    def _split_query(self, q: str) -> List[str]:
        """
        1) Split by '?' first to keep full questions together.
        2) Extract a LEADING pure math expression (e.g., 'Compute 10!/(2^3)') as its own part.
        3) If a segment looks GSM8K-style (digits + cues like 'how many/left/total/…'), keep WHOLE segment.
        4) Otherwise split on 'and', '&', semicolons, commas between clauses,
           BUT do NOT split on the comma after a doc-reference preface (e.g., 'according to our local docs, ...').
        """
        q = (q or "").strip()
        q = re.sub(r"\b(also|then|and)\b\s*[,;]\s*", r"\1 ", q, flags=re.I)

        segments = [s.strip() for s in re.split(r"\?\s*", q) if s.strip()]
        parts: List[str] = []

        gsm_pattern = re.compile(
            r"(how\s+(?:many|much|long)|left|remain|time|speed|distance|rate|each|per|total|altogether|spent|earn|cost|bought|sold|gave|times|twice|thrice|more than|less than)",
            re.I,
        )

        doc_preface_comma = re.compile(
            r"(\b(?:according to|per|from)\s+(?:our|the|this)\s+(?:local\s+)?(?:docs?|document|knowledge\s*base|kb|notes|pdf))\s*,\s*",
            re.I
        )

        def extract_leading_calc(s: str) -> Tuple[List[str], str]:
            """
            If the segment starts with 'compute/calculate/evaluate ...<EXPR>...' capture the
            leading pure math expression as its own chunk and return (chunks, remainder).
            """
            # Normalize leading verbs
            s2 = s.strip()
            lead = re.compile(r"^\s*(?:compute|calculate|evaluate|what\s+is|what's)?\s*", re.I)
            s2 = lead.sub("", s2)

            # Now try to match a pure math expression up to a separator (, ; 'and' '&') or EOL
            m = re.match(
                r"^([0-9\.\s\+\-\*\/\^\(\)!]+)\s*(?:,|\s+(?:and|&)\s+|;\s+|$)",
                s2
            )
            if not m:
                return [], s

            expr = m.group(1).strip()
            # Ensure it actually looks like math (has an operator or factorial or sqrt()/^)
            if not re.search(r"[\+\-\*\/\^!]", expr):
                return [], s

            # Build remainder starting at match end
            rem = s2[m.end():].lstrip(" ,;")
            return [expr], rem if rem else ""

        for seg in segments:
            seg_clean = seg.strip().rstrip("?").strip()
            if not seg_clean:
                continue

            # ---- Step A: peel off a leading calculator expression, if present
            calc_chunks, remainder = extract_leading_calc(seg_clean)
            for c in calc_chunks:
                parts.append(c)

            seg_work = remainder if remainder != "" else seg_clean if not calc_chunks else ""
            if not seg_work:
                continue

            # ---- Step B: GSM8K whole-segment capture
            if gsm_pattern.search(seg_work) and re.search(r"\d", seg_work):
                parts.append(seg_work)
                continue

            # ---- Step C: doc-preface comma fix to avoid splitting "according to..., <question>"
            seg_for_split = doc_preface_comma.sub(r"\1 ", seg_work)

            # ---- Step D: generic clause splitting
            sub = re.split(r"\s+(?:and|&)\s+|;\s+|(?<!\d),\s+(?=[A-Za-z])", seg_for_split)
            for s in sub:
                s = s.strip()
                if not s or re.fullmatch(r"(also|and|then)", s, flags=re.I):
                    continue
                s = re.sub(r"^(?:also|and|then)\b[:,]?\s*", "", s, flags=re.I)
                if s:
                    parts.append(s)

        return parts

    # ---------- Routing ----------

    def _route_part(self, chunk: str) -> Tuple[str, str]:
        c = chunk.strip()

        # calculator: explicit factorial
        if re.fullmatch(r"\d+\s*!\s*", c):
            return "calculator", c

        # calculator: sqrt phrase
        if re.search(r"\bsquare\s*root\s*of\s*\d+\b", c, re.I):
            n = re.findall(r"\d+", c)[0]
            return "calculator", f"sqrt({n})"

        # calculator: pure math expression with operators
        def looks_like_pure_math(s: str) -> bool:
            s_norm = s.replace(",", "")
            s_norm = re.sub(r"\bsquare\s*root\s*of\s*(\d+)\b", r"sqrt(\1)", s_norm, flags=re.I)
            s_norm = re.sub(r"(\d+)\s*\^\s*(\d+)", r"\1**\2", s_norm)
            s_tmp = re.sub(r"\b(\d+)\s*!\b", r"\1!", s_norm)  # leave ! (calc expands)
            s_no_sqrt = re.sub(r"sqrt\s*\(", "(", s_tmp, flags=re.I)
            if re.search(r"[A-Za-z]", s_no_sqrt):
                return False
            return bool(re.fullmatch(r"[0-9\.\s\+\-\*\/\(\)\!]+", s_tmp))

        if re.search(r"[\+\-\*\/\^]", c) and looks_like_pure_math(c):
            return "calculator", c

        # gsm8k cues (narrative math)
        gsm_keywords = r"(how long|how many|how much|left|remain|time|speed|distance|rate|each|per|total|altogether|spent|earn|cost|bought|sold|gave|times|twice|thrice|more than|less than)"
        if re.search(gsm_keywords, c, re.I) and re.search(r"\d", c):
            return "gsm8k", c

        # explicit doc-reference → rag (strip preface)
        doc_pref = re.compile(
            r"^(?:according to|per|from)\s+(?:our|the|this)\s+(?:local\s+)?(?:docs?|document|knowledge\s*base|kb|notes|pdf)\s*,?\s*",
            re.I
        )
        if doc_pref.search(c):
            cleaned = doc_pref.sub("", c).strip()
            return "rag", (cleaned or c)

        # world-fact 'what is/define/explain' → prefer web
        if re.search(r"\b(what is|what’s|define|definition|explain)\b", c, re.I):
            return "web_search", c

        # other web facts
        web_keywords = r"(current|who|ceo|prime minister|president|secretary\-general|capital|population|winner|founded|founder)"
        if re.search(web_keywords, c, re.I):
            return "web_search", c

        # default: web over rag to avoid empty rag misses
        return "web_search", c

    # ---------- Execution ----------

    def execute_plan(self, plan_obj: Dict[str, Any]) -> Tuple[Dict[str, str], List[str]]:
        steps = plan_obj["plan"]
        results: Dict[str, str] = {}
        trace: List[str] = []

        for step in steps:
            sid = step["id"]
            tool = (step.get("tool") or "").lower().strip()
            inp = step.get("input", "")
            out_text = self._run_tool(tool, inp, original_query=" ".join([s.get("input", "") for s in steps]))
            results[sid] = out_text
            trace.append(f'{sid} -> {tool}("{inp}")')

        return results, trace

    def _run_tool(self, tool: str, inp: str, original_query: str) -> str:
        if tool in {"web", "web_search"}:
            if not self.tavily_key:
                return "Web search not configured (missing TAVILY_API_KEY)."
            res = web_search(inp, self.tavily_key, k=5)
            snippets = []
            for i, r in enumerate(res.get("results", []), start=1):
                title = (r.get("title") or "").strip()
                content = (r.get("content") or "").strip()
                if content:
                    snippets.append(f"[{i}] {title}: {content}")
            ctx = "\n".join(snippets[:6])
            if not ctx:
                return "No relevant web results found."
            synth = self.llm.chat(
                [
                    {"role": "system", "content": "Answer ONLY the asked question using the SEARCH SNIPPETS. Do not change the entity/topic. Include brief inline citations like [#]."},
                    {"role": "user", "content": f"USER QUESTION: {inp}\nORIGINAL FULL QUERY (for context, do NOT drift): {original_query}\n\nSEARCH SNIPPETS:\n{ctx}\n\nAnswer in 1-3 sentences."},
                ],
                temperature=0.0,
            )
            return synth.strip()

        if tool == "calculator":
            out = calc.calculate(inp)
            if "result" in out:
                v = out["result"]
                if isinstance(v, float) and v.is_integer():
                    return str(int(v))
                return str(v)
            return f"Calculator error: {out.get('error')}"

        if tool == "gsm8k":
            out = gsm8k_solver.solve_with_llm(self.llm, inp)
            final = out.get("final")
            return final if final else "(no numeric answer parsed)"

        if tool == "rag":
            ret = retrieve(inp, self.index_dir, k=4)
            if "error" in ret:
                return ret["error"]
            contexts = [c["text"] for c in ret.get("contexts", [])]
            if not contexts:
                if not self.tavily_key:
                    return "I don't know based on the local documents."
                return self._run_tool("web_search", inp, original_query)
            ans = answer_with_contexts(inp, contexts, self.llm).strip()
            if re.search(r"\bi (do not|don't) know\b", ans, re.I):
                if not self.tavily_key:
                    return ans
                return self._run_tool("web_search", inp, original_query)
            return ans

        return f"(unknown tool: {tool})"

    # ---------- Rendering ----------

    def render_final(self, plan_obj: Dict[str, Any], step_results: Dict[str, str]) -> str:
        template = (plan_obj.get("final_response_instructions") or "").strip()
        if not template:
            lines = [step_results.get(s["id"], "") for s in plan_obj.get("plan", [])]
            template = " | ".join([l for l in lines if l])

        for sid, val in step_results.items():
            template = template.replace("{{" + sid + "}}", val)

        note = (plan_obj.get("notes_on_false_premises") or "").strip()
        if note:
            template = f"{template}\n\nNote: {note}"
        return template
