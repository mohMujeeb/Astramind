import json
from pathlib import Path
from dotenv import load_dotenv
from src.agent.config import Settings
from src.agent.graph import build_graph, AgentState

BENCH = Path("benchmarks/mixed_subset.jsonl")

def norm(s: str) -> str:
    return (s or "").strip().lower()

def run():
    load_dotenv()
    settings = Settings.from_env()
    graph = build_graph(settings)

    correct = 0
    total = 0

    for line in BENCH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        q = item["question"]
        expect = item["expect"]

        state = AgentState(input=q)
        state = graph.invoke(state)
        pred = norm(state.get("final_answer", ""))

        # Check: all must_contain present
        must_ok = True
        for needle in expect.get("must_contain", []):
            if norm(needle) not in pred:
                must_ok = False
                break

        # Check: at least one web_any present (if provided)
        web_ok = True
        web_any = expect.get("web_any", [])
        if web_any:
            web_ok = any(norm(w) in pred for w in web_any)

        ok = must_ok and web_ok
        correct += int(ok)
        total += 1

        print("Q:", q)
        print("Pred:", state.get("final_answer", ""))
        print("OK:", ok)
        print("---")

    acc = correct / max(total, 1)
    print(f"Mixed subset accuracy: {acc:.2%} ({correct}/{total})")

if __name__ == "__main__":
    run()
