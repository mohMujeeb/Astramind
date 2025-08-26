import json
from pathlib import Path
from dotenv import load_dotenv
from src.agent.config import Settings
from src.agent.llm_client import LLMClient
from src.agent.tools.gsm8k_solver import solve_with_llm

BENCH = Path("benchmarks/gsm8k_subset.jsonl")

def normalize_number(s: str) -> str:
    if s is None:
        return ""
    import re
    cleaned = re.sub(r"[^0-9.\-]", "", s)
    return cleaned.strip()

def run():
    load_dotenv()
    settings = Settings.from_env()
    llm = LLMClient(api_key=settings.openai_api_key, base_url=settings.openai_base_url, model=settings.model_name)

    correct = 0
    total = 0
    for line in BENCH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        q, gold = item["question"], normalize_number(item["answer"])
        out = solve_with_llm(llm, q)
        pred = normalize_number(out.get("final"))
        ok = (pred == gold)
        correct += int(ok)
        total += 1
        print(f"Q: {q}\nGold: {gold}\nPred: {pred}\nOK: {ok}\n---")
    acc = correct / max(total, 1)
    print(f"GSM8K subset accuracy (exact numeric match): {acc:.2%} ({correct}/{total})")

if __name__ == "__main__":
    run()
