import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from src.agent.config import Settings
from src.agent.graph import build_graph, AgentState

BENCH = Path("benchmarks/lama_subset.csv")

def normalize(s: str) -> str:
    return (s or "").strip().lower()

def run():
    load_dotenv()
    settings = Settings.from_env()
    graph = build_graph(settings)

    df = pd.read_csv(BENCH)
    correct = 0
    total = 0
    for _, row in df.iterrows():
        q = row["prompt"]
        gold = row["answer"]
        state = AgentState(input=q)
        state = graph.invoke(state)
        pred = normalize(state.get("final_answer", ""))
        if normalize(gold) in pred:
            correct += 1
        total += 1
        print(f"Q: {q}\nGold: {gold}\nPred: {state.get('final_answer')}\n---")
    acc = correct / max(total, 1)
    print(f"LAMA subset accuracy (substring match): {acc:.2%} ({correct}/{total})")

if __name__ == "__main__":
    run()
