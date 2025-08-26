from typing import Optional
import typer
from rich import print as rprint
from dotenv import load_dotenv

from src.agent.graph import build_graph, AgentState
from src.agent.config import Settings

app = typer.Typer(add_completion=False)


@app.command()
def chat(query: str, max_hops: int = typer.Option(3, help="Max tool hops before stopping")):
    """Chat with the agent via LangGraph."""
    load_dotenv()
    settings = Settings.from_env()
    graph = build_graph(settings)
    state = AgentState(input=query)
    for i in range(max_hops):
        state = graph.invoke(state)
        if state.get("final_answer"):
            break
    rprint(f"[bold green]FINAL[/bold green]: {state.get('final_answer')}")
    if state.get("trace"):
        rprint("\n[dim]Trace:[/dim]")
        for step in state["trace"]:
            rprint(f" - {step}")


@app.command()
def ingest(docs: str = typer.Option("data/docs", help="Folder with .txt/.md/.pdf/.docx"),
           index: str = typer.Option("data/index", help="FAISS index folder")):
    """Build (or rebuild) the local FAISS index for RAG."""
    load_dotenv()
    from src.ingest import run_ingest
    run_ingest(docs, index)
    rprint(f"[bold green]Ingest complete[/bold green] -> {index}")


@app.command()
def benchmark(which: Optional[str] = typer.Option(None, help="lama|gsm8k|all")):
    load_dotenv()
    if which in (None, "lama", "all"):
        from src.agent.eval.lama_eval import run as run_lama
        run_lama()
    if which in (None, "gsm8k", "all"):
        from src.agent.eval.gsm8k_eval import run as run_gsm
        run_gsm()


if __name__ == "__main__":
    app()
