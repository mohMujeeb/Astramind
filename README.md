# Tool-Calling Agent

A compact, batteries-included project: build and evaluate a tool‑calling agent using **LangGraph** with a controller that routes among four tools:

- **Calculator** — fast arithmetic via safe evaluation.
- **GSM8K Solver** — math word problems using an LLM prompt.
- **Web Search** — live fact lookup via Tavily.
- **RAG** — local document Q&A with FAISS + sentence-transformers.

Default model: **`llama-3.1-8b-instant`** via an OpenAI‑compatible API (e.g. OpenRouter, Groq proxy, or compatible gateway).

## Quickstart

```bash
# 1) Create venv
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)

# 2) Install deps
pip install -r requirements.txt

# 3) Configure keys
cp .env.example .env
# Fill in:
# - OPENAI_API_KEY (or a compatible provider key)
# - OPENAI_BASE_URL (e.g. https://openrouter.ai/api/v1) if not using OpenAI
# - TAVILY_API_KEY for web search
# - MODEL_NAME (defaults to llama-3.1-8b-instant)

# 4) (Optional) Ingest your local docs for RAG
python -m src.ingest --docs data/docs --index data/index

# 5) Chat from CLI
python -m src.app chat "What is the capital of France?"

# 6) Run benchmarks
bash scripts/run_benchmarks.sh
# or
python -m src.agent.eval.lama_eval
python -m src.agent.eval.gsm8k_eval
```


## Project Layout

```
ai-bootcamp-agent/
├─ src/
│  ├─ app.py                  # CLI entrypoints (chat, benchmark, ingest)
│  ├─ ingest.py               # Build a local FAISS index from /data/docs
│  └─ agent/
│     ├─ graph.py             # LangGraph: state & routing
│     ├─ controller.py        # Controller logic & tool routing heuristics
│     ├─ config.py            # Settings (env vars, defaults)
│     ├─ llm_client.py        # OpenAI-compatible client wrapper
│     ├─ tools/
│     │  ├─ calculator.py
│     │  ├─ gsm8k_solver.py
│     │  ├─ web_search.py
│     │  └─ rag.py
│     ├─ eval/
│     │  ├─ lama_eval.py
│     │  └─ gsm8k_eval.py
│     └─ prompts/
│        └─ controller_prompt.md
├─ data/
│  ├─ docs/                   # Put your PDFs/TXTs/DOCs here for RAG ingest
│  └─ index/                  # Vector index output (FAISS)
├─ scripts/
│  └─ run_benchmarks.sh
├─ tests/
│  └─ test_tools.py
├─ benchmarks/
│  ├─ lama_subset.csv
│  └─ gsm8k_subset.jsonl
├─ .env.example
├─ requirements.txt
└─ README.md
```

---

## Notes

- **LangGraph** coordinates tool calls; the controller uses light heuristics plus a model‑backed tie‑breaker.
- **RAG** uses `sentence-transformers` + **FAISS** for portability.
- **Web Search** uses Tavily; swap to SerpAPI/Bing by editing `src/agent/tools/web_search.py`.
- **Benchmarks** here are **tiny** illustrative subsets; replace with your curated splits before reporting numbers.

MIT License. Enjoy! 🚀


### Using Groq

Set your environment like this:

```bash
cp .env.example .env
# Edit .env:
OPENAI_API_KEY=<your_groq_api_key>
OPENAI_BASE_URL=https://api.groq.com/openai/v1
MODEL_NAME=llama-3.1-8b-instant
```
