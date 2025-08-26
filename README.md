AstraMind — Tool-Calling AI Agent 

AstraMind is a full-stack, agentic AI system that decomposes complex user queries into subtasks and dynamically routes each step to the most appropriate tool.
It goes beyond a prototype — featuring a Django backend, PostgreSQL memory persistence, and LangGraph orchestration — making it a deployable AI agent platform.

 Features

 Calculator — fast arithmetic with safe evaluation.

 GSM8K Solver — solves math word problems with an LLM-prompted solver.

 Web Search — live fact lookup using Tavily API.

 RAG (Retrieval-Augmented Generation) — query local docs using FAISS + SentenceTransformers.

 Memory Persistence — queries, results, and tool traces stored in PostgreSQL.

 Web UI + API — Django-powered frontend and REST API.

 Groq (OpenAI-compatible API) — ultra-fast inference with llama-3.1-8b-instant.

 System Workflow

AstraMind intelligently decomposes queries and routes subtasks:

Example:

Input: “A train travels 60 km in 1 hour. How long will it take to travel 150 km? And who is the current UN Secretary-General?”

GSM8K Solver → Calculates travel time = 2.5 hours

Web Search → Finds António Guterres is the UN Secretary-General

Output → 2.5 | António Guterres (since Jan 2017)

Memory table + tool traces are logged in PostgreSQL for reproducibility.

 Tech Stack

Python 3.10+

Django (backend, API, frontend)

PostgreSQL (memory + query persistence)

LangGraph (controller orchestration & routing)

FAISS + SentenceTransformers (vector search for RAG)

Groq API (OpenAI-compatible) for LLM calls

Tavily API for web search

TailwindCSS for clean frontend UI

Docker-ready setup for deployment

Benchmarks: LAMA (factual recall), GSM8K (reasoning)

 Quickstart
# 1) Create venv
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)

# 2) Install dependencies
pip install -r requirements.txt

# 3) Configure environment
cp .env.example .env
# Add:
# - OPENAI_API_KEY (Groq or compatible key)
# - OPENAI_BASE_URL (e.g. https://api.groq.com/openai/v1)
# - TAVILY_API_KEY for web search
# - MODEL_NAME=llama-3.1-8b-instant

# 4) (Optional) Ingest local docs for RAG
python -m src.ingest --docs data/docs --index data/index

# 5) Run agent via CLI
python -m src.app chat "What is the capital of France?"

# 6) Run benchmarks
bash scripts/run_benchmarks.sh
python -m src.agent.eval.lama_eval
python -m src.agent.eval.gsm8k_eval

# 7) Launch Django Web App
python manage.py runserver
# → Visit http://127.0.0.1:8000/agent/

 Project Layout
AstraMind/
├─ benchmarks/                 # Benchmark subsets (LAMA, GSM8K, mixed)
├─ bootcamp/agent/             # Django app: models, views, templates, static assets
│  ├─ templates/chat/          # Frontend templates (index, landing, about)
│  ├─ models.py                # PostgreSQL memory schema
│  ├─ memory.py                # Memory logic
│  ├─ views.py / urls.py       # Django views & routing
│  └─ admin.py                 # Django admin
├─ data/
│  ├─ docs/                    # Knowledge base documents
│  └─ index/                   # FAISS vector store
├─ scripts/                    # Benchmark runner scripts
├─ src/agent/                  # Core agent logic
│  ├─ tools/                   # Tool implementations (calc, web, rag, gsm8k)
│  ├─ controller.py            # Controller logic
│  ├─ graph.py                 # LangGraph orchestration
│  └─ eval/                    # Evaluation scripts
└─ requirements.txt

Evaluation

LAMA Subset → factual recall accuracy

GSM8K Subset → symbolic reasoning accuracy

Metrics tracked:

 Exact-match accuracy

 Failure modes: tool misrouting, retrieval misses, reasoning slips, stale web search

 Stretch Goals

Fine-tune GSM8K Solver (Unsloth/Ollama)

Add caching & guardrails for safer responses

Telemetry for per-tool latency & cost tracking

Ablations to analyze tool contribution