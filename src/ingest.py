"""
Lightweight document ingester for local RAG.

- Accepts: .pdf, .txt, .md
- Chunks text into overlapping windows
- Embeds with sentence-transformers (config via EMBEDDINGS_MODEL)
- Builds a cosine-similarity FAISS index (IndexFlatIP on normalized vectors)
- Writes:
    <index_dir>/
      faiss.index
      vectors.npy
      texts.json   {"texts": [...], "meta": [{"source": "..."}]}
"""

from pathlib import Path
from typing import List
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import glob
from rich import print as rprint


# ---------- Readers ----------

def _read_pdf(path: str) -> str:
    """Try to extract text from a PDF using pypdf, then pdfminer.six as a fallback."""
    # 1) pypdf (fast, works for many PDFs)
    try:
        from pypdf import PdfReader  # pip install pypdf
        reader = PdfReader(path)
        pages = []
        for p in reader.pages:
            try:
                pages.append(p.extract_text() or "")
            except Exception:
                pages.append("")
        return "\n".join(pages)
    except Exception:
        pass

    # 2) pdfminer.six (slower, but robust)
    try:
        from pdfminer.high_level import extract_text  # pip install pdfminer.six
        return extract_text(path) or ""
    except Exception:
        return ""


def _read_doc(path: str) -> str:
    """Return extracted plain text for supported formats; empty string otherwise."""
    p = Path(path)
    suf = p.suffix.lower()
    if suf in [".txt", ".md"]:
        try:
            return p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""
    if suf == ".pdf":
        return _read_pdf(path)
    return ""


# ---------- Chunking ----------

def _chunk(text: str, size: int = 500, overlap: int = 100) -> List[str]:
    """Naive character-window chunking with overlap."""
    if not text:
        return []
    chunks: List[str] = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i + size])
        i += max(1, size - overlap)
    return [c for c in chunks if c.strip()]


# ---------- Ingest ----------

def run_ingest(docs_dir: str, index_dir: str):
    """
    Build a FAISS index from files in `docs_dir` and write artifacts to `index_dir`.
    Safe to call repeatedly; it overwrites vectors/index each time.
    """
    os.makedirs(index_dir, exist_ok=True)

    model_name = os.getenv("EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    rprint(f"[cyan]Embedding model:[/cyan] {model_name}")
    embedder = SentenceTransformer(model_name)

    texts: List[str] = []
    meta: List[dict] = []

    # Collect content
    doc_paths = list(glob.glob(str(Path(docs_dir) / "**/*"), recursive=True))
    if not doc_paths:
        rprint(f"[yellow]No files found under {docs_dir}.[/yellow]")

    for path in doc_paths:
        p = Path(path)
        if not p.is_file():
            continue
        content = _read_doc(path)
        if not content:
            continue
        for chunk in _chunk(content, size=500, overlap=100):
            texts.append(chunk)
            meta.append({"source": str(p)})

    if not texts:
        rprint(f"[yellow]No .pdf/.txt/.md content found in {docs_dir}. Add docs, then rerun ingest.[/yellow]")
        return

    # Embed (normalized for cosine via inner product)
    rprint(f"[cyan]Embedding {len(texts)} chunks…[/cyan]")
    embs = embedder.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=64,
        normalize_embeddings=True,  # important: enables cosine with IndexFlatIP
    )

    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)  # inner product on normalized vectors == cosine
    index.add(embs)

    # Persist artifacts
    vectors_path = Path(index_dir) / "vectors.npy"
    texts_path   = Path(index_dir) / "texts.json"
    faiss_path   = Path(index_dir) / "faiss.index"

    np.save(vectors_path, embs)
    texts_path.write_text(
        __import__("json").dumps({"texts": texts, "meta": meta}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    faiss.write_index(index, str(faiss_path))

    rprint(f"[green]Index built[/green] → {index_dir}  (items: {len(texts)})")


# ---------- CLI ----------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build FAISS index for local RAG (.pdf, .txt, .md)")
    parser.add_argument("--docs", default="data/docs", help="Folder with docs to index")
    parser.add_argument("--index", default="data/index", help="Output index folder")
    args = parser.parse_args()
    run_ingest(args.docs, args.index)
