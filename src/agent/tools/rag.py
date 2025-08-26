from typing import Dict, Any, List
from sentence_transformers import SentenceTransformer
import json, os, faiss, numpy as np
from pathlib import Path


def _load_index(index_dir: str):
    idx_path = Path(index_dir) / "faiss.index"
    vec_path = Path(index_dir) / "vectors.npy"
    txt_path = Path(index_dir) / "texts.json"
    if not (idx_path.exists() and vec_path.exists() and txt_path.exists()):
        return None, None, None
    index = faiss.read_index(str(idx_path))
    vectors = np.load(vec_path)
    data = json.loads(Path(txt_path).read_text(encoding="utf-8"))
    return index, vectors, data


def retrieve(question: str, index_dir: str, k: int = 4) -> Dict[str, Any]:
    index, vectors, data = _load_index(index_dir)
    if index is None:
        # point user to the working command
        return {
            "error": f"Index missing in {index_dir}. Build it with: "
                     f"python -m src.ingest --docs data/docs --index {index_dir}"
        }

    model_name = os.getenv("EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embedder = SentenceTransformer(model_name)
    q_emb = embedder.encode([question], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(q_emb, k)
    I = I[0].tolist()
    hits = []
    for i in I:
        hits.append({"text": data["texts"][i], "meta": data["meta"][i]})
    return {"contexts": hits}


def answer_with_contexts(question: str, contexts: List[str], llm) -> str:
    context_block = "\n---\n".join(contexts)
    system = "You answer strictly from the provided CONTEXT. If unknown, say you don't know."
    user = f"""CONTEXT:
{context_block}

QUESTION: {question}
Answer in 2-4 sentences."""
    return llm.chat([{"role": "system", "content": system}, {"role": "user", "content": user}])
