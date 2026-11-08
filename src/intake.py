"""Intake assistant (Person B).

Step 5 - retrieval: embed every catalog type (name + description + example
titles) and every resident message, then rank types by cosine similarity.
The top-k types (k from config.yaml) are the candidates the LLM picks from.

Run:  python -m src.intake retrieval   -> prints recall@k on the test set
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())
CATALOG_PATH = ROOT / "data/processed/type_catalog.json"
TEST_PATH = ROOT / "data/test_messages.csv"
EMB_MODEL = CONFIG["llm"]["embedding_model"]
TOP_K = CONFIG["llm"]["retrieval_top_k"]
EMB_CACHE = ROOT / "data/cache/embeddings.npz"  # text hash -> vector, so reruns cost nothing

_client: OpenAI | None = None


def client() -> OpenAI:
    global _client
    if _client is None:
        load_dotenv(ROOT / ".env")
        _client = OpenAI()
    return _client


def load_catalog() -> list[dict]:
    return json.loads(CATALOG_PATH.read_text())


def type_text(entry: dict) -> str:
    """The text embedded for one catalog type."""
    titles = "; ".join(t for t in entry["example_titles"] if t.lower() != entry["type"].lower())
    text = f"{entry['type']}: {entry['description']}"
    return f"{text} Examples: {titles}" if titles else text


# ---------------------------------------------------------------- embeddings
def _key(text: str) -> str:
    return hashlib.sha256(f"{EMB_MODEL}|{text}".encode()).hexdigest()[:20]


def _load_cache() -> dict[str, np.ndarray]:
    if EMB_CACHE.exists():
        with np.load(EMB_CACHE) as z:
            return {k: z[k] for k in z.files}
    return {}


def embed(texts: list[str]) -> np.ndarray:
    """Unit-length embeddings for texts, using the cache when possible."""
    cache = _load_cache()
    missing = list(dict.fromkeys(t for t in texts if _key(t) not in cache))
    for i in range(0, len(missing), 100):  # API batches
        batch = missing[i:i + 100]
        resp = client().embeddings.create(model=EMB_MODEL, input=batch)
        for text, item in zip(batch, resp.data):
            cache[_key(text)] = np.asarray(item.embedding, dtype=np.float32)
    if missing:
        EMB_CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(EMB_CACHE, **cache)
    vecs = np.stack([cache[_key(t)] for t in texts])
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


# ----------------------------------------------------------------- retrieval
class Retriever:
    def __init__(self, catalog: list[dict] | None = None):
        self.catalog = catalog or load_catalog()
        self.types = [e["type"] for e in self.catalog]
        self.matrix = embed([type_text(e) for e in self.catalog])

    def search(self, messages: list[str], k: int = TOP_K) -> list[list[tuple[str, float]]]:
        """For each message, the k most similar types with their cosine scores."""
        scores = embed(messages) @ self.matrix.T
        top = np.argsort(-scores, axis=1)[:, :k]
        return [[(self.types[j], float(scores[i, j])) for j in row] for i, row in enumerate(top)]


def evaluate_retrieval(k: int = TOP_K) -> pd.DataFrame:
    """Rank of the true type for every clear test message."""
    test = pd.read_csv(TEST_PATH)
    clear = test[~test.is_ambiguous].copy()
    retriever = Retriever()
    ranked = retriever.search(clear.message.tolist(), k=len(retriever.types))

    def rank_of(names, ranking):
        hits = [i + 1 for i, (t, _) in enumerate(ranking) if t in names]
        return min(hits) if hits else None

    clear["rank_true"] = [rank_of({t}, r) for t, r in zip(clear.true_type, ranked)]
    clear["rank_acceptable"] = [rank_of(set(a.split("|")), r)
                                for a, r in zip(clear.acceptable_types, ranked)]
    clear["top3"] = [[t for t, _ in r[:3]] for r in ranked]
    return clear


def main_retrieval() -> None:
    res = evaluate_retrieval()
    n = len(res)
    print(f"Retrieval on {n} clear test messages ({EMB_MODEL}, k={TOP_K})")
    for k in (1, 3, 5, TOP_K):
        strict = (res.rank_true <= k).mean()
        lenient = (res.rank_acceptable <= k).mean()
        print(f"  recall@{k:<2d}  true type: {strict:6.1%}   any acceptable type: {lenient:6.1%}")
    misses = res[res.rank_true > TOP_K]
    if len(misses):
        print(f"\nTrue type NOT in top {TOP_K}:")
        for _, r in misses.iterrows():
            print(f"  {r.id} rank {r.rank_true:>3}  {r.true_type}  <- {r.message[:70]}")
            print(f"          top 3: {r.top3}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "retrieval":
        main_retrieval()
    else:
        print(__doc__)
