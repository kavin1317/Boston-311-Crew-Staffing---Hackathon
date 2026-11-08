"""Intake assistant (Person B).

Step 5 - retrieval: embed every catalog type (name + description + example
titles) and every resident message, then rank types by cosine similarity.
The top-k types (k from config.yaml) are the candidates the LLM picks from.

Step 6 - structured output: the LLM picks one type from the candidates and
returns an IntakeResult (JSON schema). Department and queue come from the
catalog in code. Every LLM call is cached in data/cache/llm/.

Run:  python -m src.intake retrieval   -> prints recall@k on the test set
      python -m src.intake demo        -> 5 test messages end to end
      python -m src.intake eval        -> V1/V2/V3 on the full test set (Step 7)
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

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


# --------------------------------------------------------- structured output
# Step 6: the LLM only picks the request type (plus alternatives and whether to
# ask a follow-up question). Department and queue are looked up from the
# catalog in code, so the model can never invent a department or queue.

MODEL = CONFIG["llm"]["model"]
TEMPERATURE = CONFIG["llm"]["temperature"]
LLM_CACHE = ROOT / CONFIG["llm"]["cache_dir"]


class IntakeResult(BaseModel):
    request_type: str
    alternatives: list[str]          # next 2 most likely types
    confidence: Literal["high", "medium", "low"]
    needs_follow_up: bool
    follow_up_question: str | None


SYSTEM_PROMPT = """You are the intake assistant for Boston's 311 city services.
A resident describes a problem. Choose the request type that fits it best from
the list of request types you are given.

Rules:
- request_type: exactly one name from the list, copied exactly.
- alternatives: the next 2 most likely names from the list, copied exactly.
  If the message reports two separate problems, put the second problem's type
  first in alternatives.
- confidence: "high" if the message clearly matches one type, "medium" if a
  close alternative also fits, "low" if you are mostly guessing.
- needs_follow_up: true only if the message does not say what the problem is,
  or if two or more types fit and the difference changes who handles it.
  If the message is clear, set it to false.
- follow_up_question: when needs_follow_up is true, ONE short, friendly
  question that would settle it. Otherwise null.
Never invent a request type that is not in the list."""


def _type_list(entries: list[dict], names_only: bool = False) -> str:
    if names_only:
        return "\n".join(f"- {e['type']}" for e in entries)
    return "\n".join(f"- {e['type']}: {e['description']}" for e in entries)


def build_prompt(message: str, candidates: list[dict], examples: str = "",
                 names_only: bool = False) -> str:
    parts = [f"Request types:\n{_type_list(candidates, names_only)}"]
    if examples:
        parts.append(f"Worked examples:\n{examples}")
    parts.append(f"Resident message:\n{message}")
    return "\n\n".join(parts)


def _call_llm(prompt: str, attempt: int = 0) -> IntakeResult:
    """One structured call, cached as JSON by a hash of model + prompts."""
    key_text = json.dumps([MODEL, TEMPERATURE, SYSTEM_PROMPT, prompt, attempt])
    path = LLM_CACHE / f"intake_{hashlib.sha256(key_text.encode()).hexdigest()[:20]}.json"
    if path.exists():
        return IntakeResult.model_validate_json(path.read_text())

    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}]
    if attempt:  # the retry tells the model what went wrong
        messages.append({"role": "user", "content":
                         "Your last answer used a request type that is not in the list. "
                         "Answer again using only names copied exactly from the list."})
    resp = client().chat.completions.parse(
        model=MODEL, temperature=TEMPERATURE, messages=messages,
        response_format=IntakeResult)
    result = resp.choices[0].message.parsed

    LLM_CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(result.model_dump_json(indent=2))
    return result


def classify(message: str, candidates: list[dict], examples: str = "",
             names_only: bool = False) -> dict:
    """Pick a type from candidates. Retry once if the answer is not a candidate,
    then flag it. Department and queue are looked up from the catalog."""
    names = {e["type"] for e in candidates}
    prompt = build_prompt(message, candidates, examples, names_only)

    result = _call_llm(prompt)
    retried = result.request_type not in names
    if retried:
        result = _call_llm(prompt, attempt=1)
    flagged = result.request_type not in names

    entry = next((e for e in candidates if e["type"] == result.request_type), None)
    return {
        "request_type": result.request_type,
        "department": entry["department"] if entry else None,
        "queue": entry["primary_queue"] if entry else None,
        "alternatives": [a for a in result.alternatives if a in names][:2],
        "confidence": result.confidence,
        "needs_follow_up": result.needs_follow_up,
        "follow_up_question": result.follow_up_question if result.needs_follow_up else None,
        "retried": retried,
        "flagged": flagged,
        "candidates": [e["type"] for e in candidates],
    }


# ------------------------------------------------------ Step 7: prompt versions
# Six worked examples. None of these messages is in the test set. They show
# the answer format, a two-problem message (no follow-up needed) and an
# unclear message (follow-up needed).
_EXAMPLES = [
    ("bags of garbage and an old tv sitting on the empty lot next to 88 geneva ave for weeks",
     dict(request_type="Trash on Vacant Lot",
          alternatives=["Illegal Dumping", "Overflowing or Un-kept Dumpster"],
          confidence="high", needs_follow_up=False, follow_up_question=None)),
    ("bed bugs came thru the wall from the apartment next door, now theyre in my bed",
     dict(request_type="Bed Bugs",
          alternatives=["Pest Infestation - Residential", "Squalid Living Conditions"],
          confidence="high", needs_follow_up=False, follow_up_question=None)),
    ("lights on the basketball court at the park are all dead at night",
     dict(request_type="Parks Lighting/Electrical Issues",
          alternatives=["Street Light Outages", "General Lighting Request"],
          confidence="high", needs_follow_up=False, follow_up_question=None)),
    ("somebody stapled yard sale signs to every single pole on my block",
     dict(request_type="Illegal Posting of Signs",
          alternatives=["PWD Graffiti", "Sign Repair"],
          confidence="high", needs_follow_up=False, follow_up_question=None)),
    ("theres an issue with the sidewalk on my street",
     dict(request_type="Sidewalk Repair (Make Safe)",
          alternatives=["Unshoveled Sidewalk", "Sidewalk Repair"],
          confidence="low", needs_follow_up=True,
          follow_up_question="Is the sidewalk cracked or broken, or is it covered in snow or ice?")),
    ("car with 2 flat tires been dumped on harvard st for a month, and the stop sign next to it is bent over",
     dict(request_type="Abandoned Vehicles",
          alternatives=["Sign Repair", "Parking Enforcement"],
          confidence="high", needs_follow_up=False, follow_up_question=None)),
]
FEW_SHOT = "\n\n".join(f"Message: {m}\nAnswer: {json.dumps(a)}" for m, a in _EXAMPLES)

VERSIONS = {
    "V1": "zero-shot, all type names",
    "V2": "V1 + 6 worked examples",
    "V3": "top-8 retrieved types (with descriptions) + 6 worked examples",
}


class IntakeAssistant:
    """Retrieval + LLM. Default is V3, the version the app uses."""

    def __init__(self):
        self.retriever = Retriever()
        self.catalog = self.retriever.catalog
        self.by_type = {e["type"]: e for e in self.catalog}
        known = set(self.by_type)
        for _, a in _EXAMPLES:  # worked examples must use real type names
            assert {a["request_type"], *a["alternatives"]} <= known, a

    def __call__(self, message: str, version: str = "V3") -> dict:
        if version == "V1":
            return classify(message, self.catalog, names_only=True)
        if version == "V2":
            return classify(message, self.catalog, FEW_SHOT, names_only=True)
        hits = self.retriever.search([message])[0]
        candidates = [self.by_type[t] for t, _ in hits]
        out = classify(message, candidates, FEW_SHOT)
        out["candidate_scores"] = [round(s, 3) for _, s in hits]
        return out


EVAL_PATH = ROOT / "data/processed/intake_eval.csv"
EVAL_SUMMARY_PATH = ROOT / "data/processed/intake_eval_summary.csv"


def evaluate_versions(versions=("V1", "V2", "V3")) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every version on the full test set. One row per message per version."""
    test = pd.read_csv(TEST_PATH)
    assistant = IntakeAssistant()
    rows = []
    for v in versions:
        for _, t in test.iterrows():
            out = assistant(t.message, v)
            ok = set(t.acceptable_types.split("|")) if isinstance(t.acceptable_types, str) else set()
            rows.append({
                "version": v, "id": t.id, "message": t.message,
                "is_ambiguous": t.is_ambiguous,
                "true_type": t.true_type, "true_department": t.true_department,
                "true_queue": t.true_queue,
                "pred_type": out["request_type"], "pred_department": out["department"],
                "pred_queue": out["queue"], "alternatives": "|".join(out["alternatives"]),
                "confidence": out["confidence"], "needs_follow_up": out["needs_follow_up"],
                "follow_up_question": out["follow_up_question"],
                "retried": out["retried"], "flagged": out["flagged"],
                "dept_correct": out["department"] == t.true_department,
                "type_top1": out["request_type"] in ok,
                "type_top3": bool(ok & {out["request_type"], *out["alternatives"]}),
                "queue_correct": out["queue"] == t.true_queue,
            })
    res = pd.DataFrame(rows)

    def summary(g):
        clear, unclear = g[~g.is_ambiguous], g[g.is_ambiguous]
        return pd.Series({
            "department_acc": clear.dept_correct.mean(),
            "type_top1": clear.type_top1.mean(),
            "type_top3": clear.type_top3.mean(),
            "queue_acc": clear.queue_correct.mean(),
            "asks_on_unclear": unclear.needs_follow_up.mean(),
            "quiet_on_clear": 1 - clear.needs_follow_up.mean(),
            "flagged": g.flagged.sum(),
        })
    summ = res.groupby("version").apply(summary, include_groups=False)
    summ.insert(0, "description", [VERSIONS[v] for v in summ.index])
    return res, summ


def main_eval() -> None:
    res, summ = evaluate_versions()
    res.to_csv(EVAL_PATH, index=False)
    summ.to_csv(EVAL_SUMMARY_PATH)
    n_clear = int((~res[res.version == "V1"].is_ambiguous).sum())
    n_unclear = int(res[res.version == "V1"].is_ambiguous.sum())
    print(f"Test set: {n_clear} clear + {n_unclear} unclear messages\n")
    show = summ.drop(columns="description").copy()
    for c in show.columns[:-1]:
        show[c] = show[c].map("{:.0%}".format)
    print(show.to_string())
    print(f"\nWrote {EVAL_PATH.relative_to(ROOT)} and {EVAL_SUMMARY_PATH.relative_to(ROOT)}")


def main_demo(n: int = 5) -> None:
    """Step 6 check: run a few test messages end to end."""
    test = pd.read_csv(TEST_PATH)
    ids = ["T01", "T17", "T21", "T35", "T40"][:n]  # clear, two-problem, BTDT, ISD, unclear
    assistant = IntakeAssistant()
    for _, row in test[test.id.isin(ids)].iterrows():
        out = assistant(row.message, "V3")
        print(f"\n{row.id}: {row.message}")
        expected = row.true_type if isinstance(row.true_type, str) else "(follow-up question)"
        print(f"  expected: {expected}")
        print(json.dumps({k: v for k, v in out.items()
                          if k not in ("candidates", "candidate_scores")}, indent=2))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "retrieval":
        main_retrieval()
    elif cmd == "demo":
        main_demo()
    elif cmd == "eval":
        main_eval()
    else:
        print(__doc__)
