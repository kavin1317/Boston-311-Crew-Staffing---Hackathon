"""Request-type catalog.

Two outputs, both built from the raw 2024 311 data:

1. type_routing.json (written first by Kavin): for each type in the four
   in-scope departments, its main department and queue and their shares.
   Run:  python -m src.catalog routing

2. type_catalog.json: the same 158 types, each with its department and
   queues, a plain-language description (LLM) and example titles (from data).
   Person B's intake assistant reads this file.
   Run:  python -m src.catalog
"""
from __future__ import annotations

import hashlib
import html
import json
import re
import sys
from collections import Counter
from pathlib import Path

import duckdb
import yaml
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/lagan_311_open_data_2024_rvsd3.csv"
OUT_ROUTING = ROOT / "data/processed/type_routing.json"
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())
DEPARTMENTS = list(CONFIG["departments"])  # PWDx, BTDT, PARK, ISD


def _counts(con: duckdb.DuckDBPyConnection) -> "pd.DataFrame":
    dept_list = ", ".join(f"'{d}'" for d in DEPARTMENTS)
    return con.execute(f"""
        SELECT department, type, queue, COUNT(*) AS n
        FROM read_csv('{RAW}', all_varchar=true)
        WHERE department IN ({dept_list})
        GROUP BY ALL
        ORDER BY n DESC
    """).df()


MIN_SHARE = 0.005  # shares below 0.5% are lumped into "_other" to keep the catalog readable


def _shares(counts: "pd.Series", total: int) -> dict[str, float]:
    shares = {k: v / total for k, v in counts.items()}
    kept = {k: round(s, 4) for k, s in shares.items() if s >= MIN_SHARE}
    other = sum(s for s in shares.values() if s < MIN_SHARE)
    if other > 0:
        kept["_other"] = round(other, 4)
    return kept


def build_routing() -> list[dict]:
    df = _counts(duckdb.connect())
    routing = []
    for type_, g in df.groupby("type", sort=False):
        total = int(g.n.sum())
        dept = g.groupby("department").n.sum().sort_values(ascending=False)
        queue = g.groupby("queue").n.sum().sort_values(ascending=False)
        routing.append({
            "type": type_,
            "n_2024": total,
            "department": dept.index[0],
            "department_shares": _shares(dept, total),
            "primary_queue": queue.index[0],
            "primary_queue_share": round(queue.iloc[0] / total, 4),
            "queue_shares": _shares(queue, total),
        })
    routing.sort(key=lambda r: (-r["n_2024"], r["type"]))  # stable order for ties
    return routing


def main_routing() -> None:
    routing = build_routing()
    OUT_ROUTING.parent.mkdir(parents=True, exist_ok=True)
    OUT_ROUTING.write_text(json.dumps(routing, indent=2))

    total = sum(r["n_2024"] for r in routing)
    print(f"Wrote {OUT_ROUTING.relative_to(ROOT)}")
    print(f"Distinct types in scope: {len(routing)}  (requests: {total:,})")
    for d in DEPARTMENTS:
        types = [r for r in routing if r["department"] == d]
        print(f"  {d:5s} {len(types):3d} types  {sum(r['n_2024'] for r in types):>7,} requests")
    mixed = [r for r in routing if max(r["department_shares"].values()) < 0.9]
    print(f"Types whose top department has <90% share: {len(mixed)} "
          f"({sum(r['n_2024'] for r in mixed):,} requests)")



# ---------------------------------------------------------------------------
# Type catalog
#
# First version by Person A (all departments, LLM-picked titles). Revised by
# Person B:
#   - only the four in-scope departments (same 158 types as type_routing.json)
#   - department / queue fields come from build_routing(), in the agreed format
#   - example titles are picked in code from real data. In this file case_title
#     is usually just the type name, so many types have only 1-2 real titles.
#   - the LLM writes only the description, and gets the city's subject and
#     category so it can explain short or coded type names.
#
# Run:  python -m src.catalog            -> data/processed/type_catalog.json
#       python -m src.catalog routing    -> data/processed/type_routing.json
# ---------------------------------------------------------------------------

OUT_CATALOG = ROOT / "data/processed/type_catalog.json"
CACHE_DIR = ROOT / CONFIG["llm"]["cache_dir"]
MODEL = CONFIG["llm"]["model"]
TEMPERATURE = CONFIG["llm"]["temperature"]

MAX_TITLES = 5           # example titles kept per type
MIN_TITLE_COUNT = 5      # a title must appear on at least 5 requests...
MIN_TITLE_SHARE = 0.005  # ...and on at least 0.5% of the type's requests
TITLES_SHOWN = 10        # cleaned titles shown to the LLM


class TypeInfo(BaseModel):
    description: str


SYSTEM_PROMPT = """You help build a catalog of City of Boston 311 request types.
Residents' messages will later be matched to these descriptions, so write the
way residents talk.

For one request type you get its name, the department that handles it, the
city's subject and category for it, and the titles staff used on those
requests (often just the type name again), with counts.

Return description: ONE sentence, at most 30 words, describing the concrete
problem a resident sees or the service they want, in everyday words. Name the
specific things they would mention, e.g. "A car is blocking my driveway, parked
at a hydrant, or parked in a no-parking or permit-only spot."
Keep it general enough to cover every request of this type: list typical
examples, but do not invent street names, addresses or one specific item.
Do not start with "There is an issue". Do not just repeat the type name.
No department names, no codes, no jargon. If the type name is unclear, use the
subject, category and titles to work out what it means."""


# Internal notes that staff add to titles; removed before titles are counted.
_TITLE_CODES = [
    re.compile(r"^SCH\s*\d+\s*/\s*\d+\s*", re.I),          # "SCH2/6Rodent Activity"
    re.compile(r"^DI?SP(?![a-z])\s*[\d/]*\s*"),             # "DISP1/5Heat ...", "DSP 12/23 ..." (caps only)
    re.compile(r"^/\s*\d*\s*"),                             # "/24 Unsatisfactory ..."
    re.compile(r"\s*-\s*BPRD\s*$", re.I),                   # park-department suffix
    re.compile(r"\s*:\s*(District\s+\w+|--\s*Not in list)\s*$", re.I),  # "Missed Trash: District 07"
]
_STAFF_MARKERS = re.compile(r"\bprinted\b", re.I)  # titles of printed work orders are dropped
_PLACE_WORDS = re.compile(
    r"\b(park|playground|play area|field|common|square|pond|garden|mall|beach|court|"
    r"reservation|cemetery|fens|not in list)\b", re.I)


def clean_title(s: str | None) -> str:
    s = html.unescape(s or "")
    if _STAFF_MARKERS.search(s):
        return ""
    for pattern in _TITLE_CODES:
        s = pattern.sub("", s)
    s = " ".join(s.replace("_", " ").split()).strip(" -:*?")
    # "Equipment Repair: Boston Common" -> "Equipment Repair" (place names are not the problem)
    if ":" in s:
        head, tail = (part.strip() for part in s.split(":", 1))
        if head and _PLACE_WORDS.search(tail):
            s = head
    return s


def _type_context(con):
    """Subject/category and raw titles for every type, in-scope rows only."""
    dept_list = ", ".join(f"'{d}'" for d in DEPARTMENTS)
    src = f"read_csv('{RAW}', all_varchar=true)"
    meta = con.execute(f"""
        SELECT type, subject, reason, COUNT(*) AS n
        FROM {src} WHERE department IN ({dept_list})
        GROUP BY ALL
    """).df()
    titles = con.execute(f"""
        SELECT type, case_title, COUNT(*) AS n
        FROM {src} WHERE department IN ({dept_list}) AND case_title IS NOT NULL
        GROUP BY ALL
    """).df()
    return meta, titles


def _clean_title_counts(raw):
    """Clean raw titles and merge ones that differ only in case or codes."""
    # Sort ties by text too: DuckDB returns rows in no fixed order, and the
    # prompt (and so the cache key) must be identical on every run.
    t = raw.assign(clean=[clean_title(s) for s in raw.case_title])
    t = t.sort_values(["n", "case_title"], ascending=[False, True], kind="mergesort")
    t = t[t.clean != ""]
    t["key"] = t.clean.str.lower()
    out = t.groupby("key", sort=False).agg(title=("clean", "first"), n=("n", "sum"))
    out = out.reset_index().sort_values(["n", "key"], ascending=[False, True], kind="mergesort")
    return out[["title", "n"]].reset_index(drop=True)


def example_titles(raw, total):
    counts = _clean_title_counts(raw)
    keep = counts[(counts.n >= MIN_TITLE_COUNT) & (counts.n / total >= MIN_TITLE_SHARE)]
    if keep.empty:  # very small types: fall back to the single most common title
        keep = counts.head(1)
    return keep.title.head(MAX_TITLES).tolist()


def _mode(meta, col):
    return meta.groupby(col).n.sum().idxmax()


def _prompt(type_, dept, subject, category, title_counts):
    lines = "\n".join(f"- {t} ({n})" for t, n in title_counts)
    return (f"Request type: {type_}\nDepartment: {dept}\nSubject: {subject}\n"
            f"Category: {category}\nTitles staff used:\n{lines}")


def ask_llm(prompt, client=None):
    """Structured LLM call with a file cache.

    The cache key is a hash of the model, temperature, system prompt and user
    prompt, so a rerun with the same inputs reads the saved answer and makes
    no API call.
    """
    key_text = json.dumps([MODEL, TEMPERATURE, SYSTEM_PROMPT, prompt])
    key = hashlib.sha256(key_text.encode()).hexdigest()[:20]
    path = CACHE_DIR / f"catalog_{key}.json"

    if path.exists():
        return TypeInfo.model_validate_json(path.read_text())

    if client is None:
        load_dotenv(ROOT / ".env")
        client = OpenAI()

    resp = client.chat.completions.parse(
        model=MODEL,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format=TypeInfo,
    )
    info = resp.choices[0].message.parsed

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(info.model_dump_json(indent=2))
    return info


def build_catalog(limit=None):
    routing = build_routing()
    if limit:
        routing = routing[:limit]
    meta, titles = _type_context(duckdb.connect())

    catalog = []
    for r in routing:
        type_ = r["type"]
        m = meta[meta.type == type_]
        raw = titles[titles.type == type_]
        shown = _clean_title_counts(raw).head(TITLES_SHOWN)

        info = ask_llm(_prompt(type_, r["department"], _mode(m, "subject"),
                               _mode(m, "reason"), list(zip(shown.title, shown.n))))

        catalog.append({
            "type": type_,
            "department": r["department"],
            "primary_queue": r["primary_queue"],
            "queue_shares": r["queue_shares"],
            "n_2024": r["n_2024"],
            "description": info.description.strip(),
            "example_titles": example_titles(raw, r["n_2024"]),
        })
    return catalog


def main_catalog():
    catalog = build_catalog()
    OUT_CATALOG.parent.mkdir(parents=True, exist_ok=True)
    OUT_CATALOG.write_text(json.dumps(catalog, indent=2))

    print(f"Wrote {len(catalog)} types to {OUT_CATALOG.relative_to(ROOT)}")
    for d in DEPARTMENTS:
        print(f"  {d:5s} {sum(c['department'] == d for c in catalog):3d} types")
    n_titles = Counter(len(c["example_titles"]) for c in catalog)
    print("Example titles per type:", dict(sorted(n_titles.items())))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "routing":
        main_routing()
    else:
        main_catalog()
