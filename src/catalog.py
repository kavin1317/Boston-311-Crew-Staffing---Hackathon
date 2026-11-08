"""Request-type catalog.

Two outputs, both built from the raw 2024 311 data:

1. type_routing.json (written first by Kavin): for each type in the four
   in-scope departments, its main department and queue and their shares.
   Run:  python -m src.catalog routing

2. type_catalog.json (Step 4): all 170 types across every department, each
   with queues, a plain-language description and example titles from the LLM.
   Person B's intake assistant reads this file.
   Run:  python -m src.catalog
"""
from __future__ import annotations

import hashlib
import html
import json
import sys
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
    routing.sort(key=lambda r: r["n_2024"], reverse=True)
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
# Type catalog (Person A, Step 4)
#
# One entry for each of the 170 request types in the full 2024 file (all
# departments). The intake assistant picks from this list, so each entry gets
# a plain-language description written by the LLM, plus a few cleaned-up
# example titles.
#
# Run:  python -m src.catalog            -> data/processed/type_catalog.json
#       python -m src.catalog routing    -> the older type_routing.json
# ---------------------------------------------------------------------------

OUT_CATALOG = ROOT / "data/processed/type_catalog.json"
CACHE_DIR = ROOT / CONFIG["llm"]["cache_dir"]
MODEL = CONFIG["llm"]["model"]
TEMPERATURE = CONFIG["llm"]["temperature"]

TITLES_SHOWN = 25  # most common raw titles we show the model for each type


class TypeInfo(BaseModel):
    description: str
    example_titles: list[str]


SYSTEM_PROMPT = """You help build a catalog of City of Boston 311 request types.
For one request type you get its name, the department that handles it, the
city's category for it, and the most common titles staff typed on those
requests (with counts). Use the category to understand short or coded type
names.

Return:
- description: ONE plain sentence describing the concrete problem a resident
  sees, the way they would say it, e.g. "A car is parked illegally or
  blocking my driveway." Do not just repeat the type name. No department
  names, no internal jargon.
- example_titles: up to 5 short, clean titles, most common first. Copy them
  from the list. You may remove internal codes and staff notes (things like
  DISP, SCH 9/9, PRINTED, initials, dates, park codes like BPRD) and fix
  capitals and spacing, but do not add words or invent new titles. Merge
  near-duplicates. Leave out titles about a different problem than this type.
  Fewer than 5 is fine."""


def _type_stats(con):
    """Department, queues and counts for every type in the full 2024 file."""
    by_dept = con.execute(f"""
        SELECT type, department, queue, reason, COUNT(*) AS n
        FROM read_csv('{RAW}', all_varchar=true)
        GROUP BY ALL
    """).df()

    titles = con.execute(f"""
        WITH t AS (
            SELECT type, case_title, COUNT(*) AS n
            FROM read_csv('{RAW}', all_varchar=true)
            WHERE case_title IS NOT NULL
            GROUP BY type, case_title
        )
        SELECT * FROM t
        QUALIFY row_number() OVER (PARTITION BY type ORDER BY n DESC, case_title) <= {TITLES_SHOWN}
    """).df()
    titles["case_title"] = titles["case_title"].map(lambda s: html.unescape(s).strip())
    return by_dept, titles


def _queue_list(g, total):
    queues = g.groupby("queue").n.sum().sort_values(ascending=False)
    return [
        {"queue": q, "share": round(float(n / total), 4)}
        for q, n in queues.items()
        if n / total >= MIN_SHARE
    ]


def _prompt(type_, dept, category, titles):
    lines = "\n".join(f"- {t} ({n})" for t, n in titles)
    return (f"Request type: {type_}\nDepartment: {dept}\nCategory: {category}\n"
            f"Most common titles:\n{lines}")


def _squash(s):
    return " ".join(s.lower().split())


def _from_data(examples, raw_titles):
    """Keep only example titles that actually appear inside a raw title.

    The model is told not to invent titles, but this makes sure of it.
    """
    raw = [_squash(t) for t in raw_titles]
    kept = []
    for ex in examples:
        if any(_squash(ex) in r for r in raw) and ex not in kept:
            kept.append(ex)
    return kept[:5]


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
    con = duckdb.connect()
    by_dept, titles = _type_stats(con)

    sizes = by_dept.groupby("type").n.sum().sort_values(ascending=False)
    if limit:
        sizes = sizes.head(limit)

    catalog = []
    for type_, total in sizes.items():
        g = by_dept[by_dept.type == type_]
        dept = g.groupby("department").n.sum().idxmax()
        category = g.groupby("reason").n.sum().idxmax()
        t = titles[titles.type == type_]
        top_titles = list(zip(t.case_title, t.n))

        info = ask_llm(_prompt(type_, dept, category, top_titles))

        catalog.append({
            "type": type_,
            "department": dept,
            "queues": _queue_list(g, total),
            "n_2024": int(total),
            "description": info.description,
            "example_titles": _from_data(info.example_titles, t.case_title),
        })
    return catalog


def main_catalog():
    catalog = build_catalog()
    OUT_CATALOG.write_text(json.dumps(catalog, indent=2))
    print(f"Wrote {len(catalog)} types to {OUT_CATALOG.relative_to(ROOT)}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "routing":
        main_routing()
    else:
        main_catalog()
