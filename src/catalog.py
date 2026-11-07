"""Request-type catalog (Person B).

Step 2: build type routing from the raw 2024 311 data.
For every request type in the four in-scope departments, record its most common
department and queue plus each department's and queue's share of that type.

Run:  python -m src.catalog
Output: data/processed/type_routing.json
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import yaml

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


def main() -> None:
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


if __name__ == "__main__":
    main()
