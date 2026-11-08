"""Who gets served how well: neighborhoods (Step 7) and employee vs resident
requests (Step 8). Both use 2024, the same year the brief's figures describe.

On-time % counts every request, the same way the brief does (requests with
no target date are marked ONTIME in the data). Median days to close only
looks at requests that were closed.

Run from the repo root:  python -m src.equity
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.data import ROOT, load_clean
from src.forecast import DEPT_NAMES, FIGURES

PROCESSED = ROOT / "data" / "processed"
YEAR = 2024
MIN_REQUESTS = 100

# The same area shows up under two names in the data. Merge them so one
# neighborhood is not split in two.
SAME_AREA = {
    "South Boston": "South Boston / South Boston Waterfront",
    "Allston": "Allston / Brighton",
    "Brighton": "Allston / Brighton",
    "Mattapan": "Greater Mattapan",
}


def _summary(df, by):
    g = df.groupby(by)
    out = pd.DataFrame({
        "requests": g.size(),
        "on_time_pct": (g.on_time.apply(lambda s: (s == "ONTIME").mean()) * 100).round(1),
        "median_days_to_close": g.days_to_close.median().round(1),
    })
    return out.reset_index()


# ---------- Step 7: neighborhoods ----------

def neighborhood_table(df=None):
    if df is None:
        df = load_clean()
    d = df[(df.year == YEAR) & df.neighborhood.notna()].copy()
    d["neighborhood"] = d.neighborhood.replace(SAME_AREA)

    t = _summary(d, ["department", "neighborhood"])
    t = t[t.requests >= MIN_REQUESTS]
    return t.sort_values(["department", "on_time_pct"]).reset_index(drop=True)


def plot_neighborhoods(table, depts=("BTDT", "PWDx"), path=None):
    """On-time % by neighborhood, one panel per department, worst at the top."""
    fig, axes = plt.subplots(1, len(depts), figsize=(13, 7))
    for ax, dept in zip(axes, depts):
        t = table[table.department == dept].sort_values("on_time_pct", ascending=False)
        overall = (t.on_time_pct * t.requests).sum() / t.requests.sum()

        ax.barh(t.neighborhood, t.on_time_pct, color="tab:blue")
        ax.axvline(overall, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"{DEPT_NAMES[dept]} (dashed line = all neighborhoods, {overall:.0f}%)",
                     fontsize=10)
        ax.set_xlabel("Requests closed on time (%)")
        ax.set_xlim(0, 100)
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle(f"On-time share by neighborhood, {YEAR}")
    fig.tight_layout()

    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
    return fig


def build_neighborhoods():
    t = neighborhood_table()
    t.to_csv(PROCESSED / "neighborhood_equity.csv", index=False)
    plot_neighborhoods(t, path=FIGURES / "neighborhood_on_time.png")
    return t


# ---------- Step 8: employee vs resident requests ----------

def source_group_table(df=None):
    """Share of each department's requests logged by employees vs residents,
    with on-time % and median days to close for each group."""
    if df is None:
        df = load_clean()
    d = df[(df.year == YEAR) & df.source_group.notna()]

    t = _summary(d, ["department", "source_group"])
    t.insert(3, "share_pct",
             (t.requests / t.groupby("department").requests.transform("sum") * 100).round(1))
    return t


def build_source_groups():
    t = source_group_table()
    t.to_csv(PROCESSED / "source_group_summary.csv", index=False)
    return t


if __name__ == "__main__":
    t = build_neighborhoods()
    pd.set_option("display.width", 140)
    for dept in ["BTDT", "PWDx", "PARK", "ISD"]:
        print(f"\n{DEPT_NAMES[dept]}")
        print(t[t.department == dept].drop(columns="department").to_string(index=False))

    print("\nEmployee vs resident requests")
    print(build_source_groups().to_string(index=False))
