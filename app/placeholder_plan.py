"""PLACEHOLDER staffing plan so the app's Staffing tab works before Person C's
real plan (data/processed/staffing_plan_2025.csv) exists.

This is NOT the staffing model. It is a "do nothing" baseline: every department
keeps its January 2025 crews all year, with no hires, releases or seasonal crews.
Demand comes from Person A's forecast; capacities and costs from config.yaml.
Requests are closed oldest first; anything not closed in its opening month is
overdue, and anything still open a month later is 30+ days overdue.

Run:  python -m app.placeholder_plan  -> data/processed/staffing_plan_2025_PLACEHOLDER.csv
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())
FORECAST = ROOT / "data/processed/forecast_2025.csv"
OUT = ROOT / "data/processed/staffing_plan_2025_PLACEHOLDER.csv"

# Column names the app expects from Person C's file too.
COLUMNS = ["scenario", "department", "month", "active_crews", "hires", "releases",
           "seasonal_crews", "cost", "closures", "overdue", "overdue_30plus", "on_time_pct"]


def _demand_multiplier(scenario: dict, dept: str, month: int) -> float:
    if "demand_multiplier" in scenario:
        return scenario["demand_multiplier"]
    spike = scenario.get("spike", {})
    if dept in spike.get("departments", []) and month == spike.get("month"):
        return spike["multiplier"]
    return 1.0


def build() -> pd.DataFrame:
    forecast = pd.read_csv(FORECAST)  # department, month, forecast
    days = CONFIG["working_days_per_month"]
    costs = CONFIG["overdue_costs"]
    rows = []
    for name, scenario in CONFIG["scenarios"].items():
        for dept, d in CONFIG["departments"].items():
            crews = d["crews_jan_2025"]
            capacity = crews * d["capacity_per_crew_per_day"] * days
            backlog = CONFIG["starting_backlog"]
            f = forecast[forecast.department == dept].set_index("month").forecast
            for month in range(1, 13):
                new = round(f[month] * _demand_multiplier(scenario, dept, month))
                old_closed = min(backlog, capacity)           # oldest first
                new_closed = min(new, capacity - old_closed)
                overdue_30 = backlog - old_closed              # carried over and still open
                overdue = new - new_closed                     # opened this month, not closed
                cost = (crews * d["annual_cost_per_crew"] / 12
                        + overdue * costs["per_overdue_request"]
                        + overdue_30 * costs["extra_if_30_days_past_target"])
                rows.append({
                    "scenario": name, "department": dept, "month": month,
                    "active_crews": crews, "hires": 0, "releases": 0, "seasonal_crews": 0,
                    "cost": round(cost), "closures": old_closed + new_closed,
                    "overdue": overdue, "overdue_30plus": overdue_30,
                    "on_time_pct": round(100 * new_closed / new, 1) if new else 100.0,
                })
                backlog = overdue + overdue_30
    return pd.DataFrame(rows, columns=COLUMNS)


if __name__ == "__main__":
    plan = build()
    plan.to_csv(OUT, index=False)
    print(f"Wrote {OUT.relative_to(ROOT)} ({len(plan)} rows) - PLACEHOLDER, not the staffing model")
    print(plan[plan.scenario == "base"].groupby("department")[["cost", "overdue", "on_time_pct"]]
          .agg({"cost": "sum", "overdue": "sum", "on_time_pct": "mean"}).round(1))
