"""Staffing planner (Person C).

Finds a low-cost month-by-month crew plan for each department with local
search, using the simulator to price every plan:
  1. Start from a reasonable plan.
  2. Try every small change: one more or one fewer crew in one month, or from
     one month to the end of the year (and, for Parks, seasonal crews).
  3. Keep the change that lowers total cost the most; repeat until no small
     change helps.
  4. Do this from a few starting plans and keep the best.

Run from the repo root:
  python -m src.planner             -> data/processed/staffing_plan_2025.csv
  python -m src.planner calibrate   -> 2024 check: simulated vs actual on-time %
"""
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from src.simulator import (CLEAN_FILE, FORECAST_FILE, PROCESSED, apply_scenario, config,
                           daily_demand, load_target_profile, simulate)

PLAN_FILE = PROCESSED / "staffing_plan_2025.csv"
PLAN_COLUMNS = ["scenario", "department", "month", "active_crews", "hires", "releases",
                "seasonal_crews", "cost", "closures", "overdue", "overdue_30plus", "on_time_pct"]
SHORTFALL_PENALTY = 1e7   # added per on-time point below a required target


def score(plan, demand, profile, min_on_time):
    s, _ = simulate(plan, demand, profile)
    cost = s["total_cost"]
    if min_on_time is not None and s["on_time_pct"] < 100 * min_on_time:
        cost += SHORTFALL_PENALTY * (100 * min_on_time - s["on_time_pct"])
    return cost


def demand_based_start(dept, demand):
    """Crews each month needed to just cover that month's demand."""
    d = config["departments"][dept]
    per_crew_month = d["capacity_per_crew_per_day"] * config["working_days_per_month"]
    dd = demand[demand.department == dept]
    by_month = dd.groupby(pd.to_datetime(dd.date).dt.month).typical.sum()
    crews = [math.ceil(by_month.get(m, 0) / per_crew_month) for m in range(1, 13)]
    crews[0] = d["crews_jan_2025"]  # January is fixed: any change needs a month's notice
    return crews


def neighbours(plan, has_seasonal, seasonal_months):
    """Every plan that differs from this one by one small change."""
    out = []
    crews = plan["crews"]
    for m in range(1, 12):  # February to December
        for step in (1, -1):
            c = list(crews)
            c[m] += step
            if c[m] >= 0:
                out.append({**plan, "crews": c})
            c = list(crews)
            for k in range(m, 12):
                c[k] += step
            if min(c) >= 0:
                out.append({**plan, "crews": c})
    if has_seasonal:
        seasonal = plan["seasonal"]
        for m in seasonal_months:
            for step in (1, -1):
                s = list(seasonal)
                s[m] += step
                if s[m] >= 0:
                    out.append({**plan, "seasonal": s})
        for step in (1, -1):  # the whole season at once
            s = list(seasonal)
            for m in seasonal_months:
                s[m] += step
            if min(s) >= 0:
                out.append({**plan, "seasonal": s})
            # swap: one seasonal crew for the season instead of one full-time
            # crew from the start of the season to the end of the year
            c = list(crews)
            for k in range(seasonal_months[0], 12):
                c[k] -= step
            if min(s) >= 0 and min(c) >= 0:
                out.append({**plan, "crews": c, "seasonal": s})
    return out


def local_search(plan, demand, profile, min_on_time, seen, max_rounds=200):
    """seen: plans already priced, so no plan is simulated twice."""
    has_seasonal = plan["department"] in config["seasonal_crews"]["departments"]
    seasonal_months = [m - 1 for m in config["seasonal_crews"]["months"]]

    def cost_of(p):
        key = (tuple(p["crews"]), tuple(p.get("seasonal", ())))
        if key not in seen:
            seen[key] = score(p, demand, profile, min_on_time)
        return seen[key]

    best, best_cost = plan, cost_of(plan)
    for _ in range(max_rounds):
        candidates = neighbours(best, has_seasonal, seasonal_months)
        costs = [cost_of(c) for c in candidates]
        i = int(np.argmin(costs))
        if costs[i] >= best_cost - 1e-6:
            break
        best, best_cost = candidates[i], costs[i]
    return best, best_cost


def optimize(dept, demand, profile, min_on_time=None):
    """Lowest-cost plan for one department.

    min_on_time: e.g. 0.90 to require at least 90% on time (None = no target).
    Returns (plan, summary, monthly table).
    """
    jan = config["departments"][dept]["crews_jan_2025"]
    smart = demand_based_start(dept, demand)
    starts = [
        [jan] * 12,                                  # keep January's level
        smart,                                       # follow demand
        [jan] + [c + 2 for c in smart[1:]],          # a bit above demand
        [jan] + [max(c - 2, 0) for c in smart[1:]],  # a bit below demand
    ]
    best_plan, best_cost = None, float("inf")
    seen = {}
    for crews in starts:
        plan = {"department": dept, "crews": crews}
        if dept in config["seasonal_crews"]["departments"]:
            plan["seasonal"] = [0] * 12
        plan, cost = local_search(plan, demand, profile, min_on_time, seen)
        if cost < best_cost:
            best_plan, best_cost = plan, cost
    summary, monthly = simulate(best_plan, demand, profile)
    return best_plan, summary, monthly


def _plan_one(args):
    """One scenario and one department (runs in its own process)."""
    name, dept, demand, profile = args
    _, _, monthly = optimize(dept, demand, profile)
    monthly.insert(0, "scenario", name)
    return name, dept, monthly


def build_plans(profile=None, scenarios=None, workers=None):
    """Best plan for every department under every scenario in config.yaml.

    The 20 scenario and department combinations are independent, so they run
    side by side on the computer's processor cores (workers=1 runs them one
    at a time).
    """
    profile = load_target_profile() if profile is None else profile
    forecast = pd.read_csv(FORECAST_FILE)
    scenarios = scenarios or list(config["scenarios"])
    tasks = []
    for name in scenarios:
        demand = daily_demand(apply_scenario(forecast, config["scenarios"][name]))
        for dept in config["departments"]:
            tasks.append((name, dept, demand[demand.department == dept], profile))

    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    print(f"Planning {len(tasks)} scenario and department combinations on {workers} cores ...")
    results = {}
    if workers == 1:
        for t in tasks:
            name, dept, monthly = _plan_one(t)
            results[(name, dept)] = monthly
            print(f"  done: {name:10s} {dept}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for name, dept, monthly in pool.map(_plan_one, tasks):
                results[(name, dept)] = monthly
                print(f"  done: {name:10s} {dept}")

    order = [(name, dept) for name in scenarios for dept in config["departments"]]
    plans = pd.concat([results[k] for k in order], ignore_index=True)[PLAN_COLUMNS]
    plans = plans.round({"cost": 0, "closures": 0, "overdue": 0,
                         "overdue_30plus": 0, "on_time_pct": 1})
    int_cols = ["cost", "closures", "overdue", "overdue_30plus"]
    plans[int_cols] = plans[int_cols].astype(int)
    plans.to_csv(PLAN_FILE, index=False)
    return plans


def calibrate_2024(profile=None):
    """Simulate 2024 with January 2025 crews all year and compare on-time %
    with what happened in 2024. Close numbers mean the simulator is believable."""
    profile = load_target_profile() if profile is None else profile
    history = pd.read_csv(PROCESSED / "monthly_demand.csv")
    h = history[history.year == 2024].rename(columns={"requests": "forecast"})
    demand = daily_demand(h[["department", "month", "forecast"]], year=2024)

    actual = pd.read_parquet(CLEAN_FILE, columns=["department", "year", "on_time"])
    actual = actual[actual.year == 2024]
    actual = (actual.on_time.eq("ONTIME").groupby(actual.department).mean() * 100).round(1)

    rows = []
    for dept, d in config["departments"].items():
        plan = {"department": dept, "crews": [d["crews_jan_2025"]] * 12}
        s, _ = simulate(plan, demand, profile)
        rows.append({"department": dept, "crews": d["crews_jan_2025"],
                     "requests_2024": int(h[h.department == dept].forecast.sum()),
                     "simulated_on_time_pct": round(s["on_time_pct"], 1),
                     "actual_on_time_pct": actual.get(dept)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 140)
    if len(sys.argv) > 1 and sys.argv[1] == "calibrate":
        print(calibrate_2024().to_string(index=False))
    else:
        plans = build_plans()
        base = plans[plans.scenario == "base"]
        print("\nBase plan, active crews by month:\n")
        print(base.pivot(index="month", columns="department", values="active_crews").to_string())
        print("\nTotal 2025 cost by scenario ($):\n")
        print(plans.pivot_table(index="scenario", columns="department", values="cost",
                                aggfunc="sum").round(0).to_string())
        print(f"\nSaved {PLAN_FILE.relative_to(PROCESSED.parents[1])}")
