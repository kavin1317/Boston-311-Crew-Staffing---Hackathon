"""Sensitivity and the cost-vs-service curve (Person C).

1. Scenarios: for every scenario in config.yaml (high, low, snowstorm,
   heat_wave) and every department, compare
     - the base plan kept as it is (does it still hold up?), with
     - the plan re-optimized for that scenario (read from staffing_plan_2025.csv).
   For the spike scenarios, also count how many weeks the backlog takes to get
   back to within 5% of its base level once the spike month is over.

2. Cost vs service: for each department, keep a flat number of crews from
   February to December, from a few below January's level to well above it,
   and record on-time % and cost. Then the lowest-cost plan for each on-time
   target in config.yaml (re-optimized only when the lowest-cost plan misses it).

Run from the repo root (after `python -m src.planner`):  python -m src.sensitivity
"""
import calendar

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.planner import PLAN_FILE, optimize
from src.simulator import (FORECAST_FILE, PROCESSED, ROOT, apply_scenario, config,
                           daily_demand, load_target_profile, simulate)

SENSITIVITY_FILE = PROCESSED / "sensitivity.csv"
CURVE_FILE = PROCESSED / "cost_curve.csv"
TARGETS_FILE = PROCESSED / "cost_curve_targets.csv"
FIGURE = ROOT / "docs" / "figures" / "cost_vs_service.png"
DEPT_NAMES = {k: v["name"] for k, v in config["departments"].items()}


def plan_from_table(plans, scenario, dept):
    t = plans[(plans.scenario == scenario) & (plans.department == dept)].sort_values("month")
    return {"department": dept, "crews": t.active_crews.tolist(),
            "seasonal": t.seasonal_crews.tolist()}


def weeks_to_recover(spike_backlog, base_backlog, spike_month, year):
    """Weeks after the spike month until the backlog is back near its base level."""
    first_day_after = sum(calendar.monthrange(year, m)[1] for m in range(1, spike_month + 1))
    for i in range(first_day_after, len(spike_backlog)):
        if spike_backlog[i] <= base_backlog[i] * 1.05 + 1:
            return round((i - first_day_after + 1) / 7, 1)
    return "not by Dec 31"


def scenario_table(plans, profile, forecast):
    base_demand = daily_demand(forecast)
    rows = []
    for dept in config["departments"]:
        base_plan = plan_from_table(plans, "base", dept)
        base, _ = simulate(base_plan, base_demand, profile)
        rows.append(_row(dept, "base", "base plan", base, base_plan, base_plan))
        for name, sc in config["scenarios"].items():
            if name == "base":
                continue
            if "spike" in sc and dept not in sc["spike"]["departments"]:
                continue
            demand = daily_demand(apply_scenario(forecast, sc))
            kept, _ = simulate(base_plan, demand, profile)
            new_plan = plan_from_table(plans, name, dept)
            new, _ = simulate(new_plan, demand, profile)
            rec = "n/a"
            if "spike" in sc:
                rec = weeks_to_recover(kept["daily_backlog"], base["daily_backlog"],
                                       sc["spike"]["month"], config["plan_year"])
            rows.append(_row(dept, name, "base plan kept", kept, base_plan, base_plan, rec))
            rows.append(_row(dept, name, "re-optimized", new, new_plan, base_plan))
    return pd.DataFrame(rows)


def _row(dept, scenario, plan_type, s, plan, base_plan, recovery="n/a"):
    avg = np.mean(plan["crews"][1:]) + np.mean(plan.get("seasonal", [0] * 12)[1:])
    base_avg = np.mean(base_plan["crews"][1:]) + np.mean(base_plan.get("seasonal", [0] * 12)[1:])
    return {
        "department": dept,
        "scenario": scenario,
        "plan": plan_type,
        "on_time_pct": round(s["on_time_pct"], 1),
        "total_cost": round(s["total_cost"]),
        "avg_crews_feb_dec": round(avg, 1),
        "crews_vs_base": round(avg - base_avg, 1),
        "open_dec_31": round(s["year_end_backlog"]),
        "weeks_to_recover": recovery,
    }


def cost_curve(plans, profile, forecast):
    demand = daily_demand(forecast)
    sweep, targets = [], []
    for dept, d in config["departments"].items():
        jan = d["crews_jan_2025"]
        for level in range(max(1, jan - 4), jan + max(10, jan // 4) + 1):
            plan = {"department": dept, "crews": [jan] + [level] * 11}
            s, _ = simulate(plan, demand, profile)
            sweep.append({"department": dept, "crews_feb_dec": level,
                          "on_time_pct": round(s["on_time_pct"], 1),
                          "crew_cost": round(s["crew_cost"] + s["release_cost"]),
                          "fines": round(s["penalty_cost"] + s["year_end_penalty"]),
                          "total_cost": round(s["total_cost"])})

        best_plan = plan_from_table(plans, "base", dept)
        best, _ = simulate(best_plan, demand, profile)
        for target in config["staffing_model"]["on_time_targets"]:
            if best["on_time_pct"] >= 100 * target:
                plan, s = best_plan, best
            else:
                print(f"  {dept}: finding the lowest-cost plan for {target:.0%} on time ...")
                plan, s, _ = optimize(dept, demand, profile, min_on_time=target)
            targets.append({"department": dept, "target_pct": round(100 * target),
                            "on_time_pct": round(s["on_time_pct"], 1),
                            "meets_target": bool(s["on_time_pct"] >= 100 * target),
                            "total_cost": round(s["total_cost"]),
                            "extra_cost_vs_lowest": round(s["total_cost"] - best["total_cost"]),
                            "avg_crews_feb_dec": round(np.mean(plan["crews"][1:]), 1)})
    return pd.DataFrame(sweep), pd.DataFrame(targets)


def plot_cost_curve(sweep, path=FIGURE):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, dept in zip(axes.flat, config["departments"]):
        s = sweep[sweep.department == dept]
        ax.plot(s.on_time_pct, s.total_cost / 1e6, "o-", label="total cost (crews + fines)")
        ax.plot(s.on_time_pct, s.crew_cost / 1e6, "s:", label="crew cost only")
        low = s.loc[s.total_cost.idxmin()]
        ax.annotate(f"lowest cost: {int(low.crews_feb_dec)} crews",
                    (low.on_time_pct, low.total_cost / 1e6), textcoords="offset points",
                    xytext=(-110, 25), arrowprops={"arrowstyle": "->"})
        ax.set_title(f"{DEPT_NAMES[dept]}: cost vs on-time, 2025")
        ax.set_xlabel("On-time %")
        ax.set_ylabel("Cost for 2025 ($ millions)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Flat crew levels from February to December")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    return fig


if __name__ == "__main__":
    matplotlib.use("Agg")
    pd.set_option("display.width", 160)
    plans = pd.read_csv(PLAN_FILE)
    profile = load_target_profile()
    forecast = pd.read_csv(FORECAST_FILE)

    table = scenario_table(plans, profile, forecast)
    table.to_csv(SENSITIVITY_FILE, index=False)
    print(table.to_string(index=False))

    sweep, targets = cost_curve(plans, profile, forecast)
    sweep.to_csv(CURVE_FILE, index=False)
    targets.to_csv(TARGETS_FILE, index=False)
    plot_cost_curve(sweep)
    print("\nLowest-cost plan for each on-time target:\n")
    print(targets.to_string(index=False))
    print("\nSaved data/processed/sensitivity.csv, cost_curve.csv, cost_curve_targets.csv")
    print("Saved docs/figures/cost_vs_service.png")
