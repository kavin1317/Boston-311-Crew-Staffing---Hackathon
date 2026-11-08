"""Staffing simulator (Person C).

Plays out one department's year day by day for a given staffing plan and
reports the cost and service level. The planner, the sensitivity runs and
the 2024 calibration check all use it.

How one day works:
  1. New requests arrive and are split by how urgent they are (the target
     profile), so each one gets a due date.
  2. Crews close as many open requests as they can, earliest due date first.
     Crews work on weekdays only.
  3. Anything still open after its due date becomes overdue ($20 once).
  4. Anything still open 30 days after its due date costs an extra $100 once.

Run from the repo root:  python -m src.simulator
"""
import calendar

import numpy as np
import pandas as pd
import yaml

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
CLEAN_FILE = PROCESSED / "requests_clean.parquet"
FORECAST_FILE = PROCESSED / "forecast_2025.csv"
PROFILE_FILE = PROCESSED / "target_profile.csv"

config = yaml.safe_load((ROOT / "config.yaml").read_text())


# Inputs: target profile and daily demand

def build_target_profile(df=None, year=2024):
    """Share of each department's requests due 0, 1, 2, ... days after opening.

    Days to target = target date minus opened date. Each request is put in
    the largest bucket that is not later than its target, so the simulator
    never gives a request more time than its target allows. Requests with no
    target date get no_target_days (they count as on time, as in the data).
    """
    sm = config["staffing_model"]
    if df is None:
        df = pd.read_parquet(CLEAN_FILE, columns=["opened", "target", "department", "year"])
    d = df[df.year == year].copy()

    days = (pd.to_datetime(d.target).dt.normalize()
            - pd.to_datetime(d.opened).dt.normalize()).dt.days
    days = days.fillna(sm["no_target_days"]).clip(lower=0)

    buckets = np.array(sorted(sm["target_buckets"]))
    idx = np.searchsorted(buckets, days.to_numpy(), side="right") - 1
    d["days_to_target"] = buckets[np.clip(idx, 0, len(buckets) - 1)]

    profile = (d.groupby(["department", "days_to_target"]).size()
                 .rename("n").reset_index())
    profile["share"] = profile.n / profile.groupby("department").n.transform("sum")
    profile = profile[["department", "days_to_target", "share"]]
    profile.to_csv(PROFILE_FILE, index=False)
    return profile


def load_target_profile():
    """The saved profile, or build it from requests_clean.parquet."""
    if PROFILE_FILE.exists():
        return pd.read_csv(PROFILE_FILE)
    if not CLEAN_FILE.exists():
        raise FileNotFoundError(
            "Need data/processed/requests_clean.parquet to build the target profile. "
            "Run `python -m src.data` first (needs the raw CSVs in data/raw/).")
    return build_target_profile()


def apply_scenario(monthly, scenario):
    """Monthly demand changed by one scenario from config.yaml.

    monthly : DataFrame with columns department, month, forecast
    scenario: e.g. {demand_multiplier: 1.2} or
              {spike: {departments: [...], month: 2, multiplier: 1.5}}
    """
    out = monthly.copy()
    out["forecast"] = out.forecast.astype(float)
    if "demand_multiplier" in scenario:
        out["forecast"] = out.forecast * scenario["demand_multiplier"]
    if "spike" in scenario:
        s = scenario["spike"]
        hit = out.department.isin(s["departments"]) & (out.month == s["month"])
        out.loc[hit, "forecast"] = out.loc[hit, "forecast"] * s["multiplier"]
    return out


def daily_demand(monthly, year=None):
    """Spread each month's requests evenly over the days of that month.

    monthly: DataFrame with columns department, month, forecast
    Returns columns date, department, typical.
    """
    year = year or config["plan_year"]
    rows = []
    for r in monthly.itertuples():
        n_days = calendar.monthrange(year, int(r.month))[1]
        for day in range(1, n_days + 1):
            rows.append((pd.Timestamp(year, int(r.month), day), r.department,
                         r.forecast / n_days))
    return pd.DataFrame(rows, columns=["date", "department", "typical"])


# Costs of a plan

def monthly_cost_per_crew(dept):
    return config["departments"][dept]["annual_cost_per_crew"] / 12


def plan_costs(plan):
    """Crew costs for each month of the plan (no penalties).

    plan["crews"]    : 12 numbers, crews WORKING in each month (Jan to Dec)
    plan["seasonal"] : 12 numbers, Parks seasonal crews (optional)
    """
    dept = plan["department"]
    life = config["crew_lifecycle"]
    monthly = monthly_cost_per_crew(dept)
    annual = config["departments"][dept]["annual_cost_per_crew"]
    crews = list(plan["crews"])
    seasonal = list(plan.get("seasonal", [0] * 12))

    hires, releases = [0] * 12, [0] * 12
    crew_cost, release_cost = [0.0] * 12, [0.0] * 12
    for m in range(12):
        crew_cost[m] += crews[m] * monthly
        crew_cost[m] += seasonal[m] * monthly * config["seasonal_crews"]["cost_multiplier"]
        if m == 0:
            continue
        change = crews[m] - crews[m - 1]
        if change > 0:
            hires[m] = change
            # a new crew trains during the month before it starts work
            if life["paid_during_training"]:
                crew_cost[m - 1] += change * monthly
        elif change < 0:
            releases[m] = -change
            # the crew works its notice month (m-1), then the release cost is due
            release_cost[m] = -change * annual * life["release_cost_share_of_annual"]

    return pd.DataFrame({
        "month": range(1, 13),
        "active_crews": crews,
        "hires": hires,
        "releases": releases,
        "seasonal_crews": seasonal,
        "crew_cost": crew_cost,
        "release_cost": release_cost,
    })


# The simulator

def simulate(plan, demand, profile, demand_col="typical"):
    """Run one department through one year.

    plan    : {"department": "BTDT", "crews": [12 values], "seasonal": [12 values]}
    demand  : daily demand, columns date, department, typical
    profile : target profile, columns department, days_to_target, share

    Returns (summary dict, monthly DataFrame).
    """
    dept = plan["department"]
    dcfg = config["departments"][dept]
    fines = config["overdue_costs"]
    sm = config["staffing_model"]

    d = demand[demand.department == dept].sort_values("date").reset_index(drop=True)
    dates = pd.to_datetime(d.date)
    arrivals = d[demand_col].to_numpy(dtype=float)
    months = dates.dt.month.to_numpy()
    is_weekday = (dates.dt.weekday < 5).to_numpy()
    n_days = len(d)

    p = profile[profile.department == dept]
    due_offsets = p.days_to_target.to_numpy(dtype=int)
    due_shares = p.share.to_numpy(dtype=float)
    max_off = due_offsets.max()

    # each month's capacity (crews x per-crew x 21 days) spread over its weekdays
    crews = np.array(plan["crews"], dtype=float)
    seasonal = np.array(plan.get("seasonal", [0] * 12), dtype=float)
    weekdays_in_month = np.array([is_weekday[months == m].sum() for m in range(1, 13)])
    monthly_capacity = ((crews + seasonal) * dcfg["capacity_per_crew_per_day"]
                        * config["working_days_per_month"])
    daily_capacity = np.where(is_weekday,
                              monthly_capacity[months - 1] / weekdays_in_month[months - 1], 0.0)

    open_by_due = np.zeros(n_days + max_off + 1)  # open requests by due day
    open_by_due[0] += config["starting_backlog"]
    earliest = 0
    closed_on_time = np.zeros(12)
    closed_late = np.zeros(12)
    overdue = np.zeros(12)
    overdue_30 = np.zeros(12)
    daily_backlog = np.zeros(n_days)

    for t in range(n_days):
        m = months[t] - 1

        # 1) arrivals get their due dates
        np.add.at(open_by_due, t + due_offsets, arrivals[t] * due_shares)
        earliest = min(earliest, t + due_offsets.min())

        # 2) close earliest due date first
        cap = daily_capacity[t]
        hi = t + max_off + 1
        if cap > 0 and earliest < hi:
            window = open_by_due[earliest:hi]
            take = np.diff(np.minimum(np.cumsum(window), cap), prepend=0.0)
            open_by_due[earliest:hi] = window - take
            late_part = max(0, t - earliest)        # due dates before today are late
            closed_late[m] += take[:late_part].sum()
            closed_on_time[m] += take[late_part:].sum()
            nonzero = np.flatnonzero(open_by_due[earliest:hi] > 1e-9)
            earliest += nonzero[0] if len(nonzero) else hi - earliest

        # 3) due today and still open: overdue.  4) 30 days past due and still open
        overdue[m] += open_by_due[t]
        if t >= 30:
            overdue_30[m] += open_by_due[t - 30]
        daily_backlog[t] = open_by_due.sum()

    year_end_backlog = open_by_due.sum()
    late_at_end = open_by_due[:n_days].sum()

    monthly = plan_costs(plan)
    monthly.insert(0, "department", dept)
    monthly["closures"] = closed_on_time + closed_late
    monthly["overdue"] = overdue
    monthly["overdue_30plus"] = overdue_30
    monthly["penalty_cost"] = (overdue * fines["per_overdue_request"]
                               + overdue_30 * fines["extra_if_30_days_past_target"])
    with np.errstate(invalid="ignore", divide="ignore"):
        monthly["on_time_pct"] = np.where(monthly.closures > 0,
                                          100 * closed_on_time / monthly.closures, 100.0)
    monthly["cost"] = monthly.crew_cost + monthly.release_cost + monthly.penalty_cost

    # Year-end rule: requests that arrived in the last N days may still be open.
    # Anything beyond that, or anything already late, means work piled up.
    allowed = arrivals[-sm["year_end_allowed_days"]:].sum()
    excess = max(late_at_end, year_end_backlog - allowed)
    year_end_penalty = excess * sm["year_end_penalty_per_request"]

    # On-time % for the year: closed by target, out of every request whose
    # outcome is known (closed, or still open and already late).
    known = closed_on_time.sum() + closed_late.sum() + late_at_end
    summary = {
        "department": dept,
        "total_cost": monthly.cost.sum() + year_end_penalty,
        "crew_cost": monthly.crew_cost.sum(),
        "release_cost": monthly.release_cost.sum(),
        "penalty_cost": monthly.penalty_cost.sum(),
        "year_end_penalty": year_end_penalty,
        "on_time_pct": 100 * closed_on_time.sum() / known if known > 0 else 100.0,
        "year_end_backlog": year_end_backlog,
        "total_arrivals": arrivals.sum(),
        "daily_backlog": daily_backlog,
    }
    return summary, monthly


if __name__ == "__main__":
    profile = load_target_profile()
    demand = daily_demand(pd.read_csv(FORECAST_FILE))
    print("January crews kept all year, 2025 forecast:\n")
    for dept, d in config["departments"].items():
        plan = {"department": dept, "crews": [d["crews_jan_2025"]] * 12}
        s, _ = simulate(plan, demand, profile)
        print(f"{dept:5s} on time {s['on_time_pct']:5.1f}%   total cost ${s['total_cost']:>12,.0f}"
              f"   open on Dec 31: {s['year_end_backlog']:>8,.0f}")
