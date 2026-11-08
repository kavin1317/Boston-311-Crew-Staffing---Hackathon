"""Monthly demand for 2023-2024 and the 2025 forecast.

Run from the repo root:  python -m src.forecast
(needs data/processed/requests_clean.parquet - run `python -m src.data` first)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.data import DEPTS, ROOT, config, load_clean

PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "docs" / "figures"
DEMAND_FILE = PROCESSED / "monthly_demand.csv"

DEPT_NAMES = {d: v["name"] for d, v in config["departments"].items()}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ---------- Step 5: monthly demand ----------

def monthly_demand(df=None):
    """Requests opened per department per month, 2023-2024 (24 rows each)."""
    if df is None:
        df = load_clean()
    out = (df.groupby(["department", "year", "month"])
             .size()
             .rename("requests")
             .reset_index())

    # make sure every month shows up, even one with no requests
    full = pd.MultiIndex.from_product(
        [DEPTS, [2023, 2024], range(1, 13)], names=["department", "year", "month"])
    out = (out.set_index(["department", "year", "month"])
              .reindex(full, fill_value=0)
              .reset_index())
    return out


def plot_monthly_demand(demand, path=None):
    """One line per department across the 24 months."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for dept in DEPTS:
        d = demand[demand.department == dept]
        x = pd.to_datetime(dict(year=d.year, month=d.month, day=1))
        ax.plot(x, d.requests, marker="o", markersize=3, label=DEPT_NAMES[dept])

    ax.set_title("Monthly 311 requests by department, 2023-2024")
    ax.set_ylabel("Requests opened")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
    return fig


def build_monthly_demand():
    demand = monthly_demand()
    demand.to_csv(DEMAND_FILE, index=False)
    plot_monthly_demand(demand, FIGURES / "monthly_demand.png")
    return demand


# ---------- Step 6: 2025 forecast ----------
#
# We only have 24 monthly points per department, which is too few for a
# fitted time-series model. So the method is simple and easy to explain:
#
#   forecast(month) = seasonal share of that month  x  2025 annual total
#
#   seasonal share   = the month's share of its year's requests,
#                      averaged over 2023 and 2024
#   2025 annual total = 2024 total x (2024 total / 2023 total),
#                      i.e. the 2023->2024 growth rate carried one more year

FORECAST_FILE = PROCESSED / "forecast_2025.csv"


def seasonal_shares(demand, years):
    """Average share of the year's requests that falls in each month."""
    d = demand[demand.year.isin(years)].copy()
    d["share"] = d.requests / d.groupby(["department", "year"]).requests.transform("sum")
    return d.groupby(["department", "month"]).share.mean()


def annual_totals(demand):
    return demand.groupby(["department", "year"]).requests.sum().unstack()


def forecast_2025(demand):
    totals = annual_totals(demand)
    growth = totals[2024] / totals[2023]
    level_2025 = totals[2024] * growth

    shares = seasonal_shares(demand, [2023, 2024]).reset_index()
    shares["forecast"] = shares.share * shares.department.map(level_2025)
    shares["forecast"] = shares.forecast.round().astype(int)
    return shares[["department", "month", "forecast"]]


def backtest_2024(demand):
    """Forecast 2024 using 2023 alone and compare with what really happened.

    With one year there is no trend to carry forward, so the 2024 forecast
    is just 2023's monthly pattern at 2023's total. MAPE is the average of
    |forecast - actual| / actual over the 12 months.
    """
    shares = seasonal_shares(demand, [2023]).reset_index()
    level = annual_totals(demand)[2023]
    shares["forecast"] = shares.share * shares.department.map(level)

    actual = demand[demand.year == 2024][["department", "month", "requests"]]
    bt = shares.merge(actual, on=["department", "month"])
    bt["ape"] = (bt.forecast - bt.requests).abs() / bt.requests

    mape = (bt.groupby("department").ape.mean() * 100).round(1)
    return mape.reindex(DEPTS).rename("MAPE %").reset_index()


def plot_forecast(demand, fc, path=None):
    """2023 and 2024 actuals with the 2025 forecast, one panel per department."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, dept in zip(axes.flat, DEPTS):
        for year, style in [(2023, ":"), (2024, "-")]:
            d = demand[(demand.department == dept) & (demand.year == year)]
            ax.plot(d.month, d.requests, style, color="grey", label=str(year))
        f = fc[fc.department == dept]
        ax.plot(f.month, f.forecast, "-o", markersize=3, color="tab:blue",
                label="2025 forecast")
        ax.set_title(DEPT_NAMES[dept])
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
        ax.grid(alpha=0.3)

    for ax in axes[1]:
        ax.set_xticks(range(1, 13), MONTHS)
    axes[0, 0].legend()
    fig.suptitle("Monthly requests: 2023, 2024 and 2025 forecast")
    fig.tight_layout()

    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
    return fig


def build_forecast():
    demand = pd.read_csv(DEMAND_FILE)
    fc = forecast_2025(demand)
    fc.to_csv(FORECAST_FILE, index=False)
    plot_forecast(demand, fc, FIGURES / "forecast_2025.png")
    return fc


if __name__ == "__main__":
    demand = build_monthly_demand()
    print(f"Saved {len(demand)} rows to {DEMAND_FILE.relative_to(ROOT)}")

    fc = build_forecast()
    print(f"Saved {len(fc)} rows to {FORECAST_FILE.relative_to(ROOT)}\n")
    print(fc.pivot(index="month", columns="department", values="forecast").to_string())
    print("\nBacktest (forecast 2024 from 2023):")
    print(backtest_2024(demand).to_string(index=False))
