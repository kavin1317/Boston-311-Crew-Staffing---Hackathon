"""Load the 2023 and 2024 311 files and keep the four departments we plan for.

Run from the repo root:  python -m src.data
"""
import html
from pathlib import Path

import duckdb
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RAW_FILES = [
    RAW_DIR / "lagan_311_open_data_2023_rvsd3.csv",
    RAW_DIR / "lagan_311_open_data_2024_rvsd3.csv",
]

config = yaml.safe_load((ROOT / "config.yaml").read_text())
DEPTS = list(config["departments"])  # PWDx, BTDT, PARK, ISD


def load_requests(con=None):
    """Return a DuckDB relation with both years, four departments only.

    Everything is read as text first because some columns have mixed values,
    then the three dates are parsed with TRY_CAST so bad values become NULL.
    Times are left in local Boston time.
    """
    if con is None:
        con = duckdb.connect()

    files = ", ".join(f"'{f}'" for f in RAW_FILES)
    depts = ", ".join(f"'{d}'" for d in DEPTS)

    sql = f"""
        SELECT
            * EXCLUDE (open_dt, sla_target_dt, closed_dt),
            TRY_CAST(open_dt AS TIMESTAMP)       AS open_dt,
            TRY_CAST(sla_target_dt AS TIMESTAMP) AS sla_target_dt,
            TRY_CAST(closed_dt AS TIMESTAMP)     AS closed_dt
        FROM read_csv([{files}], all_varchar = true, union_by_name = true)
        WHERE department IN ({depts})
    """
    return con.sql(sql)


def counts_by_year(rel):
    """Rows per department per year (year taken from open_dt)."""
    return rel.query("r", """
        SELECT department, year(open_dt) AS year, count(*) AS requests
        FROM r
        GROUP BY ALL
        ORDER BY department, year
    """).df()


def check_2024(rel):
    """2024 request count and on-time share per department.

    on_time_pct uses every request. on_time_pct_with_target leaves out
    requests that have no target date, in case that is how the brief counted.
    """
    return rel.query("r", """
        SELECT
            department,
            count(*) AS requests,
            round(100 * avg(CAST(on_time = 'ONTIME' AS INT)), 1) AS on_time_pct,
            round(100 * avg(CASE WHEN sla_target_dt IS NOT NULL
                                 THEN CAST(on_time = 'ONTIME' AS INT) END), 1)
                AS on_time_pct_with_target,
            count(*) FILTER (WHERE sla_target_dt IS NULL) AS no_target
        FROM r
        WHERE year(open_dt) = 2024
        GROUP BY department
        ORDER BY department
    """).df()


CLEAN_FILE = ROOT / "data" / "processed" / "requests_clean.parquet"

EMPLOYEE_SOURCES = ["Employee Generated", "City Worker App"]
RESIDENT_SOURCES = ["Citizens Connect App", "Constituent Call", "Self Service"]


def clean_requests(rel):
    """Rename to the team column names and add the helper columns.

    Every request stays in, whatever its closure_reason, since all of them
    are workload for the crews.
    """
    emp = ", ".join(f"'{s}'" for s in EMPLOYEE_SOURCES)
    res = ", ".join(f"'{s}'" for s in RESIDENT_SOURCES)

    df = rel.query("r", f"""
        SELECT
            case_enquiry_id AS id,
            open_dt         AS opened,
            sla_target_dt   AS target,
            closed_dt       AS closed,
            department,
            type,
            queue,
            case_title,
            neighborhood,
            source,
            on_time,
            year(open_dt)  AS year,
            month(open_dt) AS month,
            sla_target_dt IS NOT NULL AS has_target,
            date_diff('second', open_dt, closed_dt) / 86400.0 AS days_to_close,
            CASE
                WHEN source IN ({emp}) THEN 'Employee'
                WHEN source IN ({res}) THEN 'Resident'
            END AS source_group
        FROM r
        ORDER BY opened
    """).df()

    # titles can contain HTML codes like &amp; - decode them once per distinct title
    titles = df["case_title"].dropna().unique()
    fixed = {t: html.unescape(t).strip() for t in titles}
    df["case_title"] = df["case_title"].map(fixed)

    return df


def build_clean_file():
    con = duckdb.connect()
    df = clean_requests(load_requests(con))
    CLEAN_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CLEAN_FILE, index=False)
    return df


def load_clean():
    """Read the saved parquet (run `python -m src.data` first if it is missing)."""
    return pd.read_parquet(CLEAN_FILE)


if __name__ == "__main__":
    df = build_clean_file()
    print(f"Saved {len(df):,} rows to {CLEAN_FILE.relative_to(ROOT)}")
    print(df.groupby(["department", "year"]).size().unstack().to_string())
