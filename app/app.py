"""Boston 311 Crew Staffing - Streamlit app (Person B).

Tab 1  Intake assistant: a resident message -> request type, department, queue,
       confidence and a follow-up question when needed (retrieval + LLM, V3).
Tab 2  Staffing plan: Person C's month-by-month 2025 plan, by scenario.

Run:  streamlit run app/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())
DEPTS = list(CONFIG["departments"])
DEPT_NAMES = {k: v["name"] for k, v in CONFIG["departments"].items()}
PLAN_PATH = ROOT / "data/processed/staffing_plan_2025.csv"
PLACEHOLDER_PATH = ROOT / "data/processed/staffing_plan_2025_PLACEHOLDER.csv"
PLAN_COLUMNS = ["scenario", "department", "month", "active_crews", "hires", "releases",
                "seasonal_crews", "cost", "closures", "overdue", "overdue_30plus", "on_time_pct"]
# Fixed categorical order: each department always keeps the same color.
DEPT_COLORS = {"PWDx": "#2a78d6", "BTDT": "#eb6834", "PARK": "#1baf7a", "ISD": "#eda100"}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

EXAMPLES = [
    "theres a huge pothole on centre st by the T, almost lost a tire this morning",
    "somebody parked across my driveway on tremont again, i cant get out for work",
    "landlord still hasnt fixed the heat, its 52 degrees in my apartment",
    "tree came down in last nights storm and its blocking all of centre st",
    "something is wrong with the tree out front",
]

st.set_page_config(page_title="Boston 311 Crew Staffing", layout="wide")
st.title("Boston 311 Crew Staffing")
st.caption("IE 5374 Applied Generative AI - Module 7 Hackathon")


@st.cache_resource(show_spinner="Loading the request-type catalog...")
def get_assistant():
    from src.intake import IntakeAssistant
    return IntakeAssistant()


@st.cache_data
def load_plan() -> tuple[pd.DataFrame, bool]:
    """Person C's plan if it exists, otherwise the placeholder."""
    if PLAN_PATH.exists():
        return pd.read_csv(PLAN_PATH), False
    return pd.read_csv(PLACEHOLDER_PATH), True


intake_tab, staffing_tab = st.tabs(["Intake assistant", "Staffing plan 2025"])

# ------------------------------------------------------------------ Tab 1
with intake_tab:
    st.subheader("Route a resident request")
    st.write("Describe the problem the way a resident would. The assistant finds the "
             "8 most similar request types, the LLM picks one, and the department and "
             "queue are looked up from the city's 2024 data.")

    example = st.selectbox("Try an example", ["(type your own)"] + EXAMPLES)
    message = st.text_area("Resident message",
                           value="" if example == "(type your own)" else example,
                           height=100, placeholder="e.g. streetlight on my corner has been out for a week")

    if st.button("Route request", type="primary", disabled=not message.strip()):
        try:
            with st.spinner("Matching request types..."):
                out = get_assistant()(message.strip(), "V3")
        except Exception as e:  # missing API key, network, etc.
            st.error(f"Could not route the request: {e}")
            st.stop()

        if out["flagged"]:
            st.error("The model answered with a request type that is not in the catalog. "
                     "Please route this one by hand.")
        if out["needs_follow_up"]:
            st.info(f"**Follow-up question for the resident:** {out['follow_up_question']}")

        # Plain text, not st.metric: metrics cut long type names short.
        dept = out["department"]
        c1, c2, c3 = st.columns([2, 2, 1])
        c1.caption("Request type")
        c1.markdown(f"#### {out['request_type']}")
        c2.caption("Department")
        c2.markdown(f"#### {dept} - {DEPT_NAMES.get(dept, dept)}" if dept else "#### -")
        c3.caption("Confidence")
        c3.markdown(f"#### {out['confidence'].title()}")
        st.write(f"**Queue:** {out['queue'] or '-'}")
        if out["alternatives"]:
            st.write("**Other likely types:** " + ", ".join(out["alternatives"]))

        with st.expander("Retrieved candidates (what the LLM chose from)"):
            by_type = get_assistant().by_type
            st.dataframe(pd.DataFrame({
                "rank": range(1, len(out["candidates"]) + 1),
                "request type": out["candidates"],
                "similarity": out["candidate_scores"],
                "department": [by_type[t]["department"] for t in out["candidates"]],
                "description": [by_type[t]["description"] for t in out["candidates"]],
            }), hide_index=True, use_container_width=True)

# ------------------------------------------------------------------ Tab 2
with staffing_tab:
    plan, is_placeholder = load_plan()
    missing = [c for c in PLAN_COLUMNS if c not in plan.columns]
    if missing:
        st.error(f"The staffing plan file is missing columns: {', '.join(missing)}")
        st.stop()
    if is_placeholder:
        st.warning("**Placeholder data.** The staffing model's plan "
                   "(`data/processed/staffing_plan_2025.csv`) is not ready yet. These numbers "
                   "are a do-nothing baseline (January crews kept all year), not the plan.")

    f1, f2 = st.columns(2)
    scenarios = list(dict.fromkeys(plan.scenario))
    scenario = f1.selectbox("Scenario", scenarios, index=scenarios.index("base") if "base" in scenarios else 0)
    depts = f2.multiselect("Departments", DEPTS, default=DEPTS)

    view = plan[(plan.scenario == scenario) & plan.department.isin(depts)].copy()
    if view.empty:
        st.info("Pick at least one department.")
        st.stop()

    on_time = (view.on_time_pct * (view.closures + view.overdue)).sum() / (view.closures + view.overdue).sum()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total cost 2025", f"${view.cost.sum() / 1e6:,.1f}M")
    k2.metric("Crews hired / released", f"{view.hires.sum():,} / {view.releases.sum():,}")
    k3.metric("Overdue requests", f"{view.overdue.sum():,}")
    k4.metric("On-time (weighted)", f"{on_time:.0f}%")

    view["month_name"] = view.month.map(lambda m: MONTHS[m - 1])
    color = alt.Color("department:N", title="Department",
                      scale=alt.Scale(domain=list(DEPT_COLORS), range=list(DEPT_COLORS.values())))
    base = alt.Chart(view).encode(
        x=alt.X("month_name:N", sort=MONTHS, title=None),
        color=color,
        tooltip=["department", "month_name", "active_crews", "seasonal_crews", "hires", "releases",
                 alt.Tooltip("on_time_pct", title="on-time %")],
    )
    crews = (base.mark_line(strokeWidth=2) + base.mark_point(size=60, filled=True)).encode(
        y=alt.Y("active_crews:Q", title="Active crews"))
    st.subheader("Active crews by month")
    st.altair_chart(crews.properties(height=320), use_container_width=True)

    st.subheader("Plan table")
    table = view.drop(columns="month_name").rename(columns={
        "active_crews": "active crews", "seasonal_crews": "seasonal crews",
        "overdue_30plus": "30+ days overdue", "on_time_pct": "on-time %"})
    st.dataframe(table.drop(columns="scenario"), hide_index=True, use_container_width=True)
