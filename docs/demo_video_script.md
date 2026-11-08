# 3-minute demo video — script draft **[AI-drafted]**

About 1 minute per person. Record each part separately, then join them. Before uploading, check
that the link opens while logged out, and add it to the README.

## 0:00–1:00 · Person A — the data and the forecast
- The problem: 4 departments, ~276k requests in 2024, crews to plan month by month for 2025.
- Show the counts matching the brief (notebook Section 1).
- Show `docs/figures/monthly_demand.png`: seasonality (Parks triples Jan → Jun, Public Works peaks Aug–Sep).
- Show `docs/figures/forecast_2025.png` and the forecast check (error by department).

## 1:00–2:00 · Person B — the intake assistant
- `streamlit run app/app.py`, Tab 1.
- Type a clear message: "theres a huge pothole on centre st by the T" → Pothole Repair, Public Works, queue.
- Type an unclear one: "something is wrong with the tree out front" → it asks a follow-up question.
- Open "Retrieved candidates": embeddings find the 8 closest types, the LLM picks one, department and queue come from the data.
- One line on results: V1 → V2 → V3 table; V3 gets the right type into the top 3 for every test message and asks on every unclear one.

## 2:00–3:00 · Person C — the staffing plan
- Tab 2: crews by month for each department, hires/releases, Parks seasonal crews.
- Switch scenarios: base, +20%, −20%, snowstorm, heat wave; what changes and what it costs.
- Close with the key recommendation.
