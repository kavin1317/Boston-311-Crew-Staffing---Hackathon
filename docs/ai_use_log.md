# AI-Use Log

Every meaningful use of AI tools on this project. Required by the assignment's AI-use disclosure.

| Date | Step | Who | Tool | What AI did | What I reviewed / changed |
|------|------|-----|------|-------------|---------------------------|
| 2026-11-07 | Planning | Kavin (B) | Claude (claude.ai) | Drafted the project plan, team roles, file contracts, and `config.yaml` from the topic brief | TODO (Kavin): note what you reviewed or changed |
| 2026-11-07 | 1 – Setup | Kavin (B) | Claude Code | Checked the repo state, added `.env.example`, `.gitkeep` files and this log; ran a test chat call and a test embedding call | Confirmed `.env` stays untracked; both API calls worked (gpt-4o-mini reply; 1536-dim embedding) |
| 2026-11-07 | 2 – Request types | Kavin (B) | Claude Code | Checked the 2024 CSV columns; wrote `src/catalog.py` (DuckDB query → `type_routing.json`: primary department/queue and their shares per type; shares under 0.5% lumped into `_other`) | TODO (Kavin): review the output; decide how to score queues that are split by district |
| 2026-11-07 | A1–3 – Data load and clean | Jogeashwini (A) | Claude Code | Wrote `src/data.py`: DuckDB load of 2023–24, filter to four departments, check against the brief, clean file `requests_clean.parquet` | Checked counts match the brief exactly (157,610 / 81,447 / 18,736 / 18,624) and on-time shares (90 / 50 / 93 / 72%); decided to count all requests, including the ones with no target date |
| 2026-11-07 | A4 – Type catalog | Jogeashwini (A) | Claude Code + gpt-4o-mini (API) | Claude Code added the catalog code to `src/catalog.py` (kept the old routing code). gpt-4o-mini writes one plain-language sentence and cleaned example titles for each of the 170 types (structured output, temperature 0, answers cached in `data/cache/llm/`). Output: `data/processed/type_catalog.json` | First test made up example titles, so the prompt now includes the city category and the code drops any title that isn't in the data. TODO (Jogeashwini): review the 170 descriptions |
