# AI-Use Log

Every meaningful use of AI tools on this project. Required by the assignment's AI-use disclosure.

| Date | Step | Who | Tool | What AI did | What I reviewed / changed |
|------|------|-----|------|-------------|---------------------------|
| 2026-11-07 | Planning | Kavin (B) | Claude (claude.ai) | Drafted the project plan, team roles, file contracts, and `config.yaml` from the topic brief | TODO (Kavin): note what you reviewed or changed |
| 2026-11-07 | 1 – Setup | Kavin (B) | Claude Code | Checked the repo state, added `.env.example`, `.gitkeep` files and this log; ran a test chat call and a test embedding call | Confirmed `.env` stays untracked; both API calls worked (gpt-4o-mini reply; 1536-dim embedding) |
| 2026-11-07 | 2 – Request types | Kavin (B) | Claude Code | Checked the 2024 CSV columns; wrote `src/catalog.py` (DuckDB query → `type_routing.json`: primary department/queue and their shares per type; shares under 0.5% lumped into `_other`) | TODO (Kavin): review the output; decide how to score queues that are split by district |
