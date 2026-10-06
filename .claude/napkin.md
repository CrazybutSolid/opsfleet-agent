# Napkin: opsfleet-agent

## Environment & tooling
- Python 3.14 venv in `.venv`; `pip install -r requirements.txt` installs the package editable (`-e .`) and the `opsfleet` command.
- gcloud lives at `/opt/homebrew/bin/gcloud`; `bq` needs `PATH=/opt/homebrew/share/google-cloud-sdk/bin:$PATH`.
- BigQuery billing/quota project: `gen-lang-client-0796834285` (the "Gemini API" project). ADC has no default project, so `GOOGLE_CLOUD_PROJECT` must be set in `.env`.
- `pyproject` already sets `addopts = "-q"`; adding `-q` again hides the "N passed" summary. Use `-o addopts="" -q` when scripting counts.

## Gemini / ADK
- Free-tier capacity varies a lot by model: `gemini-3.8-flash` / `3.7-flash` were saturated (503s, a 90 s hang) on 2026-10-06; `3.6-flash` was fast. Probe with a tiny `generate_content` loop before picking a primary.
- `gemini-2.5-*` returns 404 "no longer available to new users". 404 must skip to the fallback, not fail the turn.
- ADK callable `instruction` providers bypass `{var}` state injection, so braces in SQL/golden text are safe.
- `LlmCallsLimitExceededError` (from `google.adk.agents.invocation_context`) is raised when `RunConfig.max_llm_calls` is hit. Catch it.

## sqlglot
- v30 stores the WITH clause under arg key `with_`, not `with` (`tree.set("with", ...)` silently does nothing).
- BigQuery string escaping is `'Levi\'s'` or `"Levi's"`; `''` is a parse error.

## Gotchas
- The user edits `.env` in an editor: a buffer saved later can overwrite programmatic edits. Re-check with `grep -E "^OPSFLEET_" .env`, never cat the file (it holds the key).
- Smoke run (`scripts/smoke.py`) takes about 5 min and writes `docs/example_run.md`; it uses the real `var/` DB.
