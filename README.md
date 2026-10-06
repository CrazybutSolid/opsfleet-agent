# Opsfleet Insights: retail data-analysis chat agent

A chat agent that lets non-technical executives ask questions about sales, products and customers in
plain English, discuss the answers, and get saved reports with action items. It is built on **Gemini**
and **BigQuery** (`bigquery-public-data.thelook_ecommerce`), orchestrated with **Google ADK**.

The agent writes BigQuery SQL on the fly, runs multi-step analyses ("…and *why*?"), and learns from
analyst-authored *golden trios* (question → SQL → report). Safety does not depend on the model
behaving. The data boundary is enforced in code:

- every query is validated and rewritten onto **governed, PII-free, per-user-scoped views**;
- personal data is filtered again on the way out;
- the model can **propose** deleting reports, but only the user's own `confirm` deletes anything.

| | |
|---|---|
| **Design** | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): production HLD on GCP, with diagrams for request flow, delete protocol and learning loops |
| **Technical explanation** | [docs/TECHNICAL.md](docs/TECHNICAL.md): technology choices, data flow, and one section per requirement 1–8 (production design · prototype · design-only) |
| **Assumptions** | [docs/ASSUMPTIONS.md](docs/ASSUMPTIONS.md): questions for OpsFleet and the default chosen for each |
| **Example run** | [docs/example_run.md](docs/example_run.md): live transcript (real Gemini + BigQuery) |

## What the prototype does

| Requirement | Prototype | Proven by |
|---|---|---|
| Analysis: customer behaviour, product comparison with "why", time metrics, schema questions, multi-step, follow-ups, reports with action items | ADK agent with `run_sql`, `describe_data`, `save_report` tools; golden trios retrieved per question | `tests/test_agent_flows.py`, `docs/example_run.md` |
| **Safety & PII** | Input guard (injection / PII requests / off-topic) → policy-first prompt → sqlglot SQL guard (SELECT-only, single statement, 4 allowed tables, no PII columns) → governed views (PII columns don't exist, rows filtered to the user's products) → output row/text redaction | `tests/test_safety.py` |
| **High-stakes oversight** | "Delete reports mentioning X / from this conversation" → exact list from the DB → explicit `confirm` within 120 s → transactional, owner-only delete → audit log | `tests/test_destructive_ops.py` |
| **Resilience** | Self-correction on SQL errors and empty results (max 3 per turn), dry run + `maximum_bytes_billed`, Gemini backoff → fallback model, rate limiter, BigQuery circuit breaker, CLI that can't crash | `tests/test_resilience.py` |
| **Observability** | One JSON trace per turn (prompt, model calls, tool calls, SQL, rows, bytes, tokens, latency, retries, errors), `/trace` and `/metrics` | `tests/test_observability.py` |
| Hybrid intelligence (light) | 13 verified golden trios, BM25 retrieval, hot reload, feedback → candidate queue | `tests/test_golden.py` |
| Persona (light) | `config/persona.md` re-read on change, no restart needed | `test_persona_file_is_reread_without_restart` |
| User preferences (light) | Tables vs bullets, depth, visuals, focus: persisted per user and applied every turn | `test_preferences_are_learned_persisted_and_applied` |

## Setup (fresh clone)

**Prerequisites:** Python 3.11+ (tested on 3.14), the [gcloud CLI](https://cloud.google.com/sdk/docs/install),
a Google Cloud project (free BigQuery sandbox is enough), and a Gemini API key from
[Google AI Studio](https://aistudio.google.com/apikey).

```bash
git clone <this repo> opsfleet-agent && cd opsfleet-agent
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # also installs this package and the `opsfleet` command

# BigQuery: authenticate Application Default Credentials and pick a project to bill queries to
gcloud auth application-default login
gcloud services enable bigquery.googleapis.com --project <your-project-id>

# Configuration
cp .env.example .env
#   GOOGLE_API_KEY=<your AI Studio key>
#   GOOGLE_CLOUD_PROJECT=<your-project-id>

pytest                                    # offline: no keys or network needed
opsfleet --user alice                     # start chatting
```

Users are defined in [`config/users.toml`](config/users.toml):

| user | role | data scope |
|---|---|---|
| `alice` | Denim category lead | categories Jeans, Pants, Pants & Capris |
| `bob` | Swim & Activewear lead | categories Swim, Active |
| `carol` | Brand partnerships | brands Calvin Klein, Tommy Hilfiger, Levi's |
| `harry` | VP Commercial | all products |

Environment variables are documented in [`.env.example`](.env.example): models, the requests-per-minute
throttle, the BigQuery byte cap and the max rows.

## Example session

```text
$ opsfleet --user alice
╭──────────────────────────────────────────────────────────────────────────╮
│ Opsfleet Insights · signed in as Alice (Denim category lead)             │
│ Data scope: category in (Jeans, Pants, Pants & Capris) · /help           │
╰──────────────────────────────────────────────────────────────────────────╯
you › Compare Levi's and Calvin Klein in my categories. Which performs better and why?
  … answer with numbers, decomposition (assortment × price × volume), caveats …
trace 6c1e0b2a91f4 · answered · 3 SQL · 14.2s · 21,904 tokens

you › Give me the email addresses of those top customers.
╭──────────────────────────────────────────────────────────────────────────╮
│ I can't share personal customer data such as names, emails, …            │
╰──────────────────────────────────────────────────────────────────────────╯

you › Delete all the reports we made in this conversation
      About to permanently delete 1 report(s): reports created in this conversation
       id  title                         created
        7  Denim Q3 2026 Performance     2026-10-06 11:02
Type  confirm  within 119s to delete these. Anything else cancels.
you › confirm
Deleted 1 report(s).

you › /trace          # full record of the last turn
you › /metrics        # success / refusal / error rates, latency p50/p95, tokens, self-correction rate…
```

The full live transcript, including the out-of-scope and injection attempts, the saved report, metrics
and a sample trace, is in [docs/example_run.md](docs/example_run.md). To regenerate it:
`python scripts/smoke.py` (≈ 5 minutes; paced to stay within free-tier limits).

### CLI commands

| command | |
|---|---|
| `/reports [text]`, `/report <id>` | list / show your saved reports |
| `/prefs`, `/prefs set format table` | show / set preferences (`format`, `depth`, `visuals`, `focus`) |
| `/trace [id\|last] [--full]` | inspect a turn's trace |
| `/metrics [--all]` | aggregate agent metrics |
| `/feedback good\|bad [comment]` | rate the last answer (good + SQL → golden-trio candidate) |
| `/audit` | your report audit log |
| `/new`, `/whoami`, `/help`, `/quit` | |

## Editing behaviour without code

- **Tone / report style:** edit [`config/persona.md`](config/persona.md). The next message uses it, with no restart.
  Safety rules are not in this file and can't be weakened from it.
- **Golden knowledge:** append a line to [`data/golden_trios.jsonl`](data/golden_trios.jsonl). It's picked up live.
- **Who sees what:** edit [`config/users.toml`](config/users.toml).

## Repository layout

```
src/opsfleet_agent/   agent code (see docs/TECHNICAL.md §2 for a module map)
tests/                117 offline tests (scripted LLM + fake BigQuery)
scripts/smoke.py      live end-to-end run → docs/example_run.md
config/               persona.md (tone), users.toml (scopes)
data/                 golden_trios.jsonl (13 verified trios)
docs/                 ARCHITECTURE.md, TECHNICAL.md, ASSUMPTIONS.md, example_run.md
var/                  runtime state, gitignored (SQLite, traces.jsonl, feedback)
```
