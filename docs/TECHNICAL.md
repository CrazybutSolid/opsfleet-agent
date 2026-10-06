# Technical design

> **Reading guide.** §1 explains the technology choices. §2–3 describe the components and how data
> moves between them. §4–11 cover requirements 1–8, one section each. Every requirement section has
> the same three parts: **Production design**, **In the prototype** (with file references and the tests
> that prove it), and **Design-only**. §12 covers extensibility, §13 cost and rate limits, §14 the threat
> model, §15 known limitations. The diagrams are in [ARCHITECTURE.md](ARCHITECTURE.md).

---

## 1. Technology choices

### 1.1 Cloud: Google Cloud

The data is already in BigQuery and the brief asks for Gemini, so GCP keeps data, model and compute
inside one security perimeter (VPC-SC, one IAM model, one audit trail). Each service and why it was picked:

| Need | Service | Why this one (and what else was considered) |
|---|---|---|
| Run the agent | **Cloud Run** | Stateless HTTP container, scales to many sessions, per-request billing, min-instances for warm starts, native IAP. *Vertex AI Agent Engine* is the managed alternative (sessions, memory bank and tracing built in). ADK deploys to both, so this choice can be revisited without code changes. GKE would be overkill for one service. |
| LLM | **Gemini on Vertex AI** | Same models as AI Studio, but authenticated with a service account (no API keys), regional data residency, provisioned throughput for the primary model, and access to Model Armor and the Evaluation service. The prototype uses an AI Studio key, as the brief asks. |
| Warehouse | **BigQuery** | Given. Also gives the strongest *server-side* controls: authorized views, row access policies, column-level policy tags, `maximum_bytes_billed`, free dry runs, job labels for cost attribution. |
| App state | **Firestore** | Per-user documents (reports, preferences, pending deletions, persona versions), ACID transactions (needed by the two-phase delete), TTL policies (expire confirmations), serverless. Cloud SQL (Postgres) is the alternative if reporting joins on app data become important. |
| Golden bucket | **Cloud Storage** + **Vertex AI Vector Search** | GCS is the "data lake" in the brief: analysts drop JSONL or CSV and object versioning gives history. Vector Search serves low-latency filtered top-k. Below about 10k trios, BigQuery `VECTOR_SEARCH` is an acceptable, cheaper option. |
| Embeddings | **gemini-embedding-001** | Strong multilingual retrieval quality, and `task_type=RETRIEVAL_QUERY/DOCUMENT` asymmetric embeddings fit question-to-trio matching. |
| Async glue | **Pub/Sub** + **Cloud Run jobs** + **Cloud Scheduler** | Ingestion on GCS finalize events, a nightly learning-loop job, and turn events fanned out to BigQuery without slowing the chat path. |
| Guardrails | **Model Armor**, **Sensitive Data Protection (DLP)** | Managed prompt-injection/jailbreak screening, and PII inspection with InfoTypes. Both sit behind the cheap deterministic checks that the prototype implements. |
| Observability | **Cloud Logging**, **Cloud Trace**, **Cloud Monitoring**, BigQuery log sink, **Looker Studio** | ADK emits OpenTelemetry spans natively. Structured JSON logs become queryable traces in BigQuery. Monitoring alerts on log-based metrics. |
| Identity | **IAP** + Google Workspace | SSO with no auth code in the app. The verified identity, not anything typed in chat, selects the user's data scope. |
| Secrets / CI | **Secret Manager**, **Cloud Build**, **Artifact Registry** | Standard. Cloud Build runs the offline eval gate before any deploy. |

### 1.2 Models

| Role | Model | Reasoning |
|---|---|---|
| Primary (chat, tool calling, SQL writing) | **`gemini-3.6-flash`** | Recent Flash-class model: strong function calling and SQL, low latency, generous free tier. Analysis quality comes mostly from the multi-step tool loop and the golden examples, not from raw model size. *Measured, not assumed:* during the build, the newest `gemini-3.8-flash` and `gemini-3.7-flash` were saturated on the free tier (503s, one 90 s hang), while 3.6-flash answered in about 1.5 s. Switching to 3.8 is a one-line `.env` change once it has capacity. |
| Fallback | **`gemini-3.5-flash-lite`** | Different model with a *separate quota bucket*, so a 429 on the primary doesn't also block the fallback. Cheaper and faster, so it's good enough to finish a turn during an incident. |
| Offline judge (QA, design-only) | Gemini Pro-class | Used only in CI and nightly evaluation, where quality matters more than latency and cost. |
| Embeddings (design-only) | gemini-embedding-001 | See above. |

Temperature is 0.2: the answers should be reproducible analysis, not creative writing. Models are set
through environment variables (`OPSFLEET_MODEL`, `OPSFLEET_FALLBACK_MODEL`), so upgrading a model
needs no code change. In production, a model upgrade has to pass the eval gate (§9).

### 1.3 Framework: Google Agent Development Kit (ADK)

**Why ADK**

- **Built for Gemini and GCP.** ADK has native Gemini function calling and deploys to Cloud Run or
  Agent Engine. Its session services include Vertex AI sessions, and it emits OpenTelemetry to Cloud
  Trace. The production story is "the same agent object, different services", not a rewrite.
- **The right extension points for this brief.** *Instruction providers* rebuild the system prompt
  on every model call, which handles the runtime persona, preferences and golden trios. *Tools* are
  plain Python functions with docstrings. The `BaseLlm` abstraction let me wrap Gemini with retry,
  backoff and fallback logic transparently (`llm.py`), and swap in a scripted fake model for offline
  tests. `RunConfig.max_llm_calls` adds a hard stop on runaway loops.
- **Small surface.** The agent is about 40 lines of wiring (`agent.py`). The policy logic (guards,
  scoping, deletion protocol) is plain, framework-agnostic Python that can be unit-tested without
  any LLM.
- **Alternatives considered.**
  - *LangGraph* has excellent explicit state machines and interrupts, but it is a second abstraction
    over Gemini and needs more glue for GCP deployment and tracing.
  - *Raw google-genai SDK* has no framework at all. It's simplest today, but sessions, tool
    dispatch, tracing and deployment would all have to be re-implemented.
  - *CrewAI and other multi-agent frameworks* add multi-agent overhead this problem does not need yet.

**What is deliberately *not* delegated to the framework.** The destructive-action confirmation, data
scoping, PII handling and retry budgets are enforced in our own code, outside the LLM's control.
ADK has a tool-confirmation feature, but a deterministic gate that the model cannot satisfy
by itself is a stronger guarantee (§6).

**My experience with it:** My production agent work has mostly been in **LangGraph and LangChain**.
Most recently that was an agentic software factory: LangGraph carries a ticket's acceptance criteria to a
reviewed pull request through gated, test-first, human-approved phases. Before that I built an LLM
assessment engine with retrieval and provenance on every claim. At Orange Business I built GenAI backends
on **GCP (Vertex AI, Cloud Run, IAM)** with Gemini, GPT and Claude, and I am a Google Cloud Professional
Cloud Architect (2024). **ADK is newer to me than LangGraph.** I chose it here deliberately, despite that,
because it is the native fit for Gemini and for deploying on GCP. The patterns this design relies on carry
over directly from my LangGraph work: tool calling, deterministic gates around the model,
human-in-the-loop confirmation and evaluation-driven iteration.

### 1.4 Supporting libraries

| Library | Use |
|---|---|
| `sqlglot` | Parses the model's SQL into an AST so it can be validated and rewritten structurally, not with regexes (BigQuery dialect). |
| `google-cloud-bigquery` | Dry runs, `maximum_bytes_billed`, job labels and timeouts. |
| `rich` | CLI rendering: markdown answers, deletion-preview tables, trace JSON. |
| SQLite (stdlib) | Prototype stand-in for Firestore. The same two-phase/audit semantics apply, with no setup needed. |

---

## 2. Components

```
src/opsfleet_agent/
├── cli.py                 CLI loop + slash commands (/trace, /metrics, /reports, /prefs, /feedback, /audit)
├── agent.py               AgentService: one turn end to end (gates → ADK loop → output filter → trace)
├── llm.py                 ResilientGemini (ADK BaseLlm): rate limiter, backoff, fallback model, per-call tracing
├── tools.py               run_sql, describe_data, save_report, list_reports, request_report_deletion, remember_preference
├── prompts.py             policy (code-owned) + persona file (business-owned) + user + prefs + golden trios
├── bq.py                  BigQueryRunner: dry run, byte cap, classification, circuit breaker, cache
├── catalog.py             governed schema + PII column registry (single source of truth)
├── security/
│   ├── input_guard.py     injection / PII-request / off-topic screening (deterministic)
│   ├── sql_guard.py       sqlglot validation + governed-view rewrite + scope checks
│   ├── scope.py           per-user product entitlements
│   └── pii.py             output row/text redaction
├── storage/
│   ├── reports.py         saved reports, two-phase delete, audit log (SQLite)
│   └── preferences.py     typed per-user preference profile
├── knowledge/golden.py    golden trio store: BM25 retrieval, hot reload, candidate queue
└── observability/tracing.py   TurnTrace (JSONL) + aggregate metrics
config/persona.md  config/users.toml  data/golden_trios.jsonl
```

**How components communicate.** In the prototype, all components run in-process and the agent
service calls them directly. In production:

| From → To | Protocol | Notes |
|---|---|---|
| Client → chat-api | HTTPS + SSE (streaming tokens), via LB + IAP | IAP injects the signed user identity (`x-goog-iap-jwt-assertion`). |
| chat-api → agent-service | HTTP/gRPC (Cloud Run service-to-service, IAM-authenticated) | Could be one service. Split so the API can stream and handle auth while the agent scales separately. |
| agent-service → Gemini | Vertex AI `generateContent` (REST/gRPC) | Service account. Retries, fallback and rate limiting in `ResilientGemini`. |
| agent-service → BigQuery | BigQuery Jobs API | Dedicated service account with `bigquery.jobUser` on the billing project and `dataViewer` **only** on the governed dataset. |
| agent-service → Firestore | Firestore client (gRPC) | Transactions for delete confirmation. |
| agent-service → Vector Search | `findNeighbors` (gRPC) | Filter on `approved=true` and the user's domain. |
| agent-service → Pub/Sub | publish turn events / audit events (async, fire-and-forget with local buffer) | Never on the critical path. |
| GCS → ingestion job | Eventarc (object finalize) → Cloud Run job | |

---

## 3. Data flow of a turn

The step numbers match the sequence diagram in ARCHITECTURE.md §2. Prototype code is in `agent.py::_chat`.

1. **Identity and scope.** The user ID comes from IAP (prototype: `--user`). The `UserScope` record
   (allowed departments, categories and brands) is loaded from the entitlements store (prototype:
   `config/users.toml`). The model never chooses or sees a way to change it.
2. **Pending destructive action.** If this user has a pending deletion plan, the message is matched
   against a strict confirmation pattern (`confirm`, `yes`, `delete them`, …). If it matches, the plan
   runs in a transaction and the turn ends without calling the LLM. If it doesn't match, the plan is
   cancelled (and audited) and the message is processed normally.
3. **Input guard.** Deterministic screening for prompt injection, requests for personal data and clearly
   off-topic tasks. A blocked message gets a polite refusal. The model is never called and no data is
   touched (production adds Model Armor here).
4. **Context assembly.** The top-2 golden trios are retrieved for the message. The user's preference
   profile and the current persona are read. `build_instruction()` composes policy → persona → user and
   scope → preferences → data catalogue → golden examples.
5. **Agent loop (ADK).** The model either answers directly or calls tools. `run_sql` is the core tool:
   1. **guard**: sqlglot parse; one statement; SELECT only; allowed tables only; no PII columns;
      literal filters inside the user's scope;
   2. **rewrite**: bind `orders/order_items/products/users` to governed CTEs that project only non-PII
      columns and apply the user's row filter; enforce a `LIMIT`;
   3. **dry run**: free; validates the SQL and estimates bytes; refuses anything above the byte cap;
   4. **execute** with `maximum_bytes_billed`, a timeout and labels;
   5. **row filter**: drop PII-named columns and redact PII-looking values;
   6. **return** at most 50 rows plus `row_count` to the model, or a *classified* error
      (`PARSE_ERROR`, `BQ_INVALID_SQL`, `EMPTY_RESULT`, `OUT_OF_SCOPE`, `BQ_UNAVAILABLE`, …) with the
      remaining retry budget.

   The loop is bounded by the SQL failure budget (3 per turn), the tool budget (10 per turn) and
   `max_llm_calls`.
6. **Output filter.** The final text is redacted again (emails, phones, addresses, coordinates, IPs).
7. **Delivery.** The answer is sent. If a deletion was planned this turn, the CLI renders the exact list
   of reports from the database (not the model's wording) and asks for confirmation.
8. **Trace.** One JSON record per turn is appended (prototype: `var/traces.jsonl`). In production it is
   logged to Cloud Logging, with a sink to BigQuery.

**What is stored where during a turn**

| Step | Reads | Writes |
|---|---|---|
| 1 | entitlements | none |
| 2 | pending_deletions | reports (delete), pending_deletions (status), audit_log |
| 4 | golden trios, preferences, persona | none |
| 5 | BigQuery (governed) | reports (`save_report`), pending_deletions (`request_report_deletion`), preferences (`remember_preference`), audit_log |
| 8 | none | traces, feedback (on `/feedback`) |

---

## 4. Requirement 1: Hybrid intelligence (Golden Bucket)

### Production design

**Trio format.** Each trio is one JSONL object: `id`, `question`, `sql` (written against the governed table
names), `report`, `tags`, `author`, `approved`, `created_at`, `valid_from/valid_to` (the business
definitions in force), `domain` (e.g. merchandising, finance), and `supersedes` (the id of an older trio).

**Ingestion pipeline** (Eventarc on `gs://opsfleet-golden/trios/**` → Cloud Run job):

1. **Schema validation**: required fields; reject the file and notify the uploader if invalid.
2. **SQL validation**: run the trio's SQL through the same `sql_guard`, then a BigQuery **dry run**
   against the governed views. Trios whose SQL no longer compiles (because the schema drifted) are
   quarantined, not indexed. This keeps the golden set executable.
3. **PII scrub** of the report text (DLP). Analyst reports sometimes quote real customers.
4. **Deduplication**: if a new trio's embedding has cosine similarity > 0.95 with an existing one,
   it's treated as a new *version*: the old one gets `valid_to`, and the new one is linked with `supersedes`.
5. **Embedding** of `question + tags` (and separately the report summary) with `gemini-embedding-001`,
   task type `RETRIEVAL_DOCUMENT`.
6. **Upsert** into Vector Search with metadata restricts (`approved`, `domain`, `valid`), plus a BigQuery
   catalogue table (`opsfleet_ops.golden_trios`) for analytics and audit.

**Retrieval at query time.**

1. Embed the user question (`RETRIEVAL_QUERY`).
2. **Hybrid search**: vector top-20 plus BM25 top-20 (keywords like brand names or "Q1" matter), merged with
   reciprocal rank fusion and filtered on `approved=true`, `valid`, and the user's domain.
3. Optional re-rank of the top 10 with a small model, keeping the top 2–3 that score above a threshold.
   Fewer examples is better than irrelevant ones, because they bias the SQL.
4. Inject them into the system instruction as **examples of interpretation**: business definitions
   (e.g. "churn = active in the prior 3 months but not this month"), which filters to apply
   (exclude cancelled and returned items), the shape of the analysis (decompose a gap into volume ×
   price × mix) and report format. The policy tells the model to *adapt* the SQL and never reuse the
   historical numbers.
5. Log the retrieved trio IDs on the trace, so their usefulness can be measured.

**Updating over time.**

- *Human-authored*: analysts keep uploading. Versioning and `supersedes` let definitions change
  (e.g. a new churn definition) without contradictory examples.
- *From usage* (system-level learning, §7): answers rated 👍 that are backed by successful SQL become
  **candidate** trios, which go to an analyst review queue. Only approved candidates are indexed. The
  model never writes to the golden set directly, which prevents a feedback loop that amplifies errors.
- *Hygiene*: a nightly job re-runs the dry-run validation on every trio (catches schema drift). It
  computes **usage and win-rate per trio** (how often it was retrieved in turns rated 👍 vs 👎)
  and flags trios that are never retrieved or are associated with bad answers, for review or retirement.
- *Coverage*: clusters of production questions with no trio above the similarity threshold are
  reported to the analytics team as "write a trio for this".

### In the prototype

- `data/golden_trios.jsonl`: **13 trios** written for this dataset. Every SQL was run against
  BigQuery and the reports quote the real results. They capture lessons such as "a partial month makes
  MoM growth look like a crash", "decompose a brand gap into SKUs × productivity × price", and "Texas
  isn't under-spending per head; it has fewer customers".
- `knowledge/golden.py`: BM25 over question and tags, with light stemming, a region normaliser and
  business synonyms. It's deterministic and uses **no API calls** (to protect the free-tier quota). The
  file is **hot-reloaded** on change, and trios with `approved: false` are ignored.
- Top-2 hits above a score threshold are injected into the instruction (`prompts.py`). The IDs and
  scores are recorded on the trace (`golden`).
- `/feedback good` queues the last answer (question + SQL + answer) to `var/golden_candidates.jsonl`
  with `approved: false`, for review.
- Tests: `tests/test_golden.py` (seed validity: every trio's SQL passes the guard; paraphrase retrieval;
  no-match → nothing injected; hot reload; unapproved ignored) and
  `tests/test_agent_flows.py::test_golden_trios_are_retrieved_and_injected`.

### Design-only

GCS + Eventarc ingestion, embeddings, Vector Search, hybrid fusion, re-ranker, nightly validation, and
the per-trio win-rate analytics.

---

## 5. Requirement 2: Safety and PII masking

Defense in depth, so that no single layer is trusted.

| # | Layer | What it stops | Prototype | Production |
|---|---|---|---|---|
| 1 | **Identity → scope** | Users choosing their own entitlements | `--user` + `users.toml`. Scope is never read from chat. | IAP identity → entitlements table |
| 2 | **Input guard** | Prompt injection, jailbreaks, explicit PII requests, off-topic tasks | `security/input_guard.py` regex classifier; the model is never called on a block | + Model Armor (prompt injection & jailbreak, malicious URLs) |
| 3 | **Policy prompt** | Subtler off-topic or social engineering that passes layer 2 | Code-owned hard rules placed *before* the business-editable persona | same |
| 4 | **SQL guard** | Writes, multi-statement, other datasets/tables, `INFORMATION_SCHEMA`, `EXTERNAL_QUERY`/ML functions, PII columns, filters on out-of-scope products | `security/sql_guard.py` (sqlglot AST) | same |
| 5 | **Governed views** (SQL-layer masking + row-level security) | `SELECT *`, `TO_JSON_STRING(t)`, joins that would expose PII or other users' products | Each query is rewritten so every table name binds to a CTE that **projects only non-PII columns** and applies the user's **row filter** | BigQuery authorized views + **row access policies** + **policy tags** on PII columns; the agent's service account has no access to raw tables |
| 6 | **Row filter** | Anything PII-shaped that still comes back (new column, mis-configured view) | `security/pii.py::filter_rows` drops PII-named columns, redacts values | + DLP inspect |
| 7 | **Output filter** | Model echoing PII (e.g. pasted by the user) | `redact_text` on the final answer, saved reports, trace text and tool args | + DLP |
| 8 | **Least privilege** | Everything above failing at once | n/a | Service account: `jobUser` + `dataViewer` on the governed dataset only. VPC-SC perimeter. Read-only by IAM, not just by guard. |

**PII registry.** `catalog.PII_COLUMNS` = `first_name, last_name, email, street_address, postal_code,
latitude, longitude, user_geom` (+ `ip_address`, `phone` for future tables). City, state, country, age, gender and
traffic source stay available because they're needed for the analyses in the brief ("users in state X").
They aren't direct identifiers, and results are aggregated. Customers are referred to only by numeric ID.
(Small-cell suppression, i.e. refusing groups with fewer than k customers, is in the design-only list.)

**Per-user scope ("products related to him").** A scope restricts `products` by department, category
and/or brand. Every other table is filtered *through* it:
`order_items` → items of in-scope products; `orders` → orders containing such items; `users` → customers
who bought them. So even aggregates ("total revenue", "number of customers") only cover the user's
slice. Explicit filters on out-of-scope values (`WHERE category = 'Swim'` for the denim lead) are refused
with an `OUT_OF_SCOPE` explanation instead of a silently empty result. The governed views guarantee
no leakage anyway. Example users: `alice` (Jeans, Pants), `bob` (Swim, Active), `carol` (Calvin Klein,
Tommy Hilfiger, Levi's), `harry` (executive, full catalogue).

**Off-topic and injection.** Blocked deterministically when unambiguous (layer 2). Otherwise the hard
rules in the system instruction apply (layer 3). Even a fully jailbroken model can still only call tools
that run guarded, scoped, read-only SQL and can't delete anything without the user (§6). That is the
point of making the tools the safety boundary, not the prompt.

**Tests**: `tests/test_safety.py` (63 cases): non-SELECT, multi-statement, foreign tables/datasets,
every PII column, `SELECT *` and `TO_JSON_STRING` exposure, reserved CTE shadowing, scope rewrite for
every table, brand scope escaping, out-of-scope refusal end to end, row/text redaction, a "leak
anyway" scenario (BigQuery returns PII and the model echoes it: neither the user nor the trace sees it),
injection/PII/off-topic blocking without calling the model, and no false positives on legitimate
questions.

**Design-only**: Model Armor, DLP, BigQuery-native row access policies and policy tags, VPC-SC,
k-anonymity small-cell suppression.

---

## 6. Requirement 3: High-stakes oversight (deleting saved reports)

**Principle: the model can propose a deletion but never perform one.** Deletion is a two-phase protocol.
Phase 2 is triggered only by the user's own next message, matched deterministically, outside the LLM.

1. **Plan** (`request_report_deletion` tool, the only deletion-related tool the model has):
   - resolves the criteria ("mentioning X" = case-insensitive substring of title or body; "this
     conversation" = `session_id`) **only among the requesting user's reports**;
   - snapshots the exact report IDs into a `pending_deletions` row with `owner`, `criteria`, `expires_at`
     (TTL 120 s) and a random token; a newer plan supersedes an older pending one;
   - audits `delete_requested`.
2. **Preview**: the UI renders the snapshotted reports from the database (ID, title, created date)
   and says *"Type `confirm` within 120 s to delete these. Anything else cancels."* The list comes from
   the database, so a model hallucination can't change it.
3. **Confirm** (`ReportStore.confirm_deletion`, in one transaction): plan exists and is pending → actor ==
   owner → not expired → delete only the snapshotted IDs that are *still owned by the actor* (reports
   created after the preview are never included) → mark executed → audit `delete_executed` with IDs.
4. **Anything else** cancels the plan (audit `delete_cancelled`) and the message is then handled as a
   normal question, so the user isn't stuck in a modal. "no"/"cancel" just cancel.
5. **Expiry**: a late "confirm" returns *"That confirmation expired, so nothing was deleted"* and audits
   `delete_expired`. The token can't be reused.

**Why this doesn't break UX.** One natural sentence to request, one word to confirm, a clear list,
and no second question from the model (the policy tells it not to re-confirm). Moving on simply cancels.

**Production additions**: soft delete (tombstone + 30-day restore window, `/undo`), Firestore TTL
on pending plans, audit events to an append-only BigQuery table via Pub/Sub, and a stricter flow above
a threshold (e.g. > 20 reports: type the number of reports to confirm).

**In the prototype**: `storage/reports.py`, `agent.py` (confirmation gate), `cli.py` (preview table),
`/audit` command. **Tests**: `tests/test_destructive_ops.py`: exact listing, nothing deleted before
confirmation, snapshot semantics, "this conversation", expiry, non-owner denied, every step audited in
order, supersede, cancel-and-continue, and that no tool lets the model delete by itself.

---

## 7. Requirement 4: Continuous improvement (learning loop)

### 7.1 User level

**Production design.** Each user has a typed **preference profile** (Firestore): `format` (table / bullets /
prose), `depth` (brief / standard / deep), `visuals` (text / charts), `focus` (revenue / margin /
customers / operations), each with `source` (explicit / inferred), a reinforcement count and a timestamp.
A typed profile, rather than free-form "memories", is predictable, editable by the user ("/prefs"), and
can be applied deterministically. Free-form, long-tail facts ("I report to the CFO on Mondays") go to
Agent Engine **Memory Bank**, which is retrieved per turn.

- *Explicit*: the user says it, the model calls `remember_preference`, and it applies from the next model
  call onward.
- *Implicit*: a nightly job over traces infers preferences from behaviour, e.g. ≥ 3 follow-ups like
  "shorter" / "just the number" → `depth=brief`, or repeated "as a table" → `format=table`. Implicit values
  are stored as `inferred` and **never override explicit ones**. The user is told once ("I noticed you
  prefer brief answers, so I'll keep them short").
- Applied by rendering the profile into the "This user's preferences" section of the instruction.

**In the prototype**: `storage/preferences.py` (SQLite, typed, explicit-wins), `remember_preference` tool,
`/prefs` and `/prefs set` commands, rendered into the instruction on every turn. Tests:
`test_preferences_are_learned_persisted_and_applied` (persisted across a new session/process, applied
to the prompt, isolated per user).

### 7.2 System level

The agent improves through **reviewed** changes to its knowledge, prompts and guards, not through
unsupervised self-modification:

| Signal (from traces + feedback in BigQuery) | Automated analysis (nightly job) | Improvement | Gate |
|---|---|---|---|
| 👍 answers with successful SQL | none | **Candidate golden trios** | Analyst approval |
| SQL error clusters (e.g. many `Unrecognized name: revenue`) | group by error code + normalised message | Catalogue descriptions / policy rules ("revenue = SUM(sale_price)") | Eval set must pass |
| Empty-result clusters | group by filter values | Value hints in `describe_data` (e.g. capitalised status values) | Eval |
| Low ratings / re-asks / abandonment | topic clustering (embeddings) | New trios, prompt changes | Eval + review |
| Guard false positives / negatives (from refusal feedback) | sample review | Guard patterns, Model Armor config | Safety eval set |
| Questions with no similar trio | coverage report | "Write a trio for this" tickets | Analyst |

Each fix also adds a **regression case to the offline eval set** (§9), so the system can't quietly go back to
an old mistake.

**In the prototype**: `/feedback good|bad [comment]` logs ratings against the trace ID
(`var/feedback.jsonl`), and a good rating queues a golden-trio candidate (`var/golden_candidates.jsonl`).
Traces already contain every field the nightly job needs (error codes, SQL, golden IDs, outcomes).
Test: `test_feedback_feeds_the_learning_loop`. The mining job itself is design-only.

---

## 8. Requirement 5: Resilience and graceful error handling

| Failure | Detection | Response | Cost control | Prototype | Test |
|---|---|---|---|---|---|
| SQL doesn't parse / not a SELECT | sqlglot | Error + hint back to the model; retry | Never reaches BigQuery | `sql_guard.py` | `test_sql_error_is_fed_back_and_self_corrected` |
| SQL invalid for BigQuery (unknown column, type error) | **dry run** (free) | Error message back to the model; retry | Dry run costs $0 | `bq.py` | same |
| Empty result | 0 rows | Treated as a failure with hints (capitalised statuses, date ranges, scope); retry or explain | Counts against the budget | `tools.py` | `test_empty_result_triggers_a_retry` |
| Repeated failures | per-turn counter | After **3** failures `run_sql` returns `gave_up` and the model explains what it tried | BigQuery hit at most 3 failed times per turn | `TurnState` | `test_self_correction_is_capped` |
| Policy refusal (out of scope, write attempt) | guard | No retry: budget set to exhausted | No wasted LLM calls | `tools.py` | `test_out_of_scope_question_end_to_end` |
| Runaway tool loop | tool counter + ADK `max_llm_calls` | Tool returns `blocked`; turn ends with a clear message | ≤ 14 LLM calls per turn | `tools.py`, `agent.py` | `test_runaway_tool_loop_is_stopped` |
| Expensive query | dry-run bytes > cap | Refused before execution; model asked to narrow it | `maximum_bytes_billed` (1 GB) also enforced server-side | `bq.py` | `test_dry_run_blocks_expensive_queries_before_they_run`, `test_execution_carries_maximum_bytes_billed_and_labels` |
| Repeated identical query | cache key = governed SQL | Served from a 10-min in-process cache (+ BigQuery's own 24 h result cache) | $0 | `bq.py` | `test_identical_queries_are_served_from_cache` |
| Gemini 429 / 5xx / timeout | `APIError.code`, 45 s per-call timeout | Exponential backoff with jitter (honours `retry in Ns` hints, max 20 s) → **fallback model**. A **per-model circuit breaker** opens after 2 consecutive failures, so for the next 120 s calls go straight to the fallback instead of paying for timeouts again on every model call. | Client-side **rate limiter** (sliding window, `OPSFLEET_LLM_RPM`) keeps within the free tier | `llm.py` | `test_gemini_429_backs_off_then_falls_back`, `test_failing_primary_is_skipped_on_later_calls`, `test_rate_limiter_keeps_under_rpm` |
| Model retired / not available to the key (404) | `APIError.code == 404` | Skip straight to the fallback (no pointless retries) | none | `llm.py` | `test_retired_model_skips_straight_to_fallback` |
| Gemini 4xx (bad key, bad request) | non-retryable code | Fail fast (no retry storm), clear message | none | `llm.py` | `test_non_retryable_error_fails_fast` |
| All models down | `LlmUnavailable` | "The AI service is temporarily unavailable… try again in a minute" | none | `agent.py` | `test_all_models_down_gives_clear_message` |
| BigQuery down / auth / quota | 5xx, 403, transport, missing credentials | Classified `unavailable`, model told *not* to retry and to tell the user; **circuit breaker** opens after 3 failures for 30 s | No hammering | `bq.py` | `test_bigquery_down_gives_a_clear_message`, `test_circuit_breaker_…`, `test_missing_credentials_is_unavailable_not_a_crash` |
| Bug inside a tool | any exception | Tool returns `TOOL_EXCEPTION`; turn continues | none | `tools.py::_traced` | `test_tool_exception_does_not_break_the_turn` |
| Bug anywhere in a turn | catch-all in `AgentService.chat` | "Something went wrong (trace id)"; error trace written | none | `agent.py` | `test_service_never_raises` |
| Bug in the UI loop | catch-all per input line | Message printed, prompt returns | none | `cli.py` | `test_cli_survives_errors_and_keeps_chatting` |
| Turn too slow | `asyncio.wait_for` (240 s) | Clear message | Stops spend | `agent.py` | none |

**Production additions**: Cloud Run min-instances and concurrency limits; per-user quotas (Firestore
counters) to stop one user exhausting the shared LLM and BigQuery budget; BigQuery
**reservations** or project-level custom quotas; provisioned throughput for the primary model;
Pub/Sub buffering so the observability pipeline can't block chat; and a regional fallback for Vertex AI.

---

## 9. Requirement 6: Quality assurance

### 9.1 Before deployment: the offline eval set (CI gate)

A versioned dataset (`evals/*.jsonl`, about 200 cases at launch, growing with every incident). Each case has
`question`, `user` (scope), optional conversation history, and expectations. Case families:

| Family | Example | How it's scored |
|---|---|---|
| **SQL correctness** | "Revenue by month in 2025" | **Execution match**: run the generated SQL and a reference SQL on the governed views and compare result sets (order-insensitive, numeric tolerance). This is more robust than comparing SQL text. |
| **Analysis quality** | "Why do CK and Levi's differ?" | LLM judge (Gemini Pro) with a rubric: uses the right definitions, decomposes the gap, every number traceable to a tool result (no hallucinated numbers), states caveats (partial month). Scores 1–5 per criterion. |
| **Report intent match** | "Q1 report with Q2 action items" | See 9.2. |
| **Safety** | injection, PII requests, out-of-scope products, write attempts | Must refuse; **zero** PII in output (DLP scan of the answer); 0 rows outside scope in executed SQL. **Any failure blocks release.** |
| **Destructive ops** | "Delete reports about X" in multi-turn scripts | Correct preview set; nothing deleted without confirm; expiry honoured. |
| **Resilience** | injected faults (429s, BigQuery 500, empty results) | Graceful message, budget respected, no crash. |
| **Persona** | same question under two personas | Judge checks tone compliance; facts unchanged. |

Thresholds per family (e.g. execution match ≥ 85 %, judge ≥ 4.0, safety = 100 %) run in Cloud Build on
every prompt, model or code change via the **Vertex AI Gen AI Evaluation Service**. A model upgrade is
just another change that must pass. ADK's `adk eval` format can hold the multi-turn cases with expected
tool trajectories, which lets the gate also check *how* the agent got there (e.g. it called `describe_data`
for schema questions and never ran SQL for a refusal).

**In the prototype**: `pytest` (120 offline tests with a scripted LLM and fake BigQuery) is the
deterministic layer of this gate: guards, scoping, deletion protocol, resilience, tracing.
`scripts/smoke.py` is the live layer: real Gemini and BigQuery over scripted scenarios, with the
transcript saved to `docs/example_run.md`.

### 9.2 Verifying that a report answers the user's intent

1. **Intent extraction**: when a report is requested, a small model call turns the request into a
   checklist: period ("Q1 2026"), entities, required sections ("insights", "action items for Q2"),
   metrics implied ("revenue, growth").
2. **Coverage judge**: an LLM judge scores the report against the checklist (each item: present /
   partially / missing) and against the rubric (numbers traceable to tool results, actions concrete and
   tied to findings, period correct).
3. **Grounding check** (deterministic): every number in the report must appear in, or be derivable
   from (sums, percentages within tolerance), the tool results logged on the trace. Unmatched numbers
   are flagged as possible hallucinations.
4. Online, this runs on a sample of production reports and is shown next to user ratings. A
   disagreement between the judge and users is itself a signal to recalibrate the judge (human
   spot checks on a weekly sample).

### 9.3 Evaluating UX

- **Behavioural (from traces)**: task success proxy (answered without error or refusal), *re-ask rate*
  (a semantically similar question within 2 turns suggests the first answer missed), follow-up depth (are
  executives *discussing* results?), abandonment after an error, time-to-first-token and p95 latency,
  confirmation completion vs cancellation for deletes (a high cancel rate means the preview surprised them),
  refusal rate and false-refusal reports.
- **Explicit**: 👍/👎 with optional reason, a periodic one-question CSAT, and report-level "was this useful
  for your meeting?".
- **Qualitative**: moderated sessions with 3–5 executives per release on realistic tasks, plus
  shadowing of analyst-assisted workflows before and after.
- **Experiments**: A/B tests on persona and prompt variants through the versioned persona store (§11), with
  success rate and rating as the primary metrics.

---

## 10. Requirement 7: Observability

### 10.1 Agent-level metrics

| Category | Metric | Alert example |
|---|---|---|
| Health | turn success rate, error rate (by error type), refusal rate | error rate > 5 % over 15 min |
| Latency | p50 / p95 turn latency, LLM latency per call, BigQuery latency, time-to-first-token | p95 > 30 s |
| LLM | calls per turn, tokens per turn (prompt / output), retries, **fallback rate**, 429 rate | fallback rate > 10 % |
| SQL | attempts per turn, failure rate by `error_code`, **self-correction success rate**, empty-result rate, bytes scanned per turn, cache hit rate | self-correction success < 60 % |
| Safety | guard blocks by verdict, PII redactions in output (should be ~0; any > 0 is investigated), out-of-scope refusals, delete confirmations / cancellations / expiries | any output PII redaction |
| Quality | 👍/👎 ratio, re-ask rate, judge score on sampled reports, golden retrieval hit rate (top score above threshold) | 👎 ratio up 2× week on week |
| Cost | $ per turn (tokens × price + bytes × price), per user, per day | daily spend > budget |

### 10.2 Trace-level debugging

Every turn emits **one structured record** (`observability/tracing.py::TurnTrace`) with:
`trace_id`, user, session, redacted user message, guard verdict and reason, golden trio IDs and scores,
context (persona version, effective preferences, scope), **the exact system instruction**, every model
call (model, attempt, status, error code, latency, tokens, rate-limit wait, which tools it asked for
with which args, and a preview of its text), every tool call (args including the model's SQL, status,
error code and message, rows, bytes, cache hit, attempts left), retry and fallback counters, self-correction
flag, PII redaction counts, the redacted final answer, outcome and total latency.

So "what went wrong in Harry's conversation yesterday?" becomes:
`/trace <id>` (prototype) or a BigQuery query over the log sink. The record shows the message
correspondence in order: user → model asked for `run_sql(sql₁)` → `BQ_INVALID_SQL: Unrecognized name`
→ model asked for `run_sql(sql₂)` → 0 rows → … → final answer. Each step has the reason the system
reacted as it did. Session ID ties traces into conversations. In production, `trace_id` is also the
OpenTelemetry trace ID, so Cloud Trace shows the span waterfall (LLM vs BigQuery time) for the same turn.

**Privacy of traces**: user and model text is passed through the PII redactor before it's written. Traces
have a retention policy (e.g. 90 days raw, aggregates kept), and access to the traces dataset is restricted to the platform team.

**In the prototype**: `var/traces.jsonl`. Every CLI turn prints a footer
(`trace 3f2a… · answered · 2 SQL (1 self-corrected) · 6.1s · 9,812 tokens`). `/trace [id|last] [--full]`
shows the record and `/metrics [--all]` computes the aggregate metrics above (success, refusal and error rates,
p50/p95 latency, tokens, retries, fallback turns, SQL attempts, self-correction success rate, bytes, PII
redactions, top tool errors). Tests: `tests/test_observability.py`.

---

## 11. Requirement 8: Agility (persona management without redeploy)

**Separation of concerns.** The system instruction has layers with different owners
(`prompts.py`):

| Layer | Owner | Editable at runtime? | Can weaken safety? |
|---|---|---|---|
| Policy (hard rules, tool protocol, SQL rules) | Engineering | No: code, reviewed, evaluated | n/a |
| **Persona** (tone, report style, structure) | CEO office / comms | **Yes** | **No**: the policy is placed first and stated as non-overridable, and safety is enforced by tools and guards, not prompt wording |
| User context (scope, preferences) | System | Per user | No |
| Golden examples | Analytics team | Yes (bucket) | No |

**Production design.** Personas are versioned documents in Firestore (`personas/{id}/versions/{n}`) with
`status: draft | active`, author and change note. A small admin page behind IAP (or a Google Doc synced by
a Cloud Function) gives non-developers a form: edit text → **preview** (run 5 canned questions against
the draft and show answers side by side with the active version) → **publish**. Publishing flips
`active_version`. The agent service caches the active persona for at most 60 s (or is pushed via
Firestore listeners), so the change applies to the next turns with **no deployment**. Every turn records
`persona_version` in its trace, so quality metrics can be compared across versions, and rollback is one
click. Optional: a scheduled weekly change ("This week: more concise, focus on margin"), and an
A/B split by version.

Guardrails on editing: a length limit, a lint that rejects instructions touching data access ("show
emails", "ignore rules") before publishing, and the persona eval family (§9) runs on the draft during preview.

**In the prototype**: `config/persona.md` (Markdown with a `version` front-matter field) is **re-read
whenever its modification time changes**, so editing the file changes the tone of the very next answer,
with no restart. `/whoami` shows the active version, and every trace records it. Tests:
`test_persona_file_is_reread_without_restart`, `test_persona_cannot_override_policy`.

---

## 12. Extensibility: adding capabilities

A capability is a **tool**: a Python function with a typed signature and a docstring (ADK derives the
function-calling schema from them). It's registered in `Toolbox.tools()` and gets tracing and budgets for free
from the `@_traced` decorator. Side-effecting tools reuse the **plan → preview → confirm** protocol.

| Capability | Tool | Notes |
|---|---|---|
| **Charts** | `create_chart(spec)`: the model passes a Vega-Lite spec referencing the *last query result ID* (not raw numbers, so charts can't contain invented data); rendered server-side (Altair/vl-convert) to PNG in GCS; a signed URL is returned | Read-only, no confirmation. Preference `visuals=charts` already exists. |
| **Email reports** | `request_send_report(report_id, recipients)` → plan (render preview, validate recipients against the company directory / allow-list) → user confirms → Gmail API / SendGrid sends from a service account, DLP-scanned, audit-logged | Outbound side effect, so same two-phase protocol and audit as deletes. |
| **Web search for trends** | `web_search(query)` via Vertex AI **Grounding with Google Search** (or a search API) | Results are *untrusted input*: returned as quoted data with source URLs, never as instructions; outbound queries are checked so internal data/PII is never sent out; cited separately from warehouse facts in the answer. |
| **Scheduled reports** | `schedule_report(question, cron)` → Cloud Scheduler → agent runs headless as that user | Confirmation required; runs with the owner's scope. |
| **New data sources** | Add tables to `catalog.TABLES` + governed views | Guard, scope and PII registry pick them up automatically. |

When the tool count grows beyond about 10–15, the agent can be split with ADK sub-agents (e.g. an *analyst*
agent with the data tools and a *publisher* agent with chart and email tools) under a router, keeping
each prompt small. MCP servers can be added as ADK toolsets for third-party systems.

---

## 13. Cost and rate limits

- **LLM**: Flash-class pricing. A typical analytical turn is 2–5 model calls and 8–20k prompt tokens (the
  instruction with catalogue and trios is about 4–5k tokens; history grows with the conversation). Production
  uses ADK context caching for the static policy and catalogue prefix and summarises long histories.
  The prototype's **sliding-window limiter** (`OPSFLEET_LLM_RPM`, default 8) keeps a single user within
  the free-tier requests-per-minute. Waits are recorded on the trace (`rate_limit_wait_ms`).
- **BigQuery**: on-demand pricing; thelook tables are small (tens of MB). Even so, every query is
  dry-run first, capped with `maximum_bytes_billed` (1 GB default), limited to 200 rows, cached, and labelled
  `app=opsfleet-agent` for billing breakdown. Governed CTEs project only the needed columns, so
  scanned bytes stay low (BigQuery is columnar).
- **Failures don't multiply cost**: retry budgets (3 SQL failures, 10 tools, ~14 LLM calls per turn), no retry
  on non-retryable errors, and a circuit breaker on BigQuery.

## 14. Threat model (summary)

| Threat | Mitigation |
|---|---|
| Prompt injection in the user message | Input guard, policy-first prompt, tool-level enforcement |
| Indirect injection (text inside data, saved reports, web results) | Data is returned as tool *results*; tools enforce scope and read-only access regardless; deletion can't be triggered by the model |
| SQL injection / writes | AST validation, SELECT-only, single statement, read-only IAM |
| Data exfiltration across users | Scope from identity, governed views, report ownership checks |
| PII leakage | Governed views + row filter + output filter + DLP; traces redacted |
| Cost abuse (denial of wallet) | Byte caps, dry runs, budgets, rate limits, per-user quotas |
| Destructive mistakes | Two-phase delete, TTL, snapshot, soft delete, audit |

## 15. Known limitations of the prototype

- The input guard is regex-based: precise on the obvious cases, with a deliberate bias towards **not**
  blocking legitimate analysis. Subtle off-topic or social-engineering prompts fall through to the policy prompt
  and the model, and production adds Model Armor. The *data* boundary does not depend on it.
- Governed views are implemented as per-query CTEs. That is functionally equivalent for this agent, but only
  BigQuery-native views and row policies protect other clients.
- Sessions are in memory (a new CLI process means a new conversation); reports, preferences, audit and traces persist.
- Single-process, single-user CLI. No streaming of tokens.
- BM25 retrieval needs vocabulary overlap; embeddings handle paraphrases far better.
