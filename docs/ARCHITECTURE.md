# Architecture (HLD)

This document holds the diagrams. [TECHNICAL.md](TECHNICAL.md) explains each block, the reasoning
behind it and how each requirement is met. The prototype in `src/` implements the **Agent service**
box end to end on a laptop. The [last section](#prototype-vs-production-mapping) maps every prototype
component to its production counterpart.

## 1. Production high-level design (GCP)

```mermaid
flowchart LR
    subgraph Clients
        EXEC["Executives<br/>(web chat, Slack, CLI)"]
        ADMIN["Persona & policy console<br/>(CEO office, analysts)"]
    end

    subgraph Edge["Edge & identity"]
        LB["External HTTPS LB<br/>+ Cloud Armor"]
        IAP["Identity-Aware Proxy<br/>(Google Workspace SSO)"]
    end

    subgraph Serving["Serving: Cloud Run (us-central1, min 1 instance)"]
        API["chat-api<br/>(FastAPI, SSE streaming)"]
        AGENT["agent-service<br/>Google ADK runner:<br/>guards, prompt assembly,<br/>tool loop, output filter"]
        TOOLS["Tool layer (in-process)<br/>run_sql · describe_data · reports ·<br/>preferences · charts · email · web_search"]
    end

    subgraph AI["Vertex AI"]
        GEM["Gemini Flash (primary)<br/>Gemini Flash-Lite (fallback)<br/>Gemini Pro (offline judge)"]
        ARMOR["Model Armor<br/>(prompt-injection & jailbreak screening)"]
        EMB["gemini-embedding-001"]
        VS["Vector Search index<br/>(golden trios)"]
        EVAL["Gen AI Evaluation Service"]
        MEM["Agent Engine Memory Bank<br/>(long-tail user memories)"]
    end

    subgraph Data["Data plane"]
        BQSRC[("BigQuery<br/>thelook_ecommerce<br/>(raw, read-only)")]
        BQGOV[("BigQuery governed dataset<br/>authorized views + row access policies<br/>+ policy tags (PII columns)")]
        DLP["Sensitive Data Protection<br/>(DLP inspect templates)"]
    end

    subgraph State["Application state"]
        FS[("Firestore<br/>sessions · saved reports ·<br/>pending deletions · preferences ·<br/>persona versions")]
        GCS[("Cloud Storage<br/>golden bucket<br/>gs://…/trios/*.jsonl")]
        AUDIT[("BigQuery: audit & traces<br/>(append-only)")]
        SM["Secret Manager"]
    end

    subgraph Pipelines["Async pipelines"]
        PS["Pub/Sub topics:<br/>turn-events · feedback · trio-uploads"]
        INGEST["Cloud Run job:<br/>trio ingestion<br/>(validate SQL, scrub PII, embed)"]
        LEARN["Cloud Run job (nightly):<br/>learning loop<br/>(mine traces + feedback)"]
        CICD["Cloud Build: eval gate<br/>(offline eval set must pass)"]
    end

    subgraph Ops["Observability"]
        LOG["Cloud Logging<br/>(structured JSON turn traces)"]
        TRACE["Cloud Trace<br/>(OpenTelemetry spans from ADK)"]
        MON["Cloud Monitoring<br/>dashboards & alerts"]
        LOOK["Looker Studio<br/>agent quality dashboard"]
    end

    EXEC --> LB --> IAP --> API
    ADMIN --> IAP
    API -->|"gRPC/HTTP, user identity in signed header"| AGENT
    AGENT --> ARMOR
    AGENT <-->|"generateContent<br/>(retry + fallback)"| GEM
    AGENT --> TOOLS
    TOOLS -->|"SELECT via governed views,<br/>dry-run + maximum_bytes_billed"| BQGOV
    BQGOV -.->|"authorized view reads"| BQSRC
    TOOLS --> DLP
    TOOLS <--> FS
    AGENT -->|"query embedding"| EMB
    AGENT -->|"top-k trios (filtered)"| VS
    AGENT <--> MEM
    AGENT --> SM
    AGENT -->|"turn event"| PS
    API -->|"thumbs up/down"| PS
    GCS -->|"object finalize"| PS --> INGEST
    INGEST --> EMB
    INGEST --> VS
    INGEST -->|"dry-run SQL"| BQGOV
    PS --> AUDIT
    LEARN --> AUDIT
    LEARN -->|"candidate trios for review"| GCS
    LEARN -->|"preference updates"| FS
    CICD --> EVAL
    EVAL --> GEM
    AGENT --> LOG
    AGENT --> TRACE
    LOG -->|"log sink"| AUDIT
    AUDIT --> LOOK
    LOG --> MON
    ADMIN -->|"edit persona (versioned)"| FS
```

**Where data lives**

| Data | Store | Why |
|---|---|---|
| Transaction logs (source of truth) | BigQuery `bigquery-public-data.thelook_ecommerce` (read-only) | Given. Never queried directly by the agent's service account. |
| Governed, PII-free, per-user-filtered view of it | BigQuery dataset `opsfleet_governed`: authorized views + row access policies + policy tags | Enforced by the warehouse itself, so every client is protected, not just the agent. |
| Golden trios (raw) | Cloud Storage bucket `gs://opsfleet-golden/trios/` (JSONL, versioned objects) | Analysts drop files; versioning gives history and rollback. |
| Golden trios (searchable) | Vertex AI Vector Search (embeddings + metadata) | Low-latency semantic top-k with filters (approved, domain, freshness). |
| Sessions, saved reports, pending deletions, preferences, persona versions | Firestore (Native mode) | Per-user documents, transactions for the delete flow, TTL on pending deletions, no ops. |
| Audit log, turn traces, feedback | BigQuery dataset `opsfleet_ops` (append-only, via Pub/Sub + log sink) | Cheap long-term storage, SQL for debugging and dashboards. |
| Secrets (API keys for 3rd-party tools) | Secret Manager | Gemini on Vertex AI uses the service account (no key). |

## 2. Request flow (one chat turn)

```mermaid
sequenceDiagram
    autonumber
    actor U as Executive
    participant API as chat-api (Cloud Run)
    participant AG as agent-service (ADK)
    participant G as Input guards<br/>(regex + Model Armor)
    participant K as Golden retrieval<br/>(Vector Search)
    participant P as Prompt assembly<br/>(policy + persona + prefs + trios)
    participant LLM as Gemini (Vertex AI)
    participant T as Tools
    participant BQ as BigQuery (governed)
    participant FS as Firestore
    participant O as Logging / Trace

    U->>API: message (SSO identity)
    API->>AG: turn(user_id, session_id, text)
    AG->>FS: pending deletion for this user?
    alt pending deletion and message == "confirm"
        AG->>FS: transactional delete of snapshotted ids (owner check, TTL check) + audit
        AG-->>U: "Deleted N reports" (no LLM involved)
    else
        AG->>G: screen(text)
        alt injection / PII request / off-topic
            G-->>AG: refuse
            AG-->>U: polite refusal (no LLM, no data access)
        else allowed
            AG->>K: top-k similar trios (scope + approved filters)
            AG->>FS: user preferences, active persona version
            AG->>P: build system instruction
            loop ADK tool loop (bounded: tool budget, SQL retry budget, max LLM calls)
                AG->>LLM: generateContent(instruction, history, tools)
                LLM-->>AG: function_call run_sql(sql)
                AG->>T: run_sql
                T->>T: sqlglot guard: 1 SELECT, allowed tables, no PII cols, scope literals
                T->>BQ: dry run (bytes estimate, free)
                T->>BQ: execute with maximum_bytes_billed, labels, timeout
                BQ-->>T: rows (already PII-free and scoped)
                T->>T: output row filter (DLP / regex)
                T-->>AG: rows or classified error (+ attempts_left)
            end
            LLM-->>AG: final answer text
            AG->>AG: output PII filter (DLP)
            AG-->>U: answer (streamed) + deletion preview if one was planned
        end
    end
    AG->>O: structured trace (prompt, calls, SQL, rows, tokens, latency, errors)
    AG->>O: Pub/Sub turn-event (async)
```

## 3. Destructive operation (delete reports): two-phase commit with a human in the loop

```mermaid
sequenceDiagram
    actor U as User
    participant AG as Agent service
    participant LLM as Gemini
    participant R as Report store (Firestore)
    participant A as Audit log

    U->>AG: "Delete all reports mentioning Client X"
    AG->>LLM: turn
    LLM->>AG: request_report_deletion(mentioning="Client X")
    AG->>R: plan: snapshot matching ids WHERE owner = user (no delete)
    R->>A: delete_requested (ids, criteria)
    AG-->>U: table of exactly those reports + "type confirm within 120 s"
    alt user types "confirm" in time
        U->>AG: confirm
        Note over AG: matched deterministically, model not consulted
        AG->>R: txn: plan still pending? owner == user? not expired?<br/>delete snapshotted ids still owned by user
        R->>A: delete_executed (ids)
        AG-->>U: "Deleted 3 reports" (soft delete, restorable 30 days)
    else user writes anything else
        AG->>R: cancel plan
        R->>A: delete_cancelled
        AG-->>U: "(cancelled)" + continues with the new question
    else confirmation arrives after TTL
        R->>A: delete_expired
        AG-->>U: "expired, nothing deleted"
    end
```

## 4. Learning loops

```mermaid
flowchart TB
    subgraph User-level ["User level (per person, immediate)"]
        S1["User states a preference<br/>'tables please', 'shorter'"] --> T1["remember_preference tool"]
        S2["Implicit signals<br/>(re-asks for brevity, asks 'as a table' 3x)"] --> T2["nightly job infers preference<br/>source=inferred, never overrides explicit"]
        T1 --> PF[("Preference profile<br/>Firestore")]
        T2 --> PF
        PF --> PROMPT["Injected into the system instruction<br/>on every turn"]
    end

    subgraph System-level ["System level (whole agent, reviewed)"]
        TR[("Traces + feedback<br/>BigQuery")] --> MINE["Nightly mining job"]
        MINE --> C1["Thumbs-up answers backed by SQL<br/>→ candidate trios"]
        MINE --> C2["Failure clusters<br/>(SQL error types, empty results,<br/>refusals, low ratings)"]
        C1 --> REVIEW["Analyst review queue"]
        REVIEW -->|approve| GOLD[("Golden bucket → Vector Search")]
        C2 --> FIX["Prompt / catalogue / guard fixes<br/>+ new eval cases"]
        FIX --> EVALSET[("Offline eval set")]
        EVALSET --> GATE["CI eval gate before deploy"]
        GOLD --> RET["Retrieved at query time"]
    end
```

## Prototype vs production mapping

| Building block | Prototype (this repo) | Production |
|---|---|---|
| Client | Rich CLI (`opsfleet --user alice`) | Web chat / Slack app behind IAP |
| Identity & scope | `--user` flag + `config/users.toml` | Workspace SSO via IAP; entitlements table keyed by principal |
| Agent orchestration | Google ADK `LlmAgent` + `Runner`, in-process | Same code on Cloud Run (or Vertex AI Agent Engine) |
| LLM | Gemini via AI Studio key, `ResilientGemini` (retry → fallback) | Gemini on Vertex AI (service account, provisioned throughput for the primary) |
| Input guard | Regex classifier (`security/input_guard.py`) | Regex fast path + Model Armor |
| SQL governance | sqlglot guard + governed CTEs per query | Same guard + BigQuery authorized views, row access policies, policy tags |
| Output PII filter | Regex row/text filter (`security/pii.py`) | Sensitive Data Protection (DLP) inspect + regex fast path |
| Golden trios | `data/golden_trios.jsonl` + BM25 (`knowledge/golden.py`), hot reload | GCS bucket → ingestion job → Vector Search, hybrid retrieval |
| Reports, deletions, audit, prefs | SQLite (`var/opsfleet.db`) | Firestore + BigQuery audit table |
| Persona | `config/persona.md`, re-read on change | Firestore-versioned persona edited from an admin console |
| Traces & metrics | JSONL (`var/traces.jsonl`), `/trace`, `/metrics` | Cloud Logging + Cloud Trace + BigQuery sink + Looker Studio / Monitoring alerts |
| Feedback → learning | `/feedback` → `var/feedback.jsonl`, `var/golden_candidates.jsonl` | Pub/Sub → BigQuery → nightly mining job → review queue |
| Evaluation | pytest (offline, mocked) + live smoke script | Offline eval set in CI with Gen AI Evaluation Service + online metrics |
