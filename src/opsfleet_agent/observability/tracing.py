"""One structured JSON trace per turn, plus aggregate metrics over them.

A trace answers "what exactly happened in this turn and why did it fail":
the (redacted) user message, guard verdict, golden trios retrieved, the system
prompt version, every model call (model, attempt, latency, tokens, error,
what it asked for), every tool call (args, governed SQL, rows, bytes, error
code), retries, fallbacks and the final (redacted) answer.

Prototype sink: append-only JSONL (``var/traces.jsonl``), inspected with the
``/trace`` and ``/metrics`` CLI commands. Production: the same record is emitted
as a structured Cloud Logging entry (``jsonPayload``) correlated with
OpenTelemetry spans in Cloud Trace (ADK emits spans natively), exported to
BigQuery via a log sink for dashboards (Looker Studio) and alerting
(Cloud Monitoring log-based metrics).
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import statistics
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..security.pii import redact_text

_current: contextvars.ContextVar["TurnTrace | None"] = contextvars.ContextVar("turn_trace", default=None)


def current_trace() -> "TurnTrace | None":
    return _current.get()


def short_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:10]


@dataclass
class TurnTrace:
    user_id: str
    session_id: str
    user_message: str
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    ts: float = field(default_factory=time.time)
    guard: dict[str, Any] = field(default_factory=dict)
    golden: list[dict[str, Any]] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)  # persona version, preferences, scope
    prompt: dict[str, Any] = field(default_factory=dict)
    model_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    llm_retries: int = 0
    fallback_used: bool = False
    sql_attempts: int = 0
    sql_failures: int = 0
    self_corrected: bool = False
    pii_redactions: dict[str, int] = field(default_factory=dict)
    final_answer: str = ""
    outcome: str = "pending"  # answered | refused | error | confirmation_required | deleted | ...
    error: str | None = None
    latency_ms: int = 0
    _t0: float = field(default_factory=time.monotonic, repr=False)

    def __post_init__(self) -> None:
        self.user_message = redact_text(self.user_message).text

    # -- recording helpers ---------------------------------------------------------

    def model_call(self, **data: Any) -> None:
        self.model_calls.append(data)

    def tool_call(self, **data: Any) -> dict[str, Any]:
        self.tool_calls.append(data)
        return data

    def add_redactions(self, findings: dict[str, int]) -> None:
        for k, n in findings.items():
            self.pii_redactions[k] = self.pii_redactions.get(k, 0) + n

    @property
    def tokens(self) -> dict[str, int]:
        p = sum(c.get("prompt_tokens") or 0 for c in self.model_calls)
        o = sum(c.get("output_tokens") or 0 for c in self.model_calls)
        return {"prompt": p, "output": o, "total": p + o}

    def finish(self, outcome: str, final_answer: str = "", error: str | None = None) -> None:
        self.outcome = outcome
        red = redact_text(final_answer)
        self.add_redactions(red.findings)
        self.final_answer = red.text
        self.error = error
        self.latency_ms = int((time.monotonic() - self._t0) * 1000)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("_t0", None)
        d["tokens"] = self.tokens
        return d


class Tracer:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def start(self, user_id: str, session_id: str, message: str) -> tuple[TurnTrace, contextvars.Token]:
        trace = TurnTrace(user_id=user_id, session_id=session_id, user_message=message)
        return trace, _current.set(trace)

    def end(self, trace: TurnTrace, token: contextvars.Token) -> None:
        _current.reset(token)
        try:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(trace.to_dict(), default=str, ensure_ascii=False) + "\n")
        except OSError:
            pass  # observability must never break the user's turn

    def load(self, user_id: str | None = None) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                t = json.loads(line)
            except json.JSONDecodeError:
                continue
            if user_id is None or t.get("user_id") == user_id:
                out.append(t)
        return out

    def get(self, trace_id: str, user_id: str | None = None) -> dict[str, Any] | None:
        traces = self.load(user_id)
        if trace_id in ("", "last"):
            return traces[-1] if traces else None
        for t in reversed(traces):
            if t["trace_id"].startswith(trace_id):
                return t
        return None


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    return float(statistics.quantiles(values, n=100, method="inclusive")[int(q) - 1])


def aggregate(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Agent-level health metrics over a set of traces."""
    n = len(traces)
    if n == 0:
        return {"turns": 0}
    outcomes: dict[str, int] = {}
    for t in traces:
        outcomes[t["outcome"]] = outcomes.get(t["outcome"], 0) + 1
    latencies = [t["latency_ms"] for t in traces]
    sql_turns = [t for t in traces if t.get("sql_attempts")]
    corrected = [t for t in sql_turns if t.get("sql_failures")]
    tool_errors: dict[str, int] = {}
    for t in traces:
        for c in t.get("tool_calls", []):
            if c.get("status") not in ("ok", "deleted", "cancelled", None):
                key = f"{c.get('name')}:{c.get('error_code') or c.get('status')}"
                tool_errors[key] = tool_errors.get(key, 0) + 1
    model_calls = [c for t in traces for c in t.get("model_calls", [])]
    return {
        "turns": n,
        "outcomes": outcomes,
        # success = the turn did what was asked (answer, confirmation prompt, deletion, cancellation)
        "success_rate": round(sum(v for k, v in outcomes.items() if k not in ("error", "refused")) / n, 3),
        "refusal_rate": round(outcomes.get("refused", 0) / n, 3),
        "error_rate": round(outcomes.get("error", 0) / n, 3),
        "latency_ms_p50": round(_pct(latencies, 50)),
        "latency_ms_p95": round(_pct(latencies, 95)),
        "tokens_total": sum(t.get("tokens", {}).get("total", 0) for t in traces),
        "tokens_per_turn_avg": round(statistics.mean(t.get("tokens", {}).get("total", 0) for t in traces), 1),
        "model_calls": len(model_calls),
        "model_errors": sum(1 for c in model_calls if c.get("status") != "ok"),
        "llm_retries": sum(t.get("llm_retries", 0) for t in traces),
        "fallback_turns": sum(1 for t in traces if t.get("fallback_used")),
        "sql_turns": len(sql_turns),
        "sql_attempts_avg": round(statistics.mean(t["sql_attempts"] for t in sql_turns), 2) if sql_turns else 0,
        "self_correction_turns": len(corrected),
        "self_correction_success_rate": (
            round(sum(1 for t in corrected if t.get("self_corrected")) / len(corrected), 3) if corrected else None
        ),
        "bytes_processed": sum(
            c.get("bytes_estimated", 0) or 0 for t in traces for c in t.get("tool_calls", []) if c.get("status") == "ok"
        ),
        "pii_redactions": sum(sum(t.get("pii_redactions", {}).values()) for t in traces),
        "top_tool_errors": dict(sorted(tool_errors.items(), key=lambda kv: -kv[1])[:5]),
    }
