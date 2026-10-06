"""BigQuery access: dry-run cost gate, hard byte cap, error classification, breaker.

Every query is executed in two steps:

1. **Dry run** (free): validates the SQL and returns the bytes it would scan.
   Syntax/semantic errors surface here at zero cost and go back to the model
   for self-correction. Queries above ``max_bytes_billed`` are refused before
   they run.
2. **Execute** with ``maximum_bytes_billed`` set on the job as a second,
   server-enforced cap, plus a timeout and job labels for cost attribution.

Failures are classified so the agent can react correctly:
``invalid_sql`` (model can fix it), ``too_expensive`` (model must narrow it),
``unavailable`` (BigQuery/network/auth down: tell the user, do not retry in a
loop). A small circuit breaker stops hammering BigQuery while it is down.
"""

from __future__ import annotations

import datetime as dt
import decimal
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

log = logging.getLogger(__name__)


class BigQueryError(Exception):
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind  # invalid_sql | too_expensive | unavailable
        self.message = message


@dataclass
class QueryResult:
    rows: list[dict[str, Any]]
    total_rows: int
    bytes_estimated: int
    bytes_billed: int = 0
    cache_hit: bool = False
    job_id: str | None = None
    latency_ms: int = 0


def _jsonable(value: Any) -> Any:
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, bytes):
        return "<bytes>"
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    return value


@dataclass
class CircuitBreaker:
    threshold: int = 3
    cooldown_s: float = 30.0
    clock: Callable[[], float] = time.monotonic
    failures: int = 0
    opened_at: float | None = None

    def allow(self) -> bool:
        if self.opened_at is None:
            return True
        if self.clock() - self.opened_at >= self.cooldown_s:
            self.opened_at = None  # half-open: let one call probe
            self.failures = self.threshold - 1
            return True
        return False

    def record(self, ok: bool) -> None:
        if ok:
            self.failures, self.opened_at = 0, None
            return
        self.failures += 1
        if self.failures >= self.threshold:
            self.opened_at = self.clock()


UNAVAILABLE_MSG = (
    "The data warehouse (BigQuery) is not reachable right now, so I can't run queries. "
    "Please try again in a few minutes. Your conversation and saved reports are unaffected."
)


@dataclass
class BigQueryRunner:
    project: str | None = None
    max_bytes_billed: int = 1_000_000_000
    max_rows: int = 200
    timeout_s: float = 60.0
    client_factory: Callable[[], Any] | None = None
    breaker: CircuitBreaker = field(default_factory=CircuitBreaker)
    cache_ttl_s: float = 600.0
    labels: dict[str, str] = field(default_factory=lambda: {"app": "opsfleet-agent"})
    _client: Any = None
    _cache: dict[str, tuple[float, QueryResult]] = field(default_factory=dict)

    def _get_client(self):
        if self._client is None:
            try:
                if self.client_factory:
                    self._client = self.client_factory()
                else:
                    from google.cloud import bigquery

                    self._client = bigquery.Client(project=self.project)
            except Exception as e:  # missing credentials, bad project, no network
                log.warning("BigQuery client init failed: %s", e)
                raise BigQueryError("unavailable", f"{UNAVAILABLE_MSG} (client init failed: {type(e).__name__})") from e
        return self._client

    @staticmethod
    def _classify(e: Exception) -> BigQueryError:
        from google.api_core import exceptions as gexc

        if isinstance(e, BigQueryError):
            return e
        if isinstance(e, gexc.BadRequest):
            msg = str(getattr(e, "message", e))
            if "bytes billed" in msg.lower():
                return BigQueryError("too_expensive", msg)
            return BigQueryError("invalid_sql", msg)
        if isinstance(e, gexc.NotFound):
            return BigQueryError("invalid_sql", str(getattr(e, "message", e)))
        # 403 (quota/permission), 5xx, timeouts, transport and auth errors.
        return BigQueryError("unavailable", f"{UNAVAILABLE_MSG} ({type(e).__name__})")

    def run(self, sql: str) -> QueryResult:
        cached = self._cache.get(sql)
        if cached and time.monotonic() - cached[0] < self.cache_ttl_s:
            res = cached[1]
            return QueryResult(res.rows, res.total_rows, res.bytes_estimated, 0, True, res.job_id, 0)
        if not self.breaker.allow():
            raise BigQueryError("unavailable", UNAVAILABLE_MSG + " (circuit open)")
        start = time.monotonic()
        try:
            result = self._run_uncached(sql)
        except Exception as e:
            err = self._classify(e)
            self.breaker.record(ok=err.kind != "unavailable")
            raise err from e
        self.breaker.record(ok=True)
        result.latency_ms = int((time.monotonic() - start) * 1000)
        self._cache[sql] = (time.monotonic(), result)
        return result

    def _run_uncached(self, sql: str) -> QueryResult:
        from google.cloud import bigquery

        client = self._get_client()
        dry = client.query(
            sql, job_config=bigquery.QueryJobConfig(dry_run=True, use_query_cache=False, labels=self.labels)
        )
        estimated = int(dry.total_bytes_processed or 0)
        if estimated > self.max_bytes_billed:
            raise BigQueryError(
                "too_expensive",
                f"Query would scan {estimated / 1e9:.2f} GB, above the {self.max_bytes_billed / 1e9:.2f} GB cap. "
                "Narrow it (fewer columns, a date filter, pre-aggregation).",
            )
        job = client.query(
            sql,
            job_config=bigquery.QueryJobConfig(maximum_bytes_billed=self.max_bytes_billed, labels=self.labels),
        )
        rows_iter = job.result(timeout=self.timeout_s, max_results=self.max_rows)
        rows = [_jsonable(dict(r.items())) for r in rows_iter]
        return QueryResult(
            rows=rows,
            total_rows=int(getattr(rows_iter, "total_rows", None) or len(rows)),
            bytes_estimated=estimated,
            bytes_billed=int(getattr(job, "total_bytes_billed", 0) or 0),
            cache_hit=bool(getattr(job, "cache_hit", False)),
            job_id=getattr(job, "job_id", None),
        )

    def table_row_counts(self) -> dict[str, int]:
        """Row counts for the schema tool (metadata call: free)."""
        from .catalog import DATASET, TABLES

        client = self._get_client()
        counts = {}
        for name in TABLES:
            try:
                counts[name] = int(client.get_table(f"{DATASET}.{name}").num_rows)
            except Exception as e:
                raise self._classify(e) from e
        return counts
