"""Agent tools: the only way the model touches data, reports or memory.

Every tool runs inside a ``TurnState`` (per-turn budgets and outputs) and
records itself on the turn trace. Adding a capability (charts, e-mail, web
search) means adding one method here plus, if it has side effects, routing it
through the same plan -> confirm pattern used for deletions.
"""

from __future__ import annotations

import contextvars
import functools
import time
from dataclasses import dataclass, field
from typing import Callable

from google.adk.tools import ToolContext

from .bq import BigQueryError, BigQueryRunner
from .catalog import PII_COLUMNS, TABLES
from .knowledge.golden import GoldenStore
from .observability.tracing import current_trace
from .security.pii import filter_rows, redact_text
from .security.scope import UserScope
from .security.sql_guard import SqlRejected, govern
from .storage.preferences import PreferenceError, PreferenceStore
from .storage.reports import DeletionPlan, ReportStore


@dataclass
class TurnState:
    user: UserScope
    session_id: str
    max_sql_failures: int = 3
    max_tool_calls: int = 10
    tool_calls: int = 0
    sql_failures: int = 0
    deletion_plan: DeletionPlan | None = None
    saved_report_ids: list[int] = field(default_factory=list)
    last_sql: str | None = None


_turn: contextvars.ContextVar[TurnState | None] = contextvars.ContextVar("turn_state", default=None)


def set_turn(state: TurnState) -> contextvars.Token:
    return _turn.set(state)


def reset_turn(token: contextvars.Token) -> None:
    _turn.reset(token)


def _state() -> TurnState:
    state = _turn.get()
    if state is None:
        raise RuntimeError("tool called outside of a turn")
    return state


def _traced(fn: Callable[..., dict]) -> Callable[..., dict]:
    """Record the call on the trace, enforce the per-turn tool budget, never raise."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs) -> dict:
        state = _state()
        trace = current_trace()
        logged_args = {
            k: redact_text(v).text if isinstance(v, str) else v for k, v in kwargs.items() if k != "tool_context"
        }
        state.tool_calls += 1
        t0 = time.monotonic()
        if state.tool_calls > state.max_tool_calls:
            result = {"status": "blocked", "message": "Tool budget for this turn is exhausted. Answer with what you have."}
        else:
            try:
                result = fn(*args, **kwargs)
            except Exception as e:  # a tool bug must not crash the turn
                result = {"status": "error", "error_code": "TOOL_EXCEPTION", "message": f"{type(e).__name__}: {e}"}
        if trace is not None:
            entry = {"name": fn.__name__, "args": logged_args, "latency_ms": int((time.monotonic() - t0) * 1000)}
            entry.update({k: v for k, v in result.items() if k in (
                "status", "error_code", "message", "row_count", "bytes_estimated", "governed_sql", "attempts_left",
                "report_id", "token", "count", "cache_hit")})
            trace.tool_call(**entry)
        return result

    return wrapper


class Toolbox:
    def __init__(self, bq: BigQueryRunner, reports: ReportStore, prefs: PreferenceStore,
                 golden: GoldenStore, max_rows: int = 200, rows_to_llm: int = 50):
        self.bq = bq
        self.reports = reports
        self.prefs = prefs
        self.golden = golden
        self.max_rows = max_rows
        self.rows_to_llm = rows_to_llm

    def tools(self) -> list[Callable[..., dict]]:
        return [self.describe_data, self.run_sql, self.save_report, self.list_reports,
                self.request_report_deletion, self.remember_preference]

    # ---------------------------------------------------------------------------------

    @_traced
    def describe_data(self, tool_context: ToolContext) -> dict:
        """Describe the available tables, columns, the user's data scope and example questions.

        Use for questions like "what data do you have?" or "what can I ask?".
        """
        state = _state()
        try:
            counts = self.bq.table_row_counts()
        except BigQueryError:
            counts = {}
        tables = {
            name: {
                "description": t.description,
                "rows_total_in_warehouse": counts.get(name),
                "columns": {c.name: f"{c.type} - {c.description}" for c in t.columns},
            }
            for name, t in TABLES.items()
        }
        return {
            "status": "ok",
            "dataset": "thelook_ecommerce (synthetic fashion e-commerce: 2019 to today)",
            "your_scope": state.user.describe(),
            "tables": tables,
            "withheld_personal_data": sorted(PII_COLUMNS),
            "example_questions": [t.question for t in self.golden.trios[:6]] if self.golden.trios else [],
        }

    @_traced
    def run_sql(self, sql: str, purpose: str, tool_context: ToolContext) -> dict:
        """Run one read-only BigQuery SELECT and return the result rows.

        Args:
            sql: BigQuery Standard SQL using only the tables orders, order_items, products, users.
            purpose: One short sentence: what this query is meant to find out.
        """
        state = _state()
        trace = current_trace()
        if state.sql_failures >= state.max_sql_failures:
            return {"status": "gave_up", "message": "Query retry budget exhausted for this question. "
                    "Explain to the user what you tried and suggest how to rephrase."}
        if trace:
            trace.sql_attempts += 1
        attempts_left = lambda: max(0, state.max_sql_failures - state.sql_failures)  # noqa: E731

        def failure(code: str, message: str, correctable: bool = True, **extra) -> dict:
            state.sql_failures += 1
            if trace:
                trace.sql_failures += 1
            status = "error" if correctable else ("out_of_scope" if code == "OUT_OF_SCOPE" else "rejected")
            if not correctable:
                state.sql_failures = state.max_sql_failures  # don't let the model burn tokens retrying
            return {"status": status, "error_code": code, "message": message, "attempts_left": attempts_left(), **extra}

        try:
            governed = govern(sql, state.user, max_rows=self.max_rows)
        except SqlRejected as e:
            return failure(e.code, e.message, e.correctable)
        try:
            result = self.bq.run(governed.sql)
        except BigQueryError as e:
            if e.kind == "unavailable":
                state.sql_failures = state.max_sql_failures
                if trace:
                    trace.sql_failures += 1
                return {"status": "unavailable", "error_code": "BQ_UNAVAILABLE", "message": e.message}
            return failure("BQ_" + e.kind.upper(), e.message[:500])
        rows, findings = filter_rows(result.rows)
        if trace and findings:
            trace.add_redactions(findings)
        if not rows:
            return failure(
                "EMPTY_RESULT",
                "The query returned 0 rows. Check filters (status values are capitalised, date ranges, "
                "product names/categories exist in the user's scope). Fix and retry, or tell the user there is no data.",
                governed_sql=governed.sql[-400:],
            )
        if trace and state.sql_failures:
            trace.self_corrected = True
        state.last_sql = sql
        shown = rows[: self.rows_to_llm]
        return {
            "status": "ok",
            "purpose": purpose,
            "row_count": result.total_rows,
            "rows_shown": len(shown),
            "truncated": result.total_rows > len(shown),
            "columns": list(shown[0].keys()),
            "rows": shown,
            "bytes_estimated": result.bytes_estimated,
            "cache_hit": result.cache_hit,
        }

    @_traced
    def save_report(self, title: str, content_markdown: str, tool_context: ToolContext) -> dict:
        """Save a finished report (markdown, including action items) to the user's report library.

        Args:
            title: Short report title, e.g. "Q1 2026 Category Performance".
            content_markdown: The full report in markdown.
        """
        state = _state()
        body = redact_text(content_markdown)
        trace = current_trace()
        if trace and body.findings:
            trace.add_redactions(body.findings)
        report = self.reports.save(state.user.user_id, state.session_id, redact_text(title).text, body.text)
        state.saved_report_ids.append(report.id)
        return {"status": "ok", "report_id": report.id, "message": f"Saved as report #{report.id}."}

    @_traced
    def list_reports(self, mentioning: str, tool_context: ToolContext) -> dict:
        """List the user's saved reports, optionally only those mentioning some text.

        Args:
            mentioning: Text to search in titles and bodies; empty string lists all.
        """
        state = _state()
        found = self.reports.list(state.user.user_id, mentioning=mentioning or None)
        return {
            "status": "ok",
            "count": len(found),
            "reports": [
                {"id": r.id, "title": r.title, "created": time.strftime("%Y-%m-%d %H:%M", time.localtime(r.created_at)),
                 "this_conversation": r.session_id == state.session_id}
                for r in found
            ],
        }

    @_traced
    def request_report_deletion(self, mentioning: str, this_conversation: bool, tool_context: ToolContext) -> dict:
        """Prepare deletion of the user's own saved reports. Nothing is deleted until the user confirms.

        Args:
            mentioning: Delete reports whose title or text mentions this (e.g. a client or brand name). Empty if not used.
            this_conversation: True to target reports created in this conversation.
        """
        state = _state()
        if not (mentioning or "").strip() and not this_conversation:
            return {"status": "error", "message": "Say which reports: some text they mention and/or this conversation."}
        plan = self.reports.plan_deletion(
            state.user.user_id, state.session_id, mentioning=(mentioning or "").strip() or None,
            this_conversation=bool(this_conversation),
        )
        state.deletion_plan = plan
        if not plan.reports:
            self.reports.cancel_deletion(state.user.user_id, plan.token, reason="nothing matched")
            return {"status": "ok", "count": 0, "token": plan.token,
                    "message": f"No reports match ({plan.criteria}). Nothing to delete."}
        return {
            "status": "ok",
            "count": len(plan.reports),
            "token": plan.token,
            "message": "Deletion prepared. The system is now showing the user the exact list and asking for "
                       "confirmation. Just tell the user briefly; do not list or confirm anything yourself.",
        }

    @_traced
    def remember_preference(self, key: str, value: str, tool_context: ToolContext) -> dict:
        """Remember how this user likes answers, for this and future conversations.

        Args:
            key: One of format, depth, visuals, focus.
            value: format: table|bullets|prose; depth: brief|standard|deep; visuals: text|charts;
                focus: revenue|margin|customers|operations.
        """
        state = _state()
        try:
            saved = self.prefs.set(state.user.user_id, key, value, source="explicit")
        except PreferenceError as e:
            return {"status": "error", "message": str(e)}
        return {"status": "ok", "message": f"Preference saved: {saved['key']} = {saved['value']}."}
