"""Requirement: resilience & graceful error handling, without inflating costs."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from conftest import FakeBigQuery, call, run, text
from google.api_core import exceptions as gexc
from google.genai import errors as genai_errors

from opsfleet_agent.bq import BigQueryError, BigQueryRunner, CircuitBreaker, QueryResult
from opsfleet_agent.cli import ChatCLI
from opsfleet_agent.llm import RateLimiter


def api_error(code: int) -> genai_errors.APIError:
    return genai_errors.APIError(code, {"error": {"code": code, "message": f"HTTP {code}", "status": "X"}})


BAD_SQL = "SELECT revenue FROM order_items"


# --- self-correction ---------------------------------------------------------------------------------


def test_sql_error_is_fed_back_and_self_corrected(make_service, settings):
    svc, llm, bq = make_service(script=[
        call("run_sql", sql="SELEC broken", purpose="try 1"),
        call("run_sql", sql=BAD_SQL, purpose="try 2"),
        call("run_sql", sql="SELECT SUM(sale_price) AS revenue FROM order_items", purpose="try 3"),
        text("Revenue is $10."),
    ])
    bq.responses = [BigQueryError("invalid_sql", "Unrecognized name: revenue at [1:8]"), [{"revenue": 10.0}]]
    result = run(svc.chat("What is our revenue?"))
    assert result.outcome == "answered" and result.text == "Revenue is $10."
    responses = [c.parts[0].function_response.response for r in llm.requests[1:] for c in r.contents[-1:]]
    assert responses[0]["error_code"] == "PARSE_ERROR" and responses[0]["attempts_left"] == 2
    assert responses[1]["error_code"] == "BQ_INVALID_SQL" and "Unrecognized name" in responses[1]["message"]
    t = svc.last_trace
    assert (t.sql_attempts, t.sql_failures, t.self_corrected) == (3, 2, True)


def test_empty_result_triggers_a_retry(make_service):
    svc, llm, bq = make_service(script=[
        call("run_sql", sql="SELECT COUNT(*) AS n FROM orders WHERE status = 'complete'", purpose="lowercase status"),
        call("run_sql", sql="SELECT COUNT(*) AS n FROM orders WHERE status = 'Complete'", purpose="fixed"),
        text("31,019 complete orders."),
    ])
    bq.responses = [[], [{"n": 31019}]]
    result = run(svc.chat("How many complete orders?"))
    first_feedback = llm.requests[1].contents[-1].parts[0].function_response.response
    assert first_feedback["error_code"] == "EMPTY_RESULT" and "capitalised" in first_feedback["message"]
    assert result.text == "31,019 complete orders."


def test_self_correction_is_capped(make_service, settings):
    script = [call("run_sql", sql=f"SELECT nope{i} FROM orders", purpose="x") for i in range(8)] + [text("I couldn't get it.")]
    svc, llm, bq = make_service(script=script)
    bq.responses = [BigQueryError("invalid_sql", "bad")] * 8
    result = run(svc.chat("Something hard"))
    assert len(bq.executed) == settings.max_sql_failures  # BigQuery hit at most N times
    assert llm.requests[-1].contents[-1].parts[0].function_response.response["status"] in ("gave_up", "blocked")
    assert result.outcome == "answered"  # graceful: the model explains instead of looping


def test_runaway_tool_loop_is_stopped(make_service, settings):
    script = [call("describe_data") for _ in range(30)]
    svc, llm, bq = make_service(script=script)
    result = run(svc.chat("loop forever"))
    assert result.outcome == "error" and "step budget" in result.text
    assert len(llm.requests) <= settings.max_tool_calls + 4


# --- cost controls -------------------------------------------------------------------------------------


class FakeJob:
    def __init__(self, bytes_processed=0, rows=None, error=None):
        self.total_bytes_processed = bytes_processed
        self.total_bytes_billed = bytes_processed
        self.cache_hit = False
        self.job_id = "job-1"
        self._rows = rows or []
        self._error = error

    def result(self, timeout=None, max_results=None):
        if self._error:
            raise self._error
        return _RowIterator(SimpleNamespace(items=r.items) for r in self._rows)


class _RowIterator(list):
    """Mimics google.cloud.bigquery.table.RowIterator: iterable rows + total_rows."""

    @property
    def total_rows(self):
        return len(self)


class FakeClient:
    def __init__(self, dry_bytes=1_000, rows=None, dry_error=None, run_error=None):
        self.configs = []
        self.dry_bytes, self.rows, self.dry_error, self.run_error = dry_bytes, rows or [{"n": 1}], dry_error, run_error

    def query(self, sql, job_config=None):
        self.configs.append(job_config)
        if job_config.dry_run:
            if self.dry_error:
                raise self.dry_error
            return FakeJob(self.dry_bytes)
        return FakeJob(self.dry_bytes, self.rows, self.run_error)


def test_dry_run_blocks_expensive_queries_before_they_run():
    client = FakeClient(dry_bytes=5_000_000_000)
    runner = BigQueryRunner(max_bytes_billed=1_000_000_000, client_factory=lambda: client)
    with pytest.raises(BigQueryError) as e:
        runner.run("SELECT * FROM huge")
    assert e.value.kind == "too_expensive"
    assert len(client.configs) == 1 and client.configs[0].dry_run  # never executed


def test_execution_carries_maximum_bytes_billed_and_labels():
    client = FakeClient(rows=[{"n": 5}])
    runner = BigQueryRunner(max_bytes_billed=123_456_789, client_factory=lambda: client)
    res = runner.run("SELECT 5 AS n")
    assert res.rows == [{"n": 5}]
    real = client.configs[1]
    assert real.maximum_bytes_billed == 123_456_789 and real.labels["app"] == "opsfleet-agent"


def test_identical_queries_are_served_from_cache():
    client = FakeClient()
    runner = BigQueryRunner(client_factory=lambda: client)
    runner.run("SELECT 1")
    second = runner.run("SELECT 1")
    assert second.cache_hit and len(client.configs) == 2  # 1 dry run + 1 run, then cached


def test_error_classification():
    bad = BigQueryRunner(client_factory=lambda: FakeClient(dry_error=gexc.BadRequest("Syntax error")))
    with pytest.raises(BigQueryError) as e:
        bad.run("x")
    assert e.value.kind == "invalid_sql"
    down = BigQueryRunner(client_factory=lambda: FakeClient(dry_error=gexc.ServiceUnavailable("down")))
    with pytest.raises(BigQueryError) as e:
        down.run("x")
    assert e.value.kind == "unavailable" and "not reachable" in e.value.message


def test_missing_credentials_is_unavailable_not_a_crash():
    def boom():
        raise RuntimeError("DefaultCredentialsError")

    with pytest.raises(BigQueryError) as e:
        BigQueryRunner(client_factory=boom).run("SELECT 1")
    assert e.value.kind == "unavailable"


def test_circuit_breaker_stops_hammering_a_dead_warehouse():
    t = [0.0]
    client = FakeClient(dry_error=gexc.ServiceUnavailable("down"))
    runner = BigQueryRunner(client_factory=lambda: client, breaker=CircuitBreaker(threshold=2, cooldown_s=30, clock=lambda: t[0]))
    for i in range(5):
        with pytest.raises(BigQueryError):
            runner.run(f"SELECT {i}")
    assert len(client.configs) == 2  # after 2 failures the breaker is open: no more calls
    t[0] = 31
    client.dry_error = None
    assert runner.run("SELECT 9").rows == [{"n": 1}]  # half-open probe succeeds and closes it


def test_bigquery_down_gives_a_clear_message(make_service):
    svc, llm, bq = make_service(script=[
        call("run_sql", sql="SELECT COUNT(*) FROM orders", purpose="count"),
        call("run_sql", sql="SELECT COUNT(*) FROM orders", purpose="again"),
        text("The data warehouse is unreachable right now; please try again shortly."),
    ])
    bq.responses = [BigQueryError("unavailable", "The data warehouse (BigQuery) is not reachable right now")]
    result = run(svc.chat("How many orders?"))
    first = llm.requests[1].contents[-1].parts[0].function_response.response
    assert first["status"] == "unavailable" and "not reachable" in first["message"]
    assert len(bq.executed) == 1  # no retry storm
    assert "unreachable" in result.text


# --- LLM failures ---------------------------------------------------------------------------------------


def test_gemini_429_backs_off_then_falls_back(make_service):
    svc, llm, bq = make_service(
        script=[api_error(429), api_error(503)],
        fallback_script=[text("Answer from the fallback model.")],
    )
    result = run(svc.chat("What is revenue?"))
    assert result.text == "Answer from the fallback model."
    t = svc.last_trace
    assert t.fallback_used and t.llm_retries == 1  # backoff once, then the breaker trips
    assert [(c["model"], c["status"]) for c in t.model_calls] == [
        ("primary", "error"), ("primary", "error"), ("fallback", "ok")]


def test_failing_primary_is_skipped_on_later_calls(make_service):
    svc, llm, bq = make_service(
        script=[api_error(503), api_error(503)],
        fallback_script=[call("run_sql", sql="SELECT COUNT(*) FROM orders", purpose="n"), text("42."), text("Again.")],
    )

    async def go():
        await svc.chat("How many orders?")
        first = [c["model"] for c in svc.last_trace.model_calls]
        await svc.chat("And now?")
        return first, [(c["model"], c["status"]) for c in svc.last_trace.model_calls]

    first, second = run(go())
    # Within the first turn: primary fails twice, breaker opens, the 2nd model call skips it.
    assert first == ["primary", "primary", "fallback", "primary", "fallback"]
    assert second == [("primary", "skipped"), ("fallback", "ok")]
    assert len(llm.requests) == 2  # the primary was not called again during the cooldown


def test_transient_error_recovers_on_primary(make_service):
    svc, llm, bq = make_service(script=[api_error(500), text("Recovered.")])
    result = run(svc.chat("What is revenue?"))
    assert result.text == "Recovered." and not svc.last_trace.fallback_used


def test_all_models_down_gives_clear_message(make_service):
    svc, llm, bq = make_service(script=[api_error(429)] * 3, fallback_script=[api_error(503)] * 3)
    result = run(svc.chat("What is revenue?"))
    assert result.outcome == "error" and "temporarily unavailable" in result.text


def test_non_retryable_error_fails_fast(make_service):
    svc, llm, bq = make_service(script=[api_error(400)], fallback_script=[text("never")])
    result = run(svc.chat("What is revenue?"))
    assert result.outcome == "error" and len(svc.last_trace.model_calls) == 1


def test_retired_model_skips_straight_to_fallback(make_service):
    svc, llm, bq = make_service(script=[api_error(404)], fallback_script=[text("Fallback answered.")])
    result = run(svc.chat("What is revenue?"))
    assert result.text == "Fallback answered."
    assert [c["model"] for c in svc.last_trace.model_calls] == ["primary", "fallback"]
    assert svc.last_trace.llm_retries == 0


def test_rate_limiter_keeps_under_rpm():
    now = [0.0]
    slept = []

    async def fake_sleep(s):
        slept.append(s)
        now[0] += s

    limiter = RateLimiter(rpm=3, clock=lambda: now[0], sleep=fake_sleep)

    async def go():
        for _ in range(4):
            await limiter.acquire()

    asyncio.run(go())
    assert len(slept) == 1 and 59 < slept[0] <= 61


# --- never crash ------------------------------------------------------------------------------------------


def test_tool_exception_does_not_break_the_turn(make_service):
    class ExplodingBQ(FakeBigQuery):
        def run(self, sql):
            raise ZeroDivisionError("boom")

    svc, llm, bq = make_service(bq=ExplodingBQ(), script=[
        call("run_sql", sql="SELECT 1 FROM orders", purpose="x"), text("Sorry, a tool failed.")])
    result = run(svc.chat("count orders"))
    assert llm.last_tool_response()["error_code"] == "TOOL_EXCEPTION"
    assert result.outcome == "answered"


def test_service_never_raises(make_service, monkeypatch):
    svc, llm, bq = make_service(script=[text("x")])

    async def explode(*a, **k):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(svc, "_run_agent", explode)
    result = run(svc.chat("hello, what is revenue"))
    assert result.outcome == "error" and "trace" in result.text


def test_cli_survives_errors_and_keeps_chatting(make_service, monkeypatch):
    from rich.console import Console

    svc, llm, bq = make_service(script=[text("Answer two.")])
    console = Console(record=True, width=120)
    cli = ChatCLI(svc, console)
    inputs = iter(["first question", "/trace", "/metrics", "/nonsense", "second question", "/quit"])
    monkeypatch.setattr(console, "input", lambda *_a, **_k: next(inputs))
    calls = {"n": 0}
    original = svc.chat

    async def flaky(msg):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("kaboom")
        return await original(msg)

    monkeypatch.setattr(svc, "chat", flaky)
    asyncio.run(cli.run())
    out = console.export_text()
    assert "Unexpected error: RuntimeError: kaboom" in out
    assert "Answer two." in out and "Unknown command" in out and "Bye." in out
