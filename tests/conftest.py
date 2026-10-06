"""Shared fixtures: everything runs offline (no Gemini, no BigQuery)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Callable

import pytest
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from pydantic import ConfigDict

from opsfleet_agent.agent import AgentService
from opsfleet_agent.bq import BigQueryError, QueryResult
from opsfleet_agent.config import REPO_ROOT, load_settings
from opsfleet_agent.llm import ResilientGemini


async def _no_sleep(_s: float) -> None:
    return None


def call(name: str, **args: Any) -> types.Part:
    return types.Part(function_call=types.FunctionCall(name=name, args=args))


def text(t: str) -> types.Part:
    return types.Part(text=t)


Step = types.Part | Exception | Callable[[LlmRequest], types.Part]


class ScriptedLlm(BaseLlm):
    """Plays back a script of model outputs; records every request it receives."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    model: str = "fake-model"
    script: list = []
    requests: list = []

    async def generate_content_async(self, llm_request: LlmRequest, stream: bool = False) -> AsyncGenerator[LlmResponse, None]:
        self.requests.append(llm_request)
        if not self.script:
            step: Any = text("(script exhausted)")
        else:
            step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        if callable(step):
            step = step(llm_request)
        yield LlmResponse(
            content=types.Content(role="model", parts=[step]),
            usage_metadata=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=100, candidates_token_count=20, total_token_count=120
            ),
        )

    def last_tool_response(self) -> dict:
        for content in reversed(self.requests[-1].contents):
            for part in content.parts or []:
                if part.function_response:
                    return dict(part.function_response.response)
        return {}

    def system_instruction(self) -> str:
        return str(self.requests[-1].config.system_instruction)


@dataclass
class FakeBigQuery:
    """Stands in for BigQueryRunner: returns canned rows, records governed SQL."""

    responses: list = field(default_factory=list)  # QueryResult | BigQueryError | list[dict]
    executed: list[str] = field(default_factory=list)
    default_rows: list[dict] = field(default_factory=lambda: [{"metric": 1}])

    def run(self, sql: str) -> QueryResult:
        self.executed.append(sql)
        item = self.responses.pop(0) if self.responses else self.default_rows
        if isinstance(item, Exception):
            raise item
        if isinstance(item, QueryResult):
            return item
        return QueryResult(rows=item, total_rows=len(item), bytes_estimated=12_345)

    def table_row_counts(self) -> dict[str, int]:
        return {"orders": 125_000, "order_items": 180_000, "products": 29_000, "users": 100_000}


@pytest.fixture
def settings(tmp_path: Path):
    return load_settings(home=tmp_path / "var", llm_rpm=1000)


@pytest.fixture
def make_service(settings):
    def _make(user: str = "alice", script: list | None = None, bq: FakeBigQuery | None = None,
              fallback_script: list | None = None, **kw) -> tuple[AgentService, ScriptedLlm, FakeBigQuery]:
        llm = ScriptedLlm(model="primary", script=list(script or []), requests=[])
        fallback = ScriptedLlm(model="fallback", script=list(fallback_script or []), requests=[])
        model = ResilientGemini(
            model="primary", fallback_model="fallback", max_retries=2,
            delegate_factory=lambda name: llm if name == "primary" else fallback,
            sleep=_no_sleep,
        )
        fake_bq = bq or FakeBigQuery()
        svc = AgentService(settings, user, model=model, bq=fake_bq, **kw)
        svc.fallback_llm = fallback  # handy for assertions
        return svc, llm, fake_bq

    return _make


def run(coro):
    return asyncio.run(coro)


__all__ = ["BigQueryError", "REPO_ROOT", "call", "text", "run"]
