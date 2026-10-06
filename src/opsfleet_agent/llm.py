"""Resilient Gemini model for ADK: rate limiting, backoff, fallback model, tracing.

``ResilientGemini`` is an ADK ``BaseLlm`` so the agent framework is unaware of
it. For each model request it:

1. waits on a sliding-window rate limiter (stay inside free-tier RPM);
2. calls the primary model; on 429 / 5xx / timeout it retries with exponential
   backoff + jitter (honouring the server's retry hint, capped);
3. after ``max_retries`` it switches to the fallback model (cheaper, separate
   quota) and repeats;
4. if every model fails it raises ``LlmUnavailable``, which the agent service
   turns into a clear message instead of a crash.

Non-retryable errors (400 bad request, 403 key problems) fail fast.
Every attempt is recorded on the current turn trace.
"""

from __future__ import annotations

import asyncio
import collections
import random
import re
import time
from typing import AsyncGenerator, Awaitable, Callable

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import errors as genai_errors
from pydantic import ConfigDict, Field, PrivateAttr

from .observability.tracing import current_trace
from .security.pii import redact_text

RETRYABLE = {408, 429, 500, 502, 503, 504}


class LlmUnavailable(Exception):
    pass


class RateLimiter:
    """Sliding-window limiter: at most ``rpm`` calls in any 60 s window."""

    def __init__(self, rpm: int, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep):
        self.rpm = max(1, rpm)
        self.clock = clock
        self.sleep = sleep
        self.calls: collections.deque[float] = collections.deque()

    async def acquire(self) -> float:
        waited = 0.0
        while True:
            now = self.clock()
            while self.calls and now - self.calls[0] >= 60:
                self.calls.popleft()
            if len(self.calls) < self.rpm:
                self.calls.append(now)
                return waited
            delay = 60 - (now - self.calls[0]) + 0.05
            waited += delay
            await self.sleep(delay)


def _retry_hint_s(err: Exception) -> float | None:
    m = re.search(r"retry in ([\d.]+)s|retryDelay['\"]?:\s*['\"]?([\d.]+)s", str(err))
    if m:
        return float(m.group(1) or m.group(2))
    return None


def _summarise_response(resp: LlmResponse) -> dict:
    parts = resp.content.parts if resp.content and resp.content.parts else []
    calls = [
        {"name": p.function_call.name, "args": redact_text(str(dict(p.function_call.args or {}))).text[:2000]}
        for p in parts if p.function_call
    ]
    text = "".join(p.text or "" for p in parts if p.text and not getattr(p, "thought", False))
    return {"function_calls": calls, "text_preview": redact_text(text[:300]).text}


class ResilientGemini(BaseLlm):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: str = "gemini-2.5-flash"
    fallback_model: str | None = "gemini-2.5-flash-lite"
    max_retries: int = 2
    base_delay_s: float = 2.0
    max_delay_s: float = 20.0
    call_timeout_s: float = 90.0
    rate_limiter: RateLimiter | None = Field(default=None, exclude=True)
    # Factory for the underlying model; tests inject fakes here.
    delegate_factory: Callable[[str], BaseLlm] | None = Field(default=None, exclude=True)
    sleep: Callable[[float], Awaitable[None]] = Field(default=asyncio.sleep, exclude=True)
    _delegates: dict = PrivateAttr(default_factory=dict)

    def _delegate(self, model: str) -> BaseLlm:
        if model not in self._delegates:
            if self.delegate_factory:
                self._delegates[model] = self.delegate_factory(model)
            else:
                from google.adk.models.google_llm import Gemini

                self._delegates[model] = Gemini(model=model)
        return self._delegates[model]

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        trace = current_trace()
        models = [self.model] + ([self.fallback_model] if self.fallback_model else [])
        if trace is not None and not trace.prompt and llm_request.config.system_instruction:
            si = str(llm_request.config.system_instruction)
            trace.prompt = {"system_instruction": si, "chars": len(si)}
        last_error: Exception | None = None
        for model_index, model in enumerate(models):
            for attempt in range(self.max_retries + 1):
                waited = await self.rate_limiter.acquire() if self.rate_limiter else 0.0
                llm_request.model = model
                t0 = time.monotonic()
                try:
                    responses = []

                    async def _collect():
                        async for r in self._delegate(model).generate_content_async(llm_request, stream=False):
                            responses.append(r)

                    await asyncio.wait_for(_collect(), timeout=self.call_timeout_s)
                except (genai_errors.APIError, asyncio.TimeoutError, ConnectionError, OSError) as e:
                    code = getattr(e, "code", None) or (408 if isinstance(e, asyncio.TimeoutError) else 503)
                    last_error = e
                    if trace:
                        trace.model_call(model=model, attempt=attempt + 1, status="error", error_code=code,
                                         error=str(e)[:300], latency_ms=int((time.monotonic() - t0) * 1000),
                                         rate_limit_wait_ms=int(waited * 1000))
                    if code not in RETRYABLE:
                        raise LlmUnavailable(f"Model {model} rejected the request ({code}).") from e
                    if attempt < self.max_retries:
                        if trace:
                            trace.llm_retries += 1
                        hint = _retry_hint_s(e)
                        delay = min(self.max_delay_s, hint if hint else self.base_delay_s * 2**attempt)
                        await self.sleep(delay + random.uniform(0, 0.5))
                    continue
                usage = next((r.usage_metadata for r in reversed(responses) if r.usage_metadata), None)
                if trace:
                    if model_index > 0:
                        trace.fallback_used = True
                    trace.model_call(
                        model=model, attempt=attempt + 1, status="ok",
                        latency_ms=int((time.monotonic() - t0) * 1000), rate_limit_wait_ms=int(waited * 1000),
                        prompt_tokens=getattr(usage, "prompt_token_count", None),
                        output_tokens=(getattr(usage, "candidates_token_count", None) or 0)
                        + (getattr(usage, "thoughts_token_count", None) or 0),
                        request_contents=len(llm_request.contents),
                        response=_summarise_response(responses[-1]) if responses else {},
                    )
                for r in responses:
                    yield r
                return
        raise LlmUnavailable(f"All models failed; last error: {last_error}")
