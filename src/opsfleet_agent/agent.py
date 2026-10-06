"""AgentService: one conversation turn, end to end.

    user message
      -> pending-deletion gate (deterministic confirm / cancel, no LLM)
      -> input guard (injection / PII request / off-topic -> refuse, no LLM)
      -> golden-trio retrieval (BM25) -> instruction (policy + persona + user + prefs + trios)
      -> ADK LlmAgent loop (ResilientGemini <-> tools: run_sql, describe_data, reports, prefs)
      -> output PII filter -> answer (+ deletion confirmation prompt if one was planned)
      -> JSON trace written

Nothing in here raises to the caller: every failure becomes a clear message
and an ``error`` trace, so the CLI cannot crash on a turn.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from dataclasses import dataclass, field

from google.adk.agents import LlmAgent
from google.adk.agents.invocation_context import LlmCallsLimitExceededError
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.agents.run_config import RunConfig
from google.adk.models.base_llm import BaseLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from .bq import BigQueryRunner
from .config import Settings
from .knowledge.golden import GoldenStore, format_for_prompt
from .llm import LlmUnavailable, RateLimiter, ResilientGemini
from .observability.tracing import Tracer, TurnTrace
from .prompts import PersonaFile, build_instruction
from .security.input_guard import REFUSALS, screen
from .security.pii import redact_text
from .security.scope import UserScope, get_user
from .storage.preferences import PreferenceStore
from .storage.reports import DeletionPlan, ReportStore
from .tools import Toolbox, TurnState, reset_turn, set_turn

log = logging.getLogger(__name__)

APP_NAME = "opsfleet_insights"
_CONFIRM = re.compile(r"^\s*(confirm|confirmed|yes|y|yes,? delete( them| it)?|delete them|proceed|go ahead)\s*[.!]*\s*$", re.I)


@dataclass
class TurnResult:
    text: str
    trace_id: str
    outcome: str
    pending_deletion: DeletionPlan | None = None
    deleted_ids: tuple[int, ...] = ()
    saved_report_ids: list[int] = field(default_factory=list)


class AgentService:
    def __init__(
        self,
        settings: Settings,
        user_id: str,
        model: BaseLlm | None = None,
        bq: BigQueryRunner | None = None,
        reports: ReportStore | None = None,
        clock=None,
    ):
        self.settings = settings
        self.user: UserScope = get_user(settings.users_path, user_id)
        self.tracer = Tracer(settings.traces_path)
        report_kwargs = {"ttl_s": settings.confirm_ttl_s}
        if clock:
            report_kwargs["clock"] = clock
        self.reports = reports or ReportStore(settings.db_path, **report_kwargs)
        self.prefs = PreferenceStore(settings.db_path)
        self.golden = GoldenStore(settings.golden_path, settings.home / "golden_candidates.jsonl")
        self.persona = PersonaFile(settings.persona_path)
        self.bq = bq or BigQueryRunner(
            project=settings.bq_project, max_bytes_billed=settings.max_bytes_billed, max_rows=settings.max_rows
        )
        self.toolbox = Toolbox(self.bq, self.reports, self.prefs, self.golden, settings.max_rows, settings.rows_to_llm)
        self.model = model or ResilientGemini(
            model=settings.model,
            fallback_model=settings.fallback_model,
            max_retries=settings.llm_max_retries,
            rate_limiter=RateLimiter(settings.llm_rpm),
        )
        self._golden_block = ""
        self.agent = LlmAgent(
            name="opsfleet_insights",
            model=self.model,
            instruction=self._instruction,
            tools=self.toolbox.tools(),
            generate_content_config=types.GenerateContentConfig(temperature=0.2),
        )
        self.sessions = InMemorySessionService()
        self.runner = Runner(app_name=APP_NAME, agent=self.agent, session_service=self.sessions)
        self.session_id = ""
        self.last_trace: TurnTrace | None = None
        self._last_turn: tuple[str, str | None, str] | None = None  # (question, sql, answer)

    # -- conversation lifecycle ----------------------------------------------------

    async def new_conversation(self) -> str:
        self.session_id = uuid.uuid4().hex[:8]
        await self.sessions.create_session(app_name=APP_NAME, user_id=self.user.user_id, session_id=self.session_id)
        return self.session_id

    def _instruction(self, ctx: ReadonlyContext) -> str:
        return build_instruction(
            persona=self.persona.get(),
            user=self.user,
            preferences=self.prefs.instruction(self.user.user_id),
            golden=self._golden_block,
        )

    # -- one turn ------------------------------------------------------------------

    async def chat(self, message: str) -> TurnResult:
        if not self.session_id:
            await self.new_conversation()
        trace, token = self.tracer.start(self.user.user_id, self.session_id, message)
        try:
            result = await self._chat(message, trace)
        except Exception as e:  # last-resort guard: never propagate to the UI
            log.exception("turn failed")
            trace.finish("error", error=f"{type(e).__name__}: {e}")
            result = TurnResult(
                f"Sorry, something went wrong on my side and I couldn't finish that. (trace {trace.trace_id})",
                trace.trace_id, "error",
            )
        finally:
            self.tracer.end(trace, token)
            self.last_trace = trace
        return result

    async def _chat(self, message: str, trace: TurnTrace) -> TurnResult:
        prefix = ""
        # 1. A pending destructive action is resolved by the user's own words, never by the model.
        plan = self.reports.pending_plan(self.user.user_id)
        if plan is not None:
            if _CONFIRM.match(message):
                outcome = self.reports.confirm_deletion(self.user.user_id, plan.token)
                trace.tool_call(name="confirm_deletion", args={"token": plan.token}, status=outcome.status,
                                count=len(outcome.deleted_ids))
                trace.finish("deleted" if outcome.status == "deleted" else outcome.status, outcome.message)
                return TurnResult(outcome.message, trace.trace_id, trace.outcome, deleted_ids=outcome.deleted_ids)
            self.reports.cancel_deletion(self.user.user_id, plan.token, reason="user did not confirm")
            trace.tool_call(name="cancel_deletion", args={"token": plan.token}, status="cancelled")
            prefix = "(The pending deletion was cancelled; nothing was deleted.)\n\n"
            if re.match(r"^\s*(no|n|cancel|stop|don'?t|nope|abort)\b", message, re.I):
                trace.finish("cancelled", prefix.strip())
                return TurnResult(prefix.strip(), trace.trace_id, "cancelled")

        # 2. Deterministic input guard.
        verdict = screen(message)
        trace.guard = {"verdict": verdict.verdict.value, "reason": verdict.reason}
        if not verdict.allowed:
            text = prefix + REFUSALS[verdict.verdict]
            trace.finish("refused", text)
            return TurnResult(text, trace.trace_id, "refused")

        # 3. Golden knowledge for this question.
        hits = self.golden.search(message, k=2)
        trace.golden = [{"id": t.id, "score": s, "question": t.question} for t, s in hits]
        self._golden_block = format_for_prompt(hits)
        self.persona.get()  # refresh from disk so the trace records the version actually used
        trace.context = {
            "persona_version": self.persona.version,
            "preferences": self.prefs.effective(self.user.user_id),
            "scope": self.user.describe(),
        }

        # 4. Agent loop.
        state = TurnState(self.user, self.session_id, self.settings.max_sql_failures, self.settings.max_tool_calls)
        turn_token = set_turn(state)
        try:
            text = await asyncio.wait_for(self._run_agent(message), timeout=240)
        except LlmUnavailable as e:
            trace.finish("error", error=f"LlmUnavailable: {e}")
            return TurnResult(
                prefix + "The AI service is temporarily unavailable (rate limit or outage), even after retrying "
                "and switching to the backup model. Please try again in a minute.",
                trace.trace_id, "error",
            )
        except LlmCallsLimitExceededError:
            trace.finish("error", error="step limit reached")
            return TurnResult(
                prefix + "I couldn't finish that analysis within my step budget. Try splitting it into smaller questions.",
                trace.trace_id, "error",
            )
        except asyncio.TimeoutError:
            trace.finish("error", error="turn timeout")
            return TurnResult(prefix + "That took too long and was stopped. Try a narrower question.", trace.trace_id, "error")
        finally:
            reset_turn(turn_token)

        # 5. Output filter (defence in depth).
        red = redact_text(text or "")
        trace.add_redactions(red.findings)
        answer = red.text.strip() or "I couldn't produce an answer for that. Could you rephrase?"
        # A saved report must be visible to the user, whatever the model chose to say about it.
        for rid in state.saved_report_ids:
            report = self.reports.get(self.user.user_id, rid)
            if report and report.body[:200] not in answer:
                answer += f"\n\n---\n**Saved report #{report.id}: {report.title}**\n\n{report.body}"
        plan = state.deletion_plan if state.deletion_plan and state.deletion_plan.reports else None
        outcome = "confirmation_required" if plan else "answered"
        trace.finish(outcome, answer)
        self._last_turn = (message, state.last_sql, answer)
        return TurnResult(prefix + answer, trace.trace_id, outcome, pending_deletion=plan,
                          saved_report_ids=state.saved_report_ids)

    def record_feedback(self, rating: str, comment: str = "") -> str:
        """User feedback on the last answer: input to the system-level learning loop.

        Every rating is logged against the trace (for eval-set curation and
        dashboards). A *good* rating on an answer backed by SQL is queued as a
        golden-trio candidate for analyst review; it never auto-enters the golden set.
        """
        if self.last_trace is None:
            return "Nothing to rate yet."
        record = {"trace_id": self.last_trace.trace_id, "user_id": self.user.user_id, "rating": rating,
                  "comment": redact_text(comment).text, "outcome": self.last_trace.outcome}
        path = self.settings.home / "feedback.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        if rating == "good" and self._last_turn and self._last_turn[1]:
            question, sql, answer = self._last_turn
            self.golden.add_candidate(question, sql, answer, self.user.user_id, self.last_trace.trace_id)
            return "Thanks! Logged, and queued as a golden-example candidate for analyst review."
        return "Thanks, feedback logged against trace " + self.last_trace.trace_id + "."

    async def _run_agent(self, message: str) -> str:
        final = ""
        content = types.Content(role="user", parts=[types.Part(text=message)])
        run_config = RunConfig(max_llm_calls=self.settings.max_tool_calls + 4)
        async for event in self.runner.run_async(
            user_id=self.user.user_id, session_id=self.session_id, new_message=content, run_config=run_config
        ):
            if event.content and event.content.parts and event.author != "user":
                text = "".join(p.text or "" for p in event.content.parts if p.text and not getattr(p, "thought", False))
                if text.strip():
                    final = text
        return final
