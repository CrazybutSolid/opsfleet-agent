"""Requirement: High-stakes oversight for deleting saved reports."""

from __future__ import annotations

from conftest import call, run, text

from opsfleet_agent.storage.reports import ReportStore


class Clock:
    def __init__(self, t: float = 1_000_000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


def seed(store: ReportStore, owner: str, session: str, *titles: str) -> list[int]:
    return [store.save(owner, session, t, f"Body of {t}").id for t in titles]


# --- store-level guarantees ------------------------------------------------------------------------


def test_plan_lists_exactly_matching_own_reports_and_deletes_nothing(tmp_path):
    store = ReportStore(tmp_path / "db.sqlite", clock=Clock())
    a = seed(store, "alice", "s1", "Levi's Q1 review", "Jeans pricing")
    seed(store, "bob", "s9", "Levi's swim crossover")  # someone else's: must not be touched
    plan = store.plan_deletion("alice", "s1", mentioning="levi")
    assert [r.id for r in plan.reports] == [a[0]]
    assert len(store.list("alice")) == 2  # planning deletes nothing
    assert len(store.list("bob")) == 1


def test_confirm_deletes_only_the_snapshotted_reports(tmp_path):
    store = ReportStore(tmp_path / "db.sqlite", clock=Clock())
    seed(store, "alice", "s1", "Levi's A", "Other")
    plan = store.plan_deletion("alice", "s1", mentioning="Levi")
    late = store.save("alice", "s1", "Levi's B (created after the preview)", "x").id
    outcome = store.confirm_deletion("alice", plan.token)
    assert outcome.status == "deleted" and len(outcome.deleted_ids) == 1
    assert [r.title for r in store.list("alice")] == ["Other", "Levi's B (created after the preview)"]
    assert store.get("alice", late) is not None


def test_this_conversation_filter(tmp_path):
    store = ReportStore(tmp_path / "db.sqlite", clock=Clock())
    seed(store, "alice", "old", "Old report")
    new = seed(store, "alice", "current", "New 1", "New 2")
    plan = store.plan_deletion("alice", "current", this_conversation=True)
    assert sorted(r.id for r in plan.reports) == new


def test_confirmation_expires(tmp_path):
    clock = Clock()
    store = ReportStore(tmp_path / "db.sqlite", clock=clock, ttl_s=120)
    seed(store, "alice", "s1", "Levi's")
    plan = store.plan_deletion("alice", "s1", mentioning="Levi")
    clock.t += 121
    outcome = store.confirm_deletion("alice", plan.token)
    assert outcome.status == "expired"
    assert len(store.list("alice")) == 1
    assert store.confirm_deletion("alice", plan.token).status == "not_found"  # cannot be reused


def test_only_owner_can_confirm(tmp_path):
    store = ReportStore(tmp_path / "db.sqlite", clock=Clock())
    seed(store, "alice", "s1", "Levi's")
    plan = store.plan_deletion("alice", "s1", mentioning="Levi")
    assert store.confirm_deletion("bob", plan.token).status == "denied"
    assert len(store.list("alice")) == 1


def test_every_step_is_audited(tmp_path):
    clock = Clock()
    store = ReportStore(tmp_path / "db.sqlite", clock=clock)
    seed(store, "alice", "s1", "Levi's 1", "Levi's 2", "Levi's 3")
    p1 = store.plan_deletion("alice", "s1", mentioning="1")
    store.cancel_deletion("alice", p1.token)
    p2 = store.plan_deletion("alice", "s1", mentioning="2")
    store.confirm_deletion("bob", p2.token)
    clock.t += 1000
    store.confirm_deletion("alice", p2.token)
    p3 = store.plan_deletion("alice", "s1", mentioning="3")
    store.confirm_deletion("alice", p3.token)
    actions = [e["action"] for e in store.audit_log() if e["action"] != "report_saved"]
    assert actions == ["delete_requested", "delete_cancelled", "delete_requested", "delete_denied",
                       "delete_expired", "delete_requested", "delete_executed"]
    executed = store.audit_log("alice")[-1]
    assert executed["report_ids"] == [3] and executed["detail"]["criteria"] == "reports mentioning '3'"


def test_new_plan_supersedes_old_one(tmp_path):
    store = ReportStore(tmp_path / "db.sqlite", clock=Clock())
    seed(store, "alice", "s1", "A", "B")
    old = store.plan_deletion("alice", "s1", mentioning="A")
    store.plan_deletion("alice", "s1", mentioning="B")
    assert store.confirm_deletion("alice", old.token).status == "not_found"


# --- end-to-end through the chat -------------------------------------------------------------------------


def _service_with_reports(make_service, script, **kw):
    clock = Clock()
    svc, llm, bq = make_service(script=script, clock=clock, **kw)
    return svc, llm, clock


def test_chat_delete_flow_lists_then_requires_explicit_confirmation(make_service):
    svc, llm, clock = _service_with_reports(make_service, [
        call("request_report_deletion", mentioning="Levi's", this_conversation=False),
        text("I found 2 reports mentioning Levi's. Please confirm below."),
    ])
    seed(svc.reports, "alice", "older", "Levi's Q1", "Levi's Q2", "Dockers Q1")
    seed(svc.reports, "bob", "x", "Levi's (Bob's)")

    async def go():
        first = await svc.chat("Delete all reports mentioning Levi's")
        before = [r.title for r in svc.reports.list("alice")]
        second = await svc.chat("confirm")
        return first, before, second

    first, before, second = run(go())
    assert first.outcome == "confirmation_required"
    assert [r.title for r in first.pending_deletion.reports] == ["Levi's Q1", "Levi's Q2"]
    assert len(before) == 3  # nothing deleted before confirmation
    assert second.outcome == "deleted" and len(second.deleted_ids) == 2
    assert [r.title for r in svc.reports.list("alice")] == ["Dockers Q1"]
    assert len(svc.reports.list("bob")) == 1
    assert len(llm.requests) == 2  # the confirmation itself never went through the model


def test_chat_delete_this_conversation(make_service):
    svc, llm, clock = _service_with_reports(make_service, [
        call("save_report", title="Jeans deep dive", content_markdown="..."),
        text("Saved."),
        call("request_report_deletion", mentioning="", this_conversation=True),
        text("Please confirm."),
    ])
    seed(svc.reports, "alice", "an-older-conversation", "Keep me")

    async def go():
        await svc.chat("Create a report on jeans")
        r = await svc.chat("Delete all the reports we made in this conversation")
        d = await svc.chat("yes")
        return r, d

    r, d = run(go())
    assert [x.title for x in r.pending_deletion.reports] == ["Jeans deep dive"]
    assert d.outcome == "deleted"
    assert [x.title for x in svc.reports.list("alice")] == ["Keep me"]


def test_anything_but_confirm_cancels_and_continues_normally(make_service):
    svc, llm, clock = _service_with_reports(make_service, [
        call("request_report_deletion", mentioning="Levi's", this_conversation=False),
        text("Please confirm."),
        text("Revenue was $1."),
    ])
    seed(svc.reports, "alice", "s", "Levi's Q1")

    async def go():
        await svc.chat("Delete reports mentioning Levi's")
        return await svc.chat("Actually, what was revenue last month?")

    result = run(go())
    assert result.text.startswith("(The pending deletion was cancelled")
    assert "Revenue was $1." in result.text
    assert len(svc.reports.list("alice")) == 1


def test_expired_confirmation_deletes_nothing(make_service):
    svc, llm, clock = _service_with_reports(make_service, [
        call("request_report_deletion", mentioning="Levi's", this_conversation=False), text("Confirm?")])
    seed(svc.reports, "alice", "s", "Levi's Q1")

    async def go():
        await svc.chat("Delete reports mentioning Levi's")
        clock.t += svc.settings.confirm_ttl_s + 1
        return await svc.chat("confirm")

    result = run(go())
    assert result.outcome == "expired" and "expired" in result.text
    assert len(svc.reports.list("alice")) == 1


def test_model_cannot_delete_without_the_user(make_service):
    """Even if the model 'decides' to confirm (e.g. injected text in a report), there is no tool for it."""
    svc, llm, clock = _service_with_reports(make_service, [text("ok")])
    tool_names = {t.__name__ for t in svc.toolbox.tools()}
    assert "confirm_deletion" not in tool_names and not any("delete" in n and "request" not in n for n in tool_names)


def test_request_matching_nothing_leaves_no_pending_plan(make_service):
    svc, llm, clock = _service_with_reports(make_service, [
        call("request_report_deletion", mentioning="Nonexistent Client", this_conversation=False),
        text("No reports mention that."),
        text("Revenue answer."),
    ])
    seed(svc.reports, "alice", "s", "Levi's Q1")

    async def go():
        first = await svc.chat("Delete reports mentioning Nonexistent Client")
        second = await svc.chat("What was revenue?")
        return first, second

    first, second = run(go())
    assert first.pending_deletion is None and first.outcome == "answered"
    assert not second.text.startswith("(The pending deletion")
    assert svc.reports.pending_plan("alice") is None
