"""Saved-reports library with a two-phase, audited delete.

Deletion is never executed by the model. The model can only *plan* a deletion
(``plan_deletion``), which snapshots exactly which report ids match, for which
owner, and when the plan expires. Execution (``confirm_deletion``) is triggered
by the user's own next message, matched deterministically outside the LLM, and
re-checks ownership and expiry inside one transaction. Every step (planned,
confirmed, cancelled, expired, denied) is written to an append-only audit log.

Production: reports live in Firestore (or Cloud SQL), audit events go to an
append-only BigQuery table via Pub/Sub, and deletes are soft (tombstone + 30-day
retention) so an operator can restore a mistaken bulk delete.
"""

from __future__ import annotations

import json
import secrets
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

SCHEMA = """
CREATE TABLE IF NOT EXISTS reports (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    owner       TEXT NOT NULL,
    session_id  TEXT NOT NULL,
    title       TEXT NOT NULL,
    body        TEXT NOT NULL,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS reports_owner ON reports(owner);
CREATE TABLE IF NOT EXISTS pending_deletions (
    token       TEXT PRIMARY KEY,
    owner       TEXT NOT NULL,
    session_id  TEXT NOT NULL,
    criteria    TEXT NOT NULL,
    report_ids  TEXT NOT NULL,
    created_at  REAL NOT NULL,
    expires_at  REAL NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending'
);
CREATE TABLE IF NOT EXISTS audit_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL NOT NULL,
    actor       TEXT NOT NULL,
    action      TEXT NOT NULL,
    token       TEXT,
    report_ids  TEXT NOT NULL,
    detail      TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class Report:
    id: int
    owner: str
    session_id: str
    title: str
    body: str
    created_at: float


@dataclass(frozen=True)
class DeletionPlan:
    token: str
    owner: str
    session_id: str
    criteria: str
    reports: tuple[Report, ...]
    expires_at: float


@dataclass(frozen=True)
class DeletionOutcome:
    status: str  # deleted | expired | not_found | denied | cancelled
    deleted_ids: tuple[int, ...] = ()
    message: str = ""


class ReportStore:
    def __init__(self, db_path: Path | str, clock: Callable[[], float] = time.time, ttl_s: int = 120):
        self.db_path = str(db_path)
        self.clock = clock
        self.ttl_s = ttl_s
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # -- reports -----------------------------------------------------------------

    def save(self, owner: str, session_id: str, title: str, body: str) -> Report:
        now = self.clock()
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO reports(owner, session_id, title, body, created_at) VALUES (?,?,?,?,?)",
                (owner, session_id, title, body, now),
            )
            self._audit(c, owner, "report_saved", None, [cur.lastrowid], {"title": title})
            return Report(cur.lastrowid, owner, session_id, title, body, now)

    def get(self, owner: str, report_id: int) -> Report | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM reports WHERE id = ? AND owner = ?", (report_id, owner)
            ).fetchone()
        return Report(**dict(row)) if row else None

    def list(self, owner: str, mentioning: str | None = None, session_id: str | None = None) -> list[Report]:
        sql, args = "SELECT * FROM reports WHERE owner = ?", [owner]
        if mentioning:
            sql += " AND (instr(lower(title), lower(?)) > 0 OR instr(lower(body), lower(?)) > 0)"
            args += [mentioning, mentioning]
        if session_id:
            sql += " AND session_id = ?"
            args.append(session_id)
        with self._conn() as c:
            rows = c.execute(sql + " ORDER BY created_at, id", args).fetchall()
        return [Report(**dict(r)) for r in rows]

    # -- two-phase delete --------------------------------------------------------

    def plan_deletion(
        self, owner: str, session_id: str, mentioning: str | None = None, this_conversation: bool = False
    ) -> DeletionPlan:
        if not mentioning and not this_conversation:
            raise ValueError("A deletion needs a filter: 'mentioning' text and/or this conversation.")
        targets = self.list(owner, mentioning=mentioning, session_id=session_id if this_conversation else None)
        parts = []
        if mentioning:
            parts.append(f"reports mentioning '{mentioning}'")
        if this_conversation:
            parts.append("reports created in this conversation")
        criteria = " and ".join(parts)
        token = secrets.token_hex(4)
        now = self.clock()
        ids = [r.id for r in targets]
        with self._conn() as c:
            # A new plan supersedes any older pending plan of this user.
            c.execute(
                "UPDATE pending_deletions SET status = 'superseded' WHERE owner = ? AND status = 'pending'",
                (owner,),
            )
            c.execute(
                "INSERT INTO pending_deletions VALUES (?,?,?,?,?,?,?, 'pending')",
                (token, owner, session_id, criteria, json.dumps(ids), now, now + self.ttl_s),
            )
            self._audit(c, owner, "delete_requested", token, ids, {"criteria": criteria})
        return DeletionPlan(token, owner, session_id, criteria, tuple(targets), now + self.ttl_s)

    def pending_plan(self, owner: str) -> DeletionPlan | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM pending_deletions WHERE owner = ? AND status = 'pending' "
                "ORDER BY created_at DESC LIMIT 1",
                (owner,),
            ).fetchone()
        if not row:
            return None
        ids = json.loads(row["report_ids"])
        reports = tuple(r for r in (self.get(owner, i) for i in ids) if r)
        return DeletionPlan(row["token"], owner, row["session_id"], row["criteria"], reports, row["expires_at"])

    def confirm_deletion(self, actor: str, token: str) -> DeletionOutcome:
        now = self.clock()
        with self._conn() as c:
            row = c.execute("SELECT * FROM pending_deletions WHERE token = ?", (token,)).fetchone()
            if row is None or row["status"] != "pending":
                self._audit(c, actor, "delete_rejected", token, [], {"reason": "no pending plan"})
                return DeletionOutcome("not_found", message="There is no pending deletion to confirm.")
            ids = json.loads(row["report_ids"])
            if row["owner"] != actor:
                self._audit(c, actor, "delete_denied", token, ids, {"reason": "not owner", "owner": row["owner"]})
                return DeletionOutcome("denied", message="You can only delete your own reports.")
            if now > row["expires_at"]:
                c.execute("UPDATE pending_deletions SET status = 'expired' WHERE token = ?", (token,))
                self._audit(c, actor, "delete_expired", token, ids, {"expired_at": row["expires_at"]})
                return DeletionOutcome(
                    "expired", message="That confirmation expired, so nothing was deleted. Ask again if you still want it."
                )
            # Delete exactly the snapshotted ids, and only those still owned by the actor.
            placeholders = ",".join("?" * len(ids)) or "NULL"
            owned = [
                r["id"]
                for r in c.execute(
                    f"SELECT id FROM reports WHERE owner = ? AND id IN ({placeholders})", [actor, *ids]
                ).fetchall()
            ]
            if owned:
                c.execute(
                    f"DELETE FROM reports WHERE owner = ? AND id IN ({','.join('?' * len(owned))})",
                    [actor, *owned],
                )
            c.execute("UPDATE pending_deletions SET status = 'executed' WHERE token = ?", (token,))
            self._audit(c, actor, "delete_executed", token, owned, {"criteria": row["criteria"]})
        return DeletionOutcome("deleted", tuple(owned), message=f"Deleted {len(owned)} report(s).")

    def cancel_deletion(self, actor: str, token: str, reason: str = "user cancelled") -> DeletionOutcome:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM pending_deletions WHERE token = ? AND owner = ? AND status = 'pending'",
                (token, actor),
            ).fetchone()
            if not row:
                return DeletionOutcome("not_found")
            c.execute("UPDATE pending_deletions SET status = 'cancelled' WHERE token = ?", (token,))
            self._audit(c, actor, "delete_cancelled", token, json.loads(row["report_ids"]), {"reason": reason})
        return DeletionOutcome("cancelled", message="Deletion cancelled; nothing was deleted.")

    # -- audit -------------------------------------------------------------------

    def _audit(self, c: sqlite3.Connection, actor: str, action: str, token: str | None, ids: list[int], detail: dict) -> None:
        c.execute(
            "INSERT INTO audit_log(ts, actor, action, token, report_ids, detail) VALUES (?,?,?,?,?,?)",
            (self.clock(), actor, action, token, json.dumps(ids), json.dumps(detail)),
        )

    def audit_log(self, actor: str | None = None) -> list[dict]:
        sql, args = "SELECT * FROM audit_log", []
        if actor:
            sql += " WHERE actor = ?"
            args.append(actor)
        with self._conn() as c:
            rows = c.execute(sql + " ORDER BY id", args).fetchall()
        return [
            {**dict(r), "report_ids": json.loads(r["report_ids"]), "detail": json.loads(r["detail"])}
            for r in rows
        ]
