"""Per-user preference memory (user-level learning loop).

Preferences are a small, typed profile, not free-form memories, so they are
easy to inspect, edit, and apply deterministically in the system instruction.
Each value records where it came from (``explicit`` = the user said so,
``inferred`` = the agent observed a pattern, e.g. repeatedly asking for
"shorter") and how many times it was reinforced. Explicit values always win
over inferred ones.

Production: the same profile lives in Firestore keyed by user, and Vertex AI
Memory Bank holds the long-tail, free-form memories.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

ALLOWED: dict[str, tuple[str, ...]] = {
    "format": ("table", "bullets", "prose"),
    "depth": ("brief", "standard", "deep"),
    "visuals": ("text", "charts"),
    "focus": ("revenue", "margin", "customers", "operations"),
}

DEFAULTS = {"format": "bullets", "depth": "standard", "visuals": "text"}

_INSTRUCTIONS = {
    ("format", "table"): "Present numbers as compact markdown tables.",
    ("format", "bullets"): "Present findings as short bullet points.",
    ("format", "prose"): "Write findings as short paragraphs.",
    ("depth", "brief"): "Keep answers brief: headline numbers and one-line takeaway.",
    ("depth", "standard"): "Give the key numbers, 2-4 insights and the reasoning behind them.",
    ("depth", "deep"): "Go deep: break down drivers, segments and caveats, and state methodology.",
    ("visuals", "charts"): "Where a trend or comparison would read better as a chart, describe the chart you would draw (type, axes, series) in one line.",
    ("visuals", "text"): "Do not propose charts unless asked.",
}


class PreferenceError(ValueError):
    pass


class PreferenceStore:
    def __init__(self, db_path: Path | str):
        self.db_path = str(db_path)
        with self._conn() as c:
            c.execute(
                """CREATE TABLE IF NOT EXISTS preferences (
                    user_id TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL,
                    source TEXT NOT NULL, hits INTEGER NOT NULL DEFAULT 1, updated_at REAL NOT NULL,
                    PRIMARY KEY (user_id, key))"""
            )

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def set(self, user_id: str, key: str, value: str, source: str = "explicit") -> dict:
        key, value = key.strip().lower(), value.strip().lower()
        if key not in ALLOWED:
            raise PreferenceError(f"Unknown preference '{key}'. Allowed: {', '.join(ALLOWED)}")
        if value not in ALLOWED[key]:
            raise PreferenceError(f"'{value}' is not valid for {key}. Allowed: {', '.join(ALLOWED[key])}")
        with self._conn() as c:
            row = c.execute(
                "SELECT value, source, hits FROM preferences WHERE user_id = ? AND key = ?", (user_id, key)
            ).fetchone()
            if row and row["source"] == "explicit" and source == "inferred" and row["value"] != value:
                return {"key": key, "value": row["value"], "source": "explicit", "changed": False}
            hits = row["hits"] + 1 if row and row["value"] == value else 1
            c.execute(
                "INSERT OR REPLACE INTO preferences VALUES (?,?,?,?,?,?)",
                (user_id, key, value, source, hits, time.time()),
            )
        return {"key": key, "value": value, "source": source, "changed": True}

    def get_all(self, user_id: str) -> dict[str, dict]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM preferences WHERE user_id = ?", (user_id,)).fetchall()
        return {r["key"]: {"value": r["value"], "source": r["source"], "hits": r["hits"]} for r in rows}

    def effective(self, user_id: str) -> dict[str, str]:
        prefs = dict(DEFAULTS)
        prefs.update({k: v["value"] for k, v in self.get_all(user_id).items()})
        return prefs

    def instruction(self, user_id: str) -> str:
        prefs = self.effective(user_id)
        lines = [_INSTRUCTIONS[(k, v)] for k, v in prefs.items() if (k, v) in _INSTRUCTIONS]
        if "focus" in prefs:
            lines.append(f"This user cares most about {prefs['focus']}; lead with it when relevant.")
        return "\n".join(f"- {line}" for line in lines)
