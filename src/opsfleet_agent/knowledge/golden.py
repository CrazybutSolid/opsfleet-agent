"""Golden Knowledge: retrieval of analyst-authored (question -> SQL -> report) trios.

Prototype: trios live in ``data/golden_trios.jsonl`` and are ranked with BM25
over question + tags. It is deterministic, needs no API calls (free-tier
friendly) and is good enough for a few hundred trios.

Production: trios land in a GCS bucket, a Cloud Run ingestion job validates the
SQL (dry run against the governed views), scrubs PII, embeds question + report
with ``gemini-embedding-001`` and upserts into Vertex AI Vector Search (or
BigQuery ``VECTOR_SEARCH``); retrieval becomes hybrid (vector + BM25, rank
fusion) with metadata filters (scope, freshness, approved=true).

The learning loop writes *candidates* (answers users rated as good) to a
separate file; an analyst approves them into the golden set. Nothing the model
produces becomes golden knowledge without human review.
"""

from __future__ import annotations

import json
import math
import re
import threading
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an the of for to in on and or is are was were our we us by with what which who how why "
    "do does did me my you your it its this that these those be from at as than vs".split()
)


# Geography words map to a shared token so "New York vs Florida" finds the
# "Texas vs California" trio. Production gets this for free from embeddings.
_REGIONS = re.compile(
    r"\b(alabama|alaska|arizona|arkansas|california|colorado|connecticut|delaware|florida|georgia|"
    r"hawaii|idaho|illinois|indiana|iowa|kansas|kentucky|louisiana|maine|maryland|massachusetts|"
    r"michigan|minnesota|mississippi|missouri|montana|nebraska|nevada|new hampshire|new jersey|"
    r"new mexico|new york|north carolina|north dakota|ohio|oklahoma|oregon|pennsylvania|"
    r"rhode island|south carolina|south dakota|tennessee|texas|utah|vermont|virginia|washington|"
    r"west virginia|wisconsin|wyoming|region|province|city|cities|country|countries)\b"
)


def tokenize(text: str) -> list[str]:
    text = _REGIONS.sub(" state ", text.lower())
    tokens = []
    for t in _TOKEN.findall(text):
        if t in _STOP:
            continue
        # crude stemming: plurals and -ing, enough for short business questions
        for suffix in ("ing", "ed", "es", "s"):
            if len(t) > 4 and t.endswith(suffix):
                t = t[: -len(suffix)]
                break
        tokens.append(t)
    return tokens


@dataclass(frozen=True)
class Trio:
    id: str
    question: str
    sql: str
    report: str
    tags: tuple[str, ...] = ()

    def document(self) -> str:
        return f"{self.question} {' '.join(self.tags)}"


@dataclass
class GoldenStore:
    path: Path
    candidates_path: Path | None = None
    k1: float = 1.5
    b: float = 0.75
    trios: list[Trio] = field(default_factory=list)
    _mtime: float = -1.0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def _load(self) -> None:
        """(Re)load when the file changes, so analysts can add trios without a restart."""
        mtime = self.path.stat().st_mtime if self.path.exists() else 0.0
        if mtime == self._mtime:
            return
        with self._lock:
            trios = []
            if self.path.exists():
                for line in self.path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    raw = json.loads(line)
                    if not raw.get("approved", True):
                        continue
                    trios.append(
                        Trio(raw["id"], raw["question"], raw["sql"], raw["report"], tuple(raw.get("tags", ())))
                    )
            self.trios = trios
            self._docs = [tokenize(t.document()) for t in trios]
            self._df = Counter(tok for doc in self._docs for tok in set(doc))
            self._avgdl = (sum(len(d) for d in self._docs) / len(self._docs)) if self._docs else 0.0
            self._mtime = mtime

    def search(self, query: str, k: int = 2, min_score: float = 2.5) -> list[tuple[Trio, float]]:
        self._load()
        q = tokenize(query)
        if not q or not self.trios:
            return []
        n = len(self.trios)
        scored = []
        for trio, doc in zip(self.trios, self._docs):
            tf = Counter(doc)
            score = 0.0
            for tok in q:
                if tok not in tf:
                    continue
                idf = math.log(1 + (n - self._df[tok] + 0.5) / (self._df[tok] + 0.5))
                f = tf[tok]
                score += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(doc) / self._avgdl))
            if score >= min_score:
                scored.append((trio, round(score, 3)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def add_candidate(self, question: str, sql: str, report: str, user_id: str, trace_id: str) -> None:
        """Queue a user-endorsed answer for analyst review (system-level learning loop)."""
        if not self.candidates_path:
            return
        self.candidates_path.parent.mkdir(parents=True, exist_ok=True)
        with self.candidates_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {"question": question, "sql": sql, "report": report, "proposed_by": user_id,
                     "trace_id": trace_id, "approved": False},
                    ensure_ascii=False,
                )
                + "\n"
            )


def format_for_prompt(results: list[tuple[Trio, float]]) -> str:
    if not results:
        return ""
    blocks = []
    for trio, score in results:
        blocks.append(
            f"### {trio.id} (similarity {score})\n"
            f"Question: {trio.question}\nSQL:\n```sql\n{trio.sql}\n```\nAnalyst report:\n{trio.report}"
        )
    return "\n\n".join(blocks)
