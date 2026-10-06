"""Golden-trio store: seed quality, similarity retrieval, hot reload, review-gated updates."""

from __future__ import annotations

import json
import os

import pytest
from conftest import REPO_ROOT

from opsfleet_agent.catalog import PII_COLUMNS
from opsfleet_agent.knowledge.golden import GoldenStore
from opsfleet_agent.security.scope import load_users
from opsfleet_agent.security.sql_guard import govern

SEED = REPO_ROOT / "data" / "golden_trios.jsonl"


def test_seed_has_at_least_ten_valid_trios():
    trios = [json.loads(line) for line in SEED.read_text().splitlines() if line.strip()]
    assert len(trios) >= 10
    harry = load_users(REPO_ROOT / "config" / "users.toml")["harry"]
    for t in trios:
        assert t["question"] and t["report"] and t["sql"]
        govern(t["sql"], harry)  # every golden SQL passes the guard (no PII, allowed tables only)
        assert not any(f" {c}" in t["sql"] for c in PII_COLUMNS)


@pytest.mark.parametrize("question,expected", [
    ("Why are customers in New York spending less than in Florida?", "trio-004"),
    ("Why did churn go up last month?", "trio-005"),
    ("Create a report for Q3 with action items for Q4", "trio-006"),
    ("Compare Nike and Adidas, why the difference?", "trio-003"),
    ("Who are the biggest spenders?", "trio-001"),
    ("Which categories make the most profit?", "trio-009"),
    ("Monthly revenue trend", "trio-002"),
])
def test_retrieval_finds_the_analogous_trio(question, expected):
    hits = GoldenStore(SEED).search(question, k=2)
    assert expected in [t.id for t, _ in hits]


def test_unrelated_questions_retrieve_nothing():
    assert GoldenStore(SEED).search("hello there", k=2) == []


def test_new_trios_are_picked_up_without_restart_and_unapproved_are_ignored(tmp_path):
    path = tmp_path / "trios.jsonl"
    path.write_text(json.dumps({"id": "a", "question": "basket size by weekday", "sql": "SELECT 1", "report": "r"}) + "\n")
    store = GoldenStore(path)
    assert store.search("basket size weekday", min_score=0.1)[0][0].id == "a"
    with path.open("a") as f:
        f.write(json.dumps({"id": "b", "question": "inventory turnover", "sql": "SELECT 2", "report": "r"}) + "\n")
        f.write(json.dumps({"id": "c", "question": "inventory turnover draft", "sql": "SELECT 3", "report": "r",
                            "approved": False}) + "\n")
    os.utime(path, (path.stat().st_atime, path.stat().st_mtime + 5))
    ids = [t.id for t, _ in store.search("inventory turnover", min_score=0.1)]
    assert ids == ["b"]
