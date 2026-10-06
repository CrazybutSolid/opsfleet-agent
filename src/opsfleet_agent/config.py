"""Runtime settings, read from the environment (and .env) once at startup.

Everything that an operator may want to tune without touching code lives here.
Things a *non-developer* may want to tune (tone, persona, golden examples, user
scopes) live in files under ``config/`` and ``data/`` and are re-read at runtime.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    model: str = "gemini-3.6-flash"
    fallback_model: str = "gemini-3.5-flash-lite"
    llm_rpm: int = 8
    llm_max_retries: int = 2  # per model, before falling back
    bq_project: str | None = None
    max_bytes_billed: int = 1_000_000_000  # 1 GB hard cap per query
    max_rows: int = 200  # rows fetched from BigQuery per query
    rows_to_llm: int = 50  # rows shown to the model per query
    max_sql_failures: int = 3  # self-correction budget per turn
    max_tool_calls: int = 10  # hard stop for runaway tool loops per turn
    confirm_ttl_s: int = 120  # destructive-op confirmation window
    home: Path = field(default_factory=lambda: REPO_ROOT / "var")
    config_dir: Path = field(default_factory=lambda: REPO_ROOT / "config")
    data_dir: Path = field(default_factory=lambda: REPO_ROOT / "data")

    @property
    def db_path(self) -> Path:
        return self.home / "opsfleet.db"

    @property
    def traces_path(self) -> Path:
        return self.home / "traces.jsonl"

    @property
    def persona_path(self) -> Path:
        return self.config_dir / "persona.md"

    @property
    def users_path(self) -> Path:
        return self.config_dir / "users.toml"

    @property
    def golden_path(self) -> Path:
        return self.data_dir / "golden_trios.jsonl"


def load_settings(**overrides) -> Settings:
    load_dotenv(REPO_ROOT / ".env")
    home = Path(os.environ.get("OPSFLEET_HOME", REPO_ROOT / "var"))
    if not home.is_absolute():
        home = REPO_ROOT / home
    values = dict(
        model=os.environ.get("OPSFLEET_MODEL", Settings.model),
        fallback_model=os.environ.get("OPSFLEET_FALLBACK_MODEL", Settings.fallback_model),
        llm_rpm=_int("OPSFLEET_LLM_RPM", Settings.llm_rpm),
        bq_project=os.environ.get("GOOGLE_CLOUD_PROJECT") or None,
        max_bytes_billed=_int("OPSFLEET_MAX_BYTES_BILLED", Settings.max_bytes_billed),
        max_rows=_int("OPSFLEET_MAX_ROWS", Settings.max_rows),
        home=home,
    )
    values.update(overrides)
    settings = Settings(**values)
    settings.home.mkdir(parents=True, exist_ok=True)
    return settings
