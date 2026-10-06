"""Output-side PII filter: the last line of defence before anything is shown or stored.

The governed views already make PII columns unqueryable. This module assumes
that layer failed anyway (a new column, a mis-configured view, the model echoing
something the user pasted) and scrubs:

* tabular results, by dropping PII-looking columns and redacting values; and
* free text (model answers, saved reports, traces), by pattern redaction.

In production this is backed by Cloud DLP (Sensitive Data Protection) inspect
templates; the regexes here are the cheap, deterministic, offline equivalent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..catalog import PII_COLUMNS

_PII_COLUMN_RE = re.compile(
    r"(e_?mail|first_?name|last_?name|full_?name|customer_?name|street|address|"
    r"latitude|longitude|\blat\b|\blng\b|\blon\b|phone|postal|zip|geom|ip_?addr)",
    re.IGNORECASE,
)

_STREET_SUFFIX = (
    r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|"
    r"Place|Pl|Square|Sq|Terrace|Way|Parkway|Pkwy|Highway|Hwy|Circle|Cir)"
)

PATTERNS: dict[str, re.Pattern[str]] = {
    "EMAIL": re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
    "PHONE": re.compile(r"(?<!\w)(?:\+\d{1,3}[\s.\-]?)?(?:\(\d{2,4}\)[\s.\-]?|\d{2,4}[\s.\-])\d{3,4}[\s.\-]\d{3,4}(?!\w)"),
    "COORDINATES": re.compile(r"-?\d{1,3}\.\d{4,}\s*,\s*-?\d{1,3}\.\d{4,}"),
    "STREET_ADDRESS": re.compile(
        rf"\b\d{{1,6}}\s+(?:[A-Z][A-Za-z']*\s+){{1,4}}{_STREET_SUFFIX}\b\.?"
    ),
    "IP_ADDRESS": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "GEOGRAPHY": re.compile(r"POINT\s*\(\s*-?\d+\.\d+\s+-?\d+\.\d+\s*\)", re.IGNORECASE),
}


@dataclass
class Redaction:
    text: str
    findings: dict[str, int] = field(default_factory=dict)

    @property
    def redacted(self) -> bool:
        return bool(self.findings)


def redact_text(text: str) -> Redaction:
    findings: dict[str, int] = {}
    out = text or ""
    for label, pattern in PATTERNS.items():
        out, n = pattern.subn(f"[REDACTED:{label}]", out)
        if n:
            findings[label] = findings.get(label, 0) + n
    return Redaction(out, findings)


def is_pii_column(name: str) -> bool:
    return name.lower() in PII_COLUMNS or bool(_PII_COLUMN_RE.search(name))


def filter_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Drop PII-named columns and redact PII-looking string values."""
    findings: dict[str, int] = {}
    clean = []
    for row in rows:
        new = {}
        for key, value in row.items():
            if is_pii_column(key):
                findings["PII_COLUMN_DROPPED"] = findings.get("PII_COLUMN_DROPPED", 0) + 1
                continue
            if isinstance(value, str):
                r = redact_text(value)
                for k, n in r.findings.items():
                    findings[k] = findings.get(k, 0) + n
                value = r.text
            new[key] = value
        clean.append(new)
    return clean, findings
