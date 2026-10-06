"""Input guard: cheap, deterministic screening of user messages before the LLM.

It catches the unambiguous cases with zero latency and zero tokens:
prompt-injection / jailbreak phrasing, explicit requests for personal data, and
clearly off-topic tasks. Everything else goes to the model, whose system
instruction carries the same policy (answer retail-analytics questions only),
so subtle cases are still refused, just not deterministically.

Production adds a classifier in front (Model Armor prompt-injection + jailbreak
filters, or a small Gemini Flash-Lite classifier with a structured verdict).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class Verdict(str, Enum):
    ALLOW = "allow"
    INJECTION = "injection"
    PII_REQUEST = "pii_request"
    OFF_TOPIC = "off_topic"


@dataclass(frozen=True)
class GuardResult:
    verdict: Verdict
    reason: str = ""

    @property
    def allowed(self) -> bool:
        return self.verdict is Verdict.ALLOW


_INJECTION = [
    r"\b(ignore|disregard|forget|override)\b.{0,40}\b(previous|prior|above|all|earlier|your|system)\b.{0,40}\b(instructions?|rules?|prompts?|guidelines|polic(y|ies))",
    r"\b(reveal|show|print|repeat|output|leak|tell me)\b.{0,40}\b(system|hidden|initial|developer)\s+(prompt|instructions?|message)",
    r"\byou are (now|no longer)\b",
    r"\b(pretend (you are|you're|to be)|act as|roleplay as|role-play as)\b.{0,40}\b(dan|developer mode|unfiltered|jailbroken|admin|root|dba|unrestricted)",
    r"\b(developer|god|dan|jailbreak) mode\b",
    r"\bjailbreak",
    r"\bnew (system )?instructions?:",
    r"<\s*/?\s*(system|instructions?)\s*>",
    r"\b(bypass|disable|turn off)\b.{0,30}\b(guard|filter|safety|restrictions?|masking|scope)",
    r"\b(drop|truncate|delete from|insert into|update\s+\w+\s+set|alter table|grant)\b.{0,40}\b(table|orders|users|products|order_items|dataset)\b",
    r"\b(other|all) (users'?|people'?s?) (reports|scope|data)\b",
    r"\b(as|impersonate|switch to|log ?in as)\s+(user\s+)?(alice|bob|carol|harry|admin|another user)\b",
]

# Note "Email" is also an acquisition channel (traffic_source), so a bare "email"
# is fine; asking for addresses / "their emails" is not.
_PII_REQUEST = [
    r"\b(e-?mail address(es)?|phone (numbers?|nos?)|mobile numbers?|home address(es)?|street address(es)?|"
    r"mailing address(es)?|postal codes?|zip codes?|gps|coordinates|latitudes?|longitudes?|geolocation)\b",
    r"\b(their|customers?'?|users?'?|buyers?'?|shoppers?'?|clients?'?|his|her)\s+(e-?mails?|names?|surnames?|addresses|address|phones?)\b",
    r"\be-?mails? (of|for) (the |our |these |those |top |each |all |\d+ )*(customers?|users?|buyers?|shoppers?|clients?)\b",
    r"\b(full|first|last) names?\b|\bsurnames?\b",
    r"\b(names?|identity|identities|contact (details?|info))\s+of\s+(the\s+|our\s+|these\s+|those\s+)?(top\s+|best\s+|biggest\s+)?\d*\s*(customers?|users?|buyers?|shoppers?)\b",
    r"\bwho (exactly )?(is|are) (customer|user) (id\s*)?#?\d+",
    r"\b(de-?anonymi[sz]e|re-?identify|unmask|dox)\b",
]

_OFF_TOPIC = [
    r"\b(write|compose|generate)\b.{0,30}\b(poem|song|story|joke|essay|cover letter|haiku|novel)\b",
    r"\b(weather|horoscope|lottery|recipe|football|soccer|nba|movie|netflix)\b",
    r"\b(write|debug|fix|explain)\b.{0,30}\b(python|javascript|java|react|regex|code|script|program)\b(?!.{0,40}\b(sql|query)\b)",
    r"\b(medical|legal|tax|investment|stock|crypto|bitcoin)\s+(advice|tips|recommendation)",
    r"\b(capital of|who won|translate (this|to)|what is the meaning of life)\b",
    r"\b(hack|exploit|malware|phishing|ddos)\b",
]

_COMPILED = {
    Verdict.INJECTION: [re.compile(p, re.IGNORECASE | re.DOTALL) for p in _INJECTION],
    Verdict.PII_REQUEST: [re.compile(p, re.IGNORECASE | re.DOTALL) for p in _PII_REQUEST],
    Verdict.OFF_TOPIC: [re.compile(p, re.IGNORECASE | re.DOTALL) for p in _OFF_TOPIC],
}

REFUSALS = {
    Verdict.INJECTION: (
        "I can't do that. I only answer retail analytics questions, within your data access, "
        "and I can't change my instructions or safeguards."
    ),
    Verdict.PII_REQUEST: (
        "I can't share personal customer data such as names, emails, phone numbers, addresses "
        "or locations. I can analyse customers by pseudonymous ID and demographics "
        "(age, gender, city, state, country, acquisition channel) instead."
    ),
    Verdict.OFF_TOPIC: (
        "That's outside what I can help with. I'm the retail analytics assistant: ask me about "
        "sales, products, customers, orders, trends, or ask for a report."
    ),
}


def screen(message: str) -> GuardResult:
    text = message or ""
    if len(text) > 4000:
        return GuardResult(Verdict.INJECTION, "message too long (possible prompt stuffing)")
    for verdict in (Verdict.INJECTION, Verdict.PII_REQUEST, Verdict.OFF_TOPIC):
        for pattern in _COMPILED[verdict]:
            m = pattern.search(text)
            if m:
                return GuardResult(verdict, f"matched: {m.group(0)[:80]!r}")
    return GuardResult(Verdict.ALLOW)
