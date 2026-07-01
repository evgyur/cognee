"""Public-safe Hermes -> Cognee memory router policy.

Cognee should receive curated, durable documents. It should not receive raw
private chats, credentials, ephemeral hook context, or temporary task progress.
This module is intentionally small and dependency-free so Hermes hooks, cron
jobs, and examples can call it before `cognee.add()`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import re
from typing import Iterable


class MemoryTarget(StrEnum):
    """Where a memory candidate should go."""

    IGNORE = "ignore"
    HERMES_MEMORY = "hermes_memory"
    SKILL = "skill"
    SESSION_ONLY = "session_only"
    COGNEE = "cognee"
    ASK_USER = "ask_user"
    REJECT = "reject"


class Sensitivity(StrEnum):
    PUBLIC = "public"
    INTERNAL = "internal"
    PRIVATE = "private"
    SECRET = "secret"


@dataclass(frozen=True)
class MemoryCandidate:
    """A candidate item before it is routed to Cognee or another memory layer."""

    text: str
    source_type: str = "note"
    sensitivity: Sensitivity | str = Sensitivity.INTERNAL
    ephemeral_hook_context: bool = False
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryDecision:
    target: MemoryTarget
    reason: str
    cleaned_text: str | None = None
    requires_review: bool = False
    metadata: dict[str, str] = field(default_factory=dict)


SECRET_PATTERNS = [
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bsk-or-v1-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bsk-proj-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{32,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"(?i)\b(api[_-]?key|token|password|secret)\s*[:=]\s*['\"][^'\"]{8,}['\"]"),
]

PRIVATE_SOURCE_TYPES = {
    "raw_chat",
    "telegram_dm",
    "private_dm",
    "session_dump",
    "transcript",
    "logs",
    "credential",
    "env",
}

COGNEE_SOURCE_TYPES = {
    "doc",
    "documentation",
    "readme",
    "spec",
    "architecture_note",
    "adr",
    "runbook",
    "research_note",
    "handoff",
    "clean_summary",
    "knowledge_base",
}

SKILL_HINTS = (
    "procedure",
    "workflow",
    "runbook steps",
    "how to",
    "checklist",
    "repeatable",
)

TEMPORARY_HINTS = (
    "todo",
    "in progress",
    "step 1",
    "step 2",
    "step 3",
    "temporary",
    "draft",
    "right now",
    "current task",
)

STABLE_PREFERENCE_HINTS = (
    "prefers",
    "preference",
    "always wants",
    "style",
    "speaks",
    "likes concise",
)


_REDACTIONS = [
    (
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
        "[REDACTED_PRIVATE_KEY]",
    ),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"), "[REDACTED_GITHUB_TOKEN]"),
    (re.compile(r"\bsk-or-v1-[A-Za-z0-9_-]{20,}\b"), "[REDACTED_OPENROUTER_KEY]"),
    (re.compile(r"\bsk-proj-[A-Za-z0-9_-]{20,}\b"), "[REDACTED_OPENAI_KEY]"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{32,}\b"), "[REDACTED_API_KEY]"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[REDACTED_AWS_KEY]"),
    (
        re.compile(r"(?i)\b(api[_-]?key|token|password|secret)\s*[:=]\s*['\"][^'\"]{8,}['\"]"),
        r"\1=[REDACTED_SECRET]",
    ),
]


def contains_secret(text: str) -> bool:
    return any(pattern.search(text) for pattern in SECRET_PATTERNS)


def redact_secrets(text: str) -> str:
    redacted = text
    for pattern, replacement in _REDACTIONS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def _contains_any(text: str, hints: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(hint in lowered for hint in hints)


def normalize_for_cognee(candidate: MemoryCandidate) -> str:
    """Return a stable, provenance-bearing document for Cognee."""

    cleaned = redact_secrets(candidate.text).strip()
    provenance = []
    if candidate.source_type:
        provenance.append(f"source_type={candidate.source_type}")
    if candidate.sensitivity:
        provenance.append(f"sensitivity={str(candidate.sensitivity)}")
    for key in ("source", "date", "title"):
        value = candidate.metadata.get(key)
        if value:
            provenance.append(f"{key}={value}")
    header = "\n".join(provenance)
    return f"---\n{header}\nderived_index_only=true\n---\n\n{cleaned}" if header else cleaned


def classify(candidate: MemoryCandidate) -> MemoryDecision:
    """Classify a candidate before any Cognee ingestion.

    Policy is default-deny for Cognee. Only curated, durable document-like
    sources are allowed automatically. Ambiguous private material requires
    review; secrets and ephemeral hook context are rejected.
    """

    text = candidate.text.strip()
    source_type = candidate.source_type.lower().strip()
    sensitivity = Sensitivity(candidate.sensitivity)

    if not text:
        return MemoryDecision(MemoryTarget.IGNORE, "empty candidate")

    if candidate.ephemeral_hook_context:
        return MemoryDecision(
            MemoryTarget.SESSION_ONLY,
            "ephemeral hook context is for the current turn only; do not persist raw chat",
        )

    if (
        contains_secret(text)
        or sensitivity is Sensitivity.SECRET
        or source_type in {"credential", "env"}
    ):
        return MemoryDecision(MemoryTarget.REJECT, "secret or credential-like content")

    if source_type in PRIVATE_SOURCE_TYPES:
        return MemoryDecision(
            MemoryTarget.REJECT,
            "raw private streams, logs, sessions, and transcripts are not Cognee inputs",
        )

    if _contains_any(text, TEMPORARY_HINTS):
        return MemoryDecision(
            MemoryTarget.SESSION_ONLY, "temporary task state should stay in session history"
        )

    if _contains_any(text, SKILL_HINTS):
        return MemoryDecision(MemoryTarget.SKILL, "repeatable procedure belongs in a Hermes skill")

    if _contains_any(text, STABLE_PREFERENCE_HINTS) and len(text) <= 500:
        return MemoryDecision(
            MemoryTarget.HERMES_MEMORY,
            "short stable preference/fact belongs in built-in Hermes memory",
            cleaned_text=redact_secrets(text),
        )

    if source_type in COGNEE_SOURCE_TYPES and sensitivity in {
        Sensitivity.PUBLIC,
        Sensitivity.INTERNAL,
    }:
        cleaned = normalize_for_cognee(candidate)
        return MemoryDecision(
            MemoryTarget.COGNEE,
            "curated durable document is safe for Cognee derived indexing",
            cleaned_text=cleaned,
            metadata={"derived_index_only": "true"},
        )

    if sensitivity is Sensitivity.PRIVATE:
        return MemoryDecision(
            MemoryTarget.ASK_USER,
            "private or ambiguous material requires explicit review before indexing",
            requires_review=True,
        )

    return MemoryDecision(
        MemoryTarget.ASK_USER,
        "default deny: not a recognized curated Cognee document",
        requires_review=True,
    )


def prepare_for_cognee(candidate: MemoryCandidate) -> str | None:
    """Return cleaned text if the candidate is allowed into Cognee; otherwise None."""

    decision = classify(candidate)
    if decision.target is MemoryTarget.COGNEE:
        return decision.cleaned_text
    return None
