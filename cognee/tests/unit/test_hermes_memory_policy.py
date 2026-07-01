"""Tests for the Hermes -> Cognee memory router policy."""

from pathlib import Path

from cognee.hermes_installer import install_hermes_skill
from cognee.hermes_memory_policy import MemoryCandidate, MemoryTarget, classify, prepare_for_cognee


def test_curated_document_routes_to_cognee():
    decision = classify(
        MemoryCandidate(
            text="This runbook describes how the local agent memory index is rebuilt.",
            source_type="runbook",
            sensitivity="internal",
            metadata={"title": "Memory index rebuild", "date": "2026-07-01"},
        )
    )

    assert decision.target is MemoryTarget.COGNEE
    assert decision.cleaned_text is not None
    assert "derived_index_only=true" in decision.cleaned_text
    assert "title=Memory index rebuild" in decision.cleaned_text


def test_ephemeral_hook_context_stays_session_only():
    decision = classify(
        MemoryCandidate(
            text="Recent private reply-chain context used to answer the current turn.",
            source_type="telegram_dm",
            sensitivity="private",
            ephemeral_hook_context=True,
        )
    )

    assert decision.target is MemoryTarget.SESSION_ONLY
    assert "ephemeral" in decision.reason


def test_raw_private_chat_is_rejected():
    decision = classify(
        MemoryCandidate(
            text="Raw direct-message transcript with personal details.",
            source_type="raw_chat",
            sensitivity="private",
        )
    )

    assert decision.target is MemoryTarget.REJECT


def test_secret_is_rejected_even_in_document():
    decision = classify(
        MemoryCandidate(
            text="Deployment note: token='super-secret-token-value' should never be indexed.",
            source_type="runbook",
            sensitivity="internal",
        )
    )

    assert decision.target is MemoryTarget.REJECT


def test_short_preference_routes_to_hermes_memory():
    decision = classify(
        MemoryCandidate(
            text="User prefers concise responses.",
            source_type="note",
            sensitivity="internal",
        )
    )

    assert decision.target is MemoryTarget.HERMES_MEMORY


def test_prepare_for_cognee_returns_none_for_rejected_items():
    result = prepare_for_cognee(
        MemoryCandidate(
            text="API_KEY='super-secret-token-value'",
            source_type="documentation",
            sensitivity="internal",
        )
    )

    assert result is None


def test_hermes_skill_installer_writes_public_safe_skill(tmp_path: Path):
    skill_path = install_hermes_skill(tmp_path)
    text = skill_path.read_text(encoding="utf-8")

    assert skill_path == tmp_path / "skills" / "cognee-memory-router" / "SKILL.md"
    assert "Context-first hook gate" in text
    assert "Do **not** save raw private messages" in text
    assert "private operator name" not in text
