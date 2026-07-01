"""Tests for OPENROUTER_API_KEY one-key defaults."""

from cognee.hermes_openrouter import (
    DEFAULT_EMBEDDING_DIMENSIONS,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_MODEL,
    OPENROUTER_ENDPOINT,
    apply_openrouter_defaults,
)


_OPENROUTER_ENV = [
    "OPENROUTER_API_KEY",
    "COGNEE_LLM_PROVIDER",
    "LLM_PROVIDER",
    "LLM_API_KEY",
    "LLM_MODEL",
    "LLM_ENDPOINT",
    "LITELLM_API_BASE",
    "EMBEDDING_PROVIDER",
    "EMBEDDING_API_KEY",
    "EMBEDDING_MODEL",
    "EMBEDDING_ENDPOINT",
    "EMBEDDING_DIMENSIONS",
    "COGNEE_SKIP_CONNECTION_TEST",
]


def test_apply_openrouter_defaults_requires_key(monkeypatch):
    for name in _OPENROUTER_ENV:
        monkeypatch.delenv(name, raising=False)

    assert apply_openrouter_defaults() is False
    assert "LLM_API_KEY" not in __import__("os").environ


def test_apply_openrouter_defaults_maps_one_key(monkeypatch):
    for name in _OPENROUTER_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")

    assert apply_openrouter_defaults() is True

    import os

    assert os.environ["LLM_PROVIDER"] == "openai"
    assert os.environ["LLM_API_KEY"] == "test-openrouter-key"
    assert os.environ["LLM_MODEL"] == DEFAULT_LLM_MODEL
    assert os.environ["LLM_ENDPOINT"] == OPENROUTER_ENDPOINT
    assert os.environ["EMBEDDING_PROVIDER"] == "openai"
    assert os.environ["EMBEDDING_API_KEY"] == "test-openrouter-key"
    assert os.environ["EMBEDDING_MODEL"] == DEFAULT_EMBEDDING_MODEL
    assert os.environ["EMBEDDING_ENDPOINT"] == OPENROUTER_ENDPOINT
    assert os.environ["EMBEDDING_DIMENSIONS"] == DEFAULT_EMBEDDING_DIMENSIONS
    assert os.environ["COGNEE_SKIP_CONNECTION_TEST"] == "true"


def test_apply_openrouter_defaults_preserves_explicit_overrides(monkeypatch):
    for name in _OPENROUTER_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("LLM_MODEL", "anthropic/claude-sonnet-4.5")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "1536")

    assert apply_openrouter_defaults() is True

    import os

    assert os.environ["LLM_MODEL"] == "anthropic/claude-sonnet-4.5"
    assert os.environ["EMBEDDING_DIMENSIONS"] == "1536"
