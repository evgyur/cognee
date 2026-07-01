"""OpenRouter quick-start defaults for Hermes/Cognee memory.

This module intentionally uses environment variables only. It lets a user run
Cognee with graph + vector memory by setting a single secret:

    OPENROUTER_API_KEY=sk-or-v1-...

The rest of the OpenAI-compatible Cognee settings are filled in as safe defaults
unless the user already set them explicitly.
"""

from __future__ import annotations

import os

OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1"
DEFAULT_LLM_MODEL = "deepseek/deepseek-chat-v3"
DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-large"
DEFAULT_EMBEDDING_DIMENSIONS = "3072"


def apply_openrouter_defaults() -> bool:
    """Apply Cognee defaults when OPENROUTER_API_KEY is present.

    Returns True when defaults were applied. Existing explicit Cognee settings
    win; this never overwrites LLM_API_KEY, LLM_MODEL, EMBEDDING_MODEL, etc.
    """

    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_key:
        return False

    defaults = {
        # Cognee's OpenAI provider routes through LiteLLM/OpenAI-compatible APIs.
        "COGNEE_LLM_PROVIDER": "openai",
        "LLM_PROVIDER": "openai",
        "LLM_API_KEY": openrouter_key,
        "LLM_MODEL": DEFAULT_LLM_MODEL,
        "LLM_ENDPOINT": OPENROUTER_ENDPOINT,
        "LITELLM_API_BASE": OPENROUTER_ENDPOINT,
        # Cognee/LanceDB default path is 3072-dimensional. Use the matching
        # OpenAI embedding model via OpenRouter.
        "EMBEDDING_PROVIDER": "openai",
        "EMBEDDING_API_KEY": openrouter_key,
        "EMBEDDING_MODEL": DEFAULT_EMBEDDING_MODEL,
        "EMBEDDING_ENDPOINT": OPENROUTER_ENDPOINT,
        "EMBEDDING_DIMENSIONS": DEFAULT_EMBEDDING_DIMENSIONS,
        # OpenRouter connection tests can be too strict for compatible routing;
        # the real smoke is add -> cognify -> search.
        "COGNEE_SKIP_CONNECTION_TEST": "true",
    }

    for key, value in defaults.items():
        os.environ.setdefault(key, value)

    return True
