"""Smoke test for Cognee graph + vector memory through OpenRouter.

Usage:
    export OPENROUTER_API_KEY="your-openrouter-key"
    python examples/hermes_openrouter_memory.py

or:
    cp .env.openrouter.template .env
    python examples/hermes_openrouter_memory.py
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

# Prefer a local .env created from .env.openrouter.template.
repo_root = Path(__file__).resolve().parents[1]
load_dotenv(repo_root / ".env", override=True)

import cognee  # noqa: E402  # loads OPENROUTER_API_KEY defaults when present


async def main() -> None:
    if not os.getenv("OPENROUTER_API_KEY") and not os.getenv("LLM_API_KEY"):
        raise SystemExit("Set OPENROUTER_API_KEY or copy .env.openrouter.template to .env first.")

    await cognee.add(
        [
            "Hermes is an AI agent framework with persistent sessions, tools, and skills.",
            "Cognee can turn documents into graph and vector memory for retrieval.",
            "OpenRouter provides OpenAI-compatible chat and embedding endpoints.",
        ]
    )
    print("add: ok")

    await cognee.cognify()
    print("cognify: started; waiting 90 seconds for the background pipeline")
    await asyncio.sleep(90)

    result = await cognee.search("How can Hermes use Cognee memory with OpenRouter?")
    print("search result:")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
