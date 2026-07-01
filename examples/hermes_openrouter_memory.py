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
from cognee.hermes_memory_policy import MemoryCandidate, MemoryTarget, classify  # noqa: E402


async def main() -> None:
    if not os.getenv("OPENROUTER_API_KEY") and not os.getenv("LLM_API_KEY"):
        raise SystemExit("Set OPENROUTER_API_KEY or copy .env.openrouter.template to .env first.")

    documents = [
        MemoryCandidate(
            text="Hermes is an AI agent framework with persistent sessions, tools, and skills.",
            source_type="documentation",
            sensitivity="public",
            metadata={"title": "Hermes overview"},
        ),
        MemoryCandidate(
            text="Cognee can turn documents into graph and vector memory for retrieval.",
            source_type="documentation",
            sensitivity="public",
            metadata={"title": "Cognee overview"},
        ),
        MemoryCandidate(
            text="OpenRouter provides OpenAI-compatible chat and embedding endpoints.",
            source_type="documentation",
            sensitivity="public",
            metadata={"title": "OpenRouter overview"},
        ),
    ]
    allowed_docs = []
    for document in documents:
        decision = classify(document)
        if decision.target is MemoryTarget.COGNEE and decision.cleaned_text:
            allowed_docs.append(decision.cleaned_text)
        else:
            print(
                f"skipped {document.metadata.get('title', '<untitled>')}: {decision.target} ({decision.reason})"
            )

    await cognee.add(allowed_docs)
    print("add: ok")

    await cognee.cognify()
    print("cognify: started; waiting 90 seconds for the background pipeline")
    await asyncio.sleep(90)

    result = await cognee.search("How can Hermes use Cognee memory with OpenRouter?")
    print("search result:")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
