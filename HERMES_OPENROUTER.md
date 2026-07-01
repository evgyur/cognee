# Cognee memory for Hermes with OpenRouter

This fork includes a one-key OpenRouter profile for running Cognee as a local
memory layer with:

- LLM graph extraction via OpenRouter
- 3072-dimensional embeddings via OpenRouter
- local SQLite relational store
- local Kuzu graph store
- local LanceDB vector store

No Human20 gateway, proxy, or shared key is required.

## Quick start

```bash
git clone https://github.com/evgyur/cognee.git
cd cognee

python -m venv .venv
source .venv/bin/activate
pip install -e .

cp .env.openrouter.template .env
# edit .env and set OPENROUTER_API_KEY to your own key

python examples/hermes_openrouter_memory.py
```

You can also skip the template and set only one environment variable:

```bash
export OPENROUTER_API_KEY="your-openrouter-key"
python examples/hermes_openrouter_memory.py
```

When `OPENROUTER_API_KEY` is present, `cognee.hermes_openrouter` fills in
these defaults unless you already set them yourself:

```bash
LLM_PROVIDER=openai
LLM_API_KEY=$OPENROUTER_API_KEY
LLM_MODEL=deepseek/deepseek-chat-v3
LLM_ENDPOINT=https://openrouter.ai/api/v1

EMBEDDING_PROVIDER=openai
EMBEDDING_API_KEY=$OPENROUTER_API_KEY
EMBEDDING_MODEL=openai/text-embedding-3-large
EMBEDDING_ENDPOINT=https://openrouter.ai/api/v1
EMBEDDING_DIMENSIONS=3072

COGNEE_SKIP_CONNECTION_TEST=true
```

## Hermes usage pattern

A Hermes agent can use this fork directly from Python:

```python
import asyncio
import cognee

async def main():
    await cognee.add(["Hermes can use Cognee as graph + vector memory."])
    await cognee.cognify()
    await asyncio.sleep(90)  # cognify starts an async pipeline
    print(await cognee.search("What can Hermes use Cognee for?"))

asyncio.run(main())
```

## Notes

- Use `openai/text-embedding-3-large`, not `text-embedding-3-small`, because
  the default LanceDB path expects 3072-dimensional vectors.
- The OpenRouter embedding route needs `encoding_format="float"`; this fork
  patches the LiteLLM embedding call for OpenRouter endpoints.
- `cognify()` returns after starting the pipeline. Wait for completion before
  calling `search()` in simple scripts.
- Runtime state is local and rebuildable. Do not ingest raw private chat logs or
  secrets unless that is an explicit, local-only choice.
