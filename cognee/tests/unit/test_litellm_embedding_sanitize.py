import pytest

from cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine import (
    LiteLLMEmbeddingEngine,
)


@pytest.mark.asyncio
async def test_embed_text_filters_invalid_inputs(monkeypatch):
    # Use monkeypatch so env is restored automatically
    monkeypatch.setenv("MOCK_EMBEDDING", "false")

    engine = LiteLLMEmbeddingEngine(dimensions=4)

    # Stub litellm call to return distinct vectors for each valid input
    UNIQUE_A = [1.0, 1.0, 1.0, 1.0]
    UNIQUE_B = [2.0, 2.0, 2.0, 2.0]
    UNIQUE_C = [2.0, 2.0, 2.0, 2.0]
    UNIQUE_D = [2.0, 2.0, 2.0, 2.0]
    UNIQUE_E = [2.0, 2.0, 2.0, 2.0]

    class _Resp:
        def __init__(self, data):
            self.data = data

    async def fake_aembedding(**kwargs):
        # kwargs["input"] should already be filtered by engine
        # In our inputs below, valid entries are "valid" and "ok!"
        return _Resp(
            [
                {"embedding": UNIQUE_A},
                {"embedding": UNIQUE_B},
                {"embedding": UNIQUE_C},
                {"embedding": UNIQUE_D},
                {"embedding": UNIQUE_E},
            ]
        )

    # Patch the litellm call used in the engine module
    import cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine as mod

    monkeypatch.setattr(mod.litellm, "aembedding", fake_aembedding)

    inputs = ["", "(", "valid", "   ", "ok!"]
    result = await engine.embed_text(inputs)

    # Output length must match input length
    assert len(result) == len(inputs)

    # Invalid entries should be zero vectors
    assert result[0] == [0.0] * 4
    assert result[3] == [0.0] * 4

    # Valid entries must map to the correct positions
    assert result[1] == UNIQUE_B
    assert result[2] == UNIQUE_C
    assert result[4] == UNIQUE_E


@pytest.mark.asyncio
async def test_openrouter_embedding_uses_float_encoding_and_omits_dimensions(monkeypatch):
    monkeypatch.setenv("MOCK_EMBEDDING", "false")

    captured_kwargs = {}

    class _Resp:
        data = [{"embedding": [1.0, 2.0, 3.0, 4.0]}]

    async def fake_aembedding(**kwargs):
        captured_kwargs.update(kwargs)
        return _Resp()

    import cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine as mod

    monkeypatch.setattr(mod.litellm, "aembedding", fake_aembedding)

    engine = LiteLLMEmbeddingEngine(
        model="openai/text-embedding-3-large",
        provider="openai",
        dimensions=4,
        api_key="test-openrouter-key",
        endpoint="https://openrouter.ai/api/v1",
    )

    result = await engine.embed_text(["hello"])

    assert result == [[1.0, 2.0, 3.0, 4.0]]
    assert captured_kwargs["encoding_format"] == "float"
    assert "dimensions" not in captured_kwargs
