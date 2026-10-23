"""Text embedders for long-term memory.

`HashingEmbedder` is deterministic and dependency-free (signed feature hashing over word
unigrams and bigrams). It gives useful lexical recall with no API calls, which keeps
tests and offline demos self-contained. For semantic recall in production, use a model
embedder (`OpenAIEmbedder`, or implement `Embedder` for Voyage, Cohere, a local model).
"""

from __future__ import annotations

import hashlib
import itertools
import math
import re
from collections.abc import Sequence
from typing import Protocol

_WORD = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    dim: int

    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm else vec


class HashingEmbedder:
    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _one(self, text: str) -> list[float]:
        words = _WORD.findall(text.lower())
        features = words + [f"{a} {b}" for a, b in itertools.pairwise(words)]
        vec = [0.0] * self.dim
        for feat in features:
            h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "big")
            vec[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return _normalize(vec)

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]


class OpenAIEmbedder:
    """OpenAI embeddings with `dimensions` matched to the vector column."""

    def __init__(
        self,
        *,
        model: str = "text-embedding-3-small",
        dim: int = 256,
        api_key: str | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        import openai

        self.model = model
        self.dim = dim
        self._client = openai.AsyncOpenAI(api_key=api_key, timeout=timeout_s, max_retries=2)

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        resp = await self._client.embeddings.create(
            model=self.model, input=list(texts), dimensions=self.dim
        )
        return [list(d.embedding) for d in resp.data]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0
