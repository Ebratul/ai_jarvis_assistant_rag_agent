"""Text embedders.

* ``hash``        – deterministic hashed character/word n-grams. Zero downloads, works for any
                    script (English, Bangla, Banglish) and never changes, so vectors written today
                    are still comparable in 20 years.
* ``fastembed``   – ONNX multilingual MiniLM (optional ``requirements-ml.txt``).
* ``huggingface`` – Hugging Face Inference API feature-extraction (needs ``HF_TOKEN``).

Every embedder returns L2-normalised float32 vectors. If a neural backend fails, NONO falls back
to ``hash`` so a memory is never lost because a model was unavailable.
"""
from __future__ import annotations

import logging
import re
import unicodedata
import zlib
from typing import Protocol

import httpx
import numpy as np

from .config import settings

log = logging.getLogger("nono.rag.embed")

_WORD = re.compile(r"[\wঀ-৿]+", re.UNICODE)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "").lower()
    return re.sub(r"\s+", " ", text).strip()


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


class HashEmbedder:
    """Signed feature hashing of word unigrams/bigrams and char 3/4-grams."""

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.name = f"hash-ngram-v1-{dim}"

    def _features(self, text: str) -> list[tuple[str, float]]:
        t = normalize(text)
        words = _WORD.findall(t)
        feats: list[tuple[str, float]] = [("w:" + w, 2.0) for w in words]
        feats += [("b:" + a + "_" + b, 1.5) for a, b in zip(words, words[1:])]
        for w in words:
            padded = f"<{w}>"
            for n in (3, 4):
                if len(padded) >= n:
                    feats += [("c:" + padded[i : i + n], 1.0) for i in range(len(padded) - n + 1)]
        return feats

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feat, weight in self._features(text):
                h = zlib.crc32(feat.encode("utf-8"))
                sign = 1.0 if (h >> 31) & 1 else -1.0
                out[row, h % self.dim] += sign * weight
        return _l2(out)


class FastEmbedEmbedder:
    def __init__(self, model: str):
        from fastembed import TextEmbedding  # optional dependency

        self._model = TextEmbedding(model)
        probe = np.array(list(self._model.embed(["probe"])), dtype=np.float32)
        self.dim = int(probe.shape[1])
        self.name = f"fastembed:{model}"

    def embed(self, texts: list[str]) -> np.ndarray:
        return _l2(np.array(list(self._model.embed(texts)), dtype=np.float32))


class HuggingFaceEmbedder:
    def __init__(self, model: str, token: str):
        if not token:
            raise RuntimeError("HF_TOKEN is required for the huggingface embedding backend")
        self._url = f"https://router.huggingface.co/hf-inference/models/{model}/pipeline/feature-extraction"
        self._headers = {"Authorization": f"Bearer {token}"}
        self.name = f"hf:{model}"
        self.dim = int(self.embed(["probe"]).shape[1])

    def embed(self, texts: list[str]) -> np.ndarray:
        r = httpx.post(self._url, headers=self._headers, json={"inputs": texts}, timeout=60)
        r.raise_for_status()
        arr = np.array(r.json(), dtype=np.float32)
        if arr.ndim == 3:  # token embeddings -> mean pool
            arr = arr.mean(axis=1)
        return _l2(arr)


def _l2(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms).astype(np.float32)


def build_embedder() -> Embedder:
    backend = settings.embedding_backend
    try:
        if backend == "fastembed":
            return FastEmbedEmbedder(settings.fastembed_model)
        if backend == "huggingface":
            return HuggingFaceEmbedder(settings.hf_embed_model, settings.hf_token)
    except Exception as exc:  # pragma: no cover - depends on optional deps / network
        log.warning("Embedding backend %s unavailable (%s); falling back to hash", backend, exc)
    return HashEmbedder()
