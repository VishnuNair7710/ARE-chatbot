"""Dependency-free lexical 'embeddings' (hashed bag of words).

Two jobs: deterministic embeddings for offline tests, and an emergency fallback if the local
neural model cannot be downloaded. It matches on shared words, not meaning, so its
similarity scale differs from a neural model: it carries its own thresholds and never
returns answers verbatim on its own (direct threshold > 1)."""
from __future__ import annotations

import hashlib
import math
import re

DIM = 4096
_STOP = {"a", "an", "the", "is", "are", "do", "does", "you", "your", "i", "me", "my", "we", "of", "to",
         "and", "or", "in", "on", "for", "can", "how", "what", "who", "where", "when", "which", "it",
         "there", "have", "has", "with", "be", "at", "as", "this", "that", "will", "would", "could"}


def _tokens(text: str) -> list[str]:
    out = []
    for w in re.findall(r"[a-z0-9£']+", text.lower()):
        if w in _STOP:
            continue
        out.append(w[:-1] if len(w) > 3 and w.endswith("s") else w)
    return out


class HashEmbedder:
    model_name = "hash-bow-4096"
    lexical = True              # matches shared words, not meaning: must not overrule a model's judgement
    thresholds = (1.01, 0.30)   # (direct, min): never answer verbatim; low floor for 'not in data'

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * DIM
        for t in _tokens(text):
            v[int(hashlib.md5(t.encode()).hexdigest(), 16) % DIM] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)
