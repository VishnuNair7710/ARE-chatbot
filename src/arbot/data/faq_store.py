"""FAQ retrieval (RAG) over faq.csv with a Chroma vector store.

Each FAQ row is one chunk (47 short Q&A pairs: splitting would only hurt). The stored
answer is returned, never a generated paraphrase, unless the LLM is asked to synthesise.
The collection name embeds a hash of the CSV and the embedding model, so editing the FAQ or
switching models transparently builds a fresh index.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class FaqHit:
    faq_id: str
    category: str
    question: str
    answer: str
    score: float      # cosine similarity, 1.0 = identical


class FaqStore:
    def __init__(self, faq: pd.DataFrame, embedder, path: str | None = None):
        import chromadb

        self.embedder = embedder
        self._faq = faq
        digest = hashlib.sha256(faq.to_csv(index=False).encode()).hexdigest()[:12]
        model = hashlib.sha256(str(getattr(embedder, "model_name", type(embedder).__name__)).encode()).hexdigest()[:8]
        name = f"faq_v2_{digest}_{model}"   # v2 = question-only embeddings; bump when indexing changes
        self._client = chromadb.PersistentClient(path=path) if path else chromadb.EphemeralClient()
        existing = {c if isinstance(c, str) else c.name for c in self._client.list_collections()}
        if path:  # drop stale indexes from earlier CSV versions / models
            for old in existing - {name}:
                if old.startswith("faq_"):
                    self._client.delete_collection(old)
        self._col = self._client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
        if self._col.count() != len(faq):
            self._index()

    def _index(self) -> None:
        f = self._faq
        # Embed the QUESTION only: visitors' queries resemble questions, and answers carry
        # unrelated entities (an answer listing every branch would match any branch query).
        self._col.upsert(
            ids=list(f["faq_id"]),
            embeddings=self.embedder.embed_documents(list(f["question"])),
            documents=list(f["answer"]),
            metadatas=[{"category": c, "question": q} for c, q in zip(f["category"], f["question"])],
        )

    def search(self, query: str, k: int = 3) -> list[FaqHit]:
        res = self._col.query(query_embeddings=[self.embedder.embed_query(query)], n_results=k)
        return [
            FaqHit(fid, meta["category"], meta["question"], doc, 1.0 - dist)
            for fid, meta, doc, dist in zip(res["ids"][0], res["metadatas"][0],
                                            res["documents"][0], res["distances"][0])
        ]
