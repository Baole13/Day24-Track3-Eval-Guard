from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys
from dataclasses import dataclass
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    try:
        from underthesea import word_tokenize
        segmented = word_tokenize(text, format="text")
        return segmented.replace("_", " ")
    except Exception:
        return text


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        self.documents = chunks
        self.corpus_tokens = [segment_vietnamese(chunk["text"]).split() for chunk in chunks]
        from rank_bm25 import BM25Okapi
        self.bm25 = BM25Okapi(self.corpus_tokens) if self.corpus_tokens else None

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None:
            return []
        tokenized_query = segment_vietnamese(query).split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            SearchResult(
                text=self.documents[i]["text"],
                score=float(scores[i]),
                metadata=self.documents[i].get("metadata", {}),
                method="bm25",
            )
            for i in top_indices
            if scores[i] > 0
        ]


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self._encoder = None
        self._fallback_documents = []
        self._fallback_vectors = []

    def _get_encoder(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                try:
                    self._encoder = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
                except Exception as primary_error:
                    print(f"  [warn] Failed to load {EMBEDDING_MODEL}: {primary_error}")
                    self._encoder = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
            except Exception as fallback_error:
                print(f"  [warn] Falling back to lightweight hash encoder: {fallback_error}")
                self._encoder = _HashEncoder()
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        from qdrant_client.models import Distance, VectorParams, PointStruct

        texts = [c["text"] for c in chunks]
        if not texts:
            return
        vectors = self._get_encoder().encode(texts, show_progress_bar=True)
        self._fallback_documents = chunks
        self._fallback_vectors = vectors

        try:
            self.client.recreate_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE),
            )
            points = [
                PointStruct(
                    id=i,
                    vector=v.tolist(),
                    payload={**c.get("metadata", {}), "text": c["text"]},
                )
                for i, (c, v) in enumerate(zip(chunks, vectors))
            ]
            self.client.upsert(collection_name=collection, points=points)
        except Exception as e:
            print(f"  [warn] Qdrant unavailable, using in-memory dense index: {e}")

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        query_vector = self._get_encoder().encode(query)
        try:
            response = self.client.query_points(
                collection_name=collection,
                query=query_vector.tolist(),
                limit=top_k,
            )
            return [
                SearchResult(
                    text=pt.payload["text"],
                    score=float(pt.score),
                    metadata=dict(pt.payload),
                    method="dense",
                )
                for pt in response.points
            ]
        except Exception:
            if len(self._fallback_documents) == 0:
                return []

            from numpy import dot
            from numpy.linalg import norm

            scored = []
            for doc, vector in zip(self._fallback_documents, self._fallback_vectors):
                score = float(dot(query_vector, vector) / (norm(query_vector) * norm(vector) + 1e-9))
                scored.append((score, doc))
            scored.sort(key=lambda item: item[0], reverse=True)
            return [
                SearchResult(
                    text=doc["text"],
                    score=score,
                    metadata=doc.get("metadata", {}),
                    method="dense",
                )
                for score, doc in scored[:top_k]
            ]


class _HashEncoder:
    """Small in-memory fallback encoder when sentence-transformer models are unavailable."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _encode_one(self, text: str):
        import numpy as np

        vector = np.zeros(self.dim, dtype=float)
        tokens = re.findall(r"\w+", text.lower())
        if not tokens:
            return vector

        for token in tokens:
            idx = hash(token) % self.dim
            vector[idx] += 1.0

        norm = float((vector ** 2).sum() ** 0.5)
        return vector / (norm + 1e-9)

    def encode(self, texts, show_progress_bar: bool = False):
        import numpy as np

        if isinstance(texts, str):
            return self._encode_one(texts)
        return np.array([self._encode_one(text) for text in texts])


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    rrf_scores: dict[str, dict] = {}
    for result_list in results_list:
        for rank, result in enumerate(result_list):
            if result.text not in rrf_scores:
                rrf_scores[result.text] = {"score": 0.0, "result": result}
            rrf_scores[result.text]["score"] += 1.0 / (k + rank + 1)

    ranked = sorted(rrf_scores.values(), key=lambda item: item["score"], reverse=True)[:top_k]
    return [
        SearchResult(
            text=item["result"].text,
            score=float(item["score"]),
            metadata=item["result"].metadata,
            method="hybrid",
        )
        for item in ranked
    ]


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
