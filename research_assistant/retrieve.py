import hashlib
import math
import re

from research_assistant.corpus import chunk_paper
from research_assistant.types import Chunk, Hit, Paper


STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "does",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "which",
        "who",
        "why",
        "with",
    }
)
_TOKEN = re.compile(r"[a-z0-9]+")
RRF_K = 60


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in _TOKEN.findall(text.lower())
        if token not in STOPWORDS and len(token) > 2
    ]


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    k: int = RRF_K,
) -> dict[str, float]:
    """Fuse ranked id lists. Rank 1 contributes 1/(k+1)."""
    if k < 1:
        raise ValueError("RRF k must be at least 1.")
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return scores


class BM25:
    def __init__(self, documents: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.count = len(documents)
        lengths = [len(doc) for doc in documents]
        self.average_length = sum(lengths) / self.count if self.count else 0.0
        self.term_frequency: list[dict[str, int]] = []
        document_frequency: dict[str, int] = {}
        for doc in documents:
            counts: dict[str, int] = {}
            for token in doc:
                counts[token] = counts.get(token, 0) + 1
            self.term_frequency.append(counts)
            for token in counts:
                document_frequency[token] = document_frequency.get(token, 0) + 1
        self.idf = {
            token: math.log(1.0 + (self.count - freq + 0.5) / (freq + 0.5))
            for token, freq in document_frequency.items()
        }

    def scores(self, query: str) -> list[float]:
        tokens = tokenize(query)
        if self.average_length == 0:
            return [0.0] * self.count
        scored = []
        for counts in self.term_frequency:
            length = sum(counts.values())
            total = 0.0
            for token in tokens:
                frequency = counts.get(token, 0)
                if frequency == 0:
                    continue
                denominator = frequency + self.k1 * (
                    1.0 - self.b + self.b * length / self.average_length
                )
                total += self.idf.get(token, 0.0) * (frequency * (self.k1 + 1.0)) / denominator
            scored.append(total)
        return scored


class HashingEmbedder:
    """Sparse token hashing for offline retrieval.

    This is the stand-in used by tests and the local demo. The graded
    experiment should swap in SentenceTransformerEmbedder when that package
    is installed. Shared tokens produce a positive cosine. Unrelated tokens
    do not, aside from a 32-bit hash collision.
    """

    name = "hashing"

    def embed(self, texts: list[str]) -> list[dict[int, float]]:
        return [_hash_vector(text) for text in texts]


class SentenceTransformerEmbedder:
    """Dense embedder. Imported only when embed() is called."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.name = model_name
        self._model = None

    def embed(self, texts: list[str]) -> list[dict[int, float]]:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is not installed. "
                    "The offline demo uses HashingEmbedder. "
                    "Install sentence-transformers to embed with all-MiniLM-L6-v2."
                ) from exc
            self._model = SentenceTransformer(self.model_name)
        matrix = self._model.encode(list(texts), normalize_embeddings=True)
        vectors = []
        for row in matrix:
            vectors.append(
                {
                    index: float(value)
                    for index, value in enumerate(row)
                    if float(value) != 0.0
                }
            )
        return vectors


def _hash_vector(text: str) -> dict[int, float]:
    counts: dict[int, float] = {}
    for token in tokenize(text):
        digest = hashlib.sha1(token.encode("utf-8")).hexdigest()[:8]
        bucket = int(digest, 16)
        counts[bucket] = counts.get(bucket, 0.0) + 1.0
    norm = math.sqrt(sum(value * value for value in counts.values()))
    if norm == 0.0:
        return {}
    return {bucket: value / norm for bucket, value in counts.items()}


def cosine(left: dict[int, float], right: dict[int, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(key, 0.0) for key, value in left.items())


class HybridIndex:
    def __init__(self, chunks: list[Chunk], embedder: HashingEmbedder | SentenceTransformerEmbedder):
        if not chunks:
            raise ValueError("Index needs at least one chunk.")
        self.chunks = list(chunks)
        self.embedder = embedder
        self._by_id = {chunk.chunk_id: chunk for chunk in self.chunks}
        self._bm25 = BM25([tokenize(chunk.text) for chunk in self.chunks])
        self._vectors = embedder.embed([chunk.text for chunk in self.chunks])

    @classmethod
    def build(
        cls,
        papers: list[Paper],
        embedder: HashingEmbedder | SentenceTransformerEmbedder | None = None,
        chunk_words: int = 200,
        overlap: int = 40,
    ) -> "HybridIndex":
        chunks: list[Chunk] = []
        for paper in papers:
            chunks.extend(chunk_paper(paper, chunk_words, overlap))
        if not chunks:
            raise ValueError("The papers produced no chunks.")
        return cls(chunks, embedder or HashingEmbedder())

    def search(
        self,
        query: str,
        mode: str = "hybrid",
        k: int = 5,
        candidates: int = 20,
    ) -> list[Hit]:
        if mode not in {"bm25", "dense", "hybrid"}:
            raise ValueError("mode must be bm25, dense, or hybrid.")
        if k < 1 or candidates < 1:
            raise ValueError("k and candidates must be at least 1.")

        bm25_scores = self._bm25.scores(query)
        bm25_ids = self._ranked_ids(bm25_scores, candidates)
        dense_scores = [
            cosine(self._query_vector(query), vector) for vector in self._vectors
        ]
        dense_ids = self._ranked_ids(dense_scores, candidates)

        if mode == "bm25":
            return self._hits_from_ids(bm25_ids[:k], bm25_scores)
        if mode == "dense":
            return self._hits_from_ids(dense_ids[:k], dense_scores)

        fused = reciprocal_rank_fusion([bm25_ids, dense_ids])
        bm25_by_id = {
            chunk.chunk_id: bm25_scores[index]
            for index, chunk in enumerate(self.chunks)
        }
        # Equal fusion scores are broken by keyword score so a rare term
        # such as "metric" wins over a short abstract that repeats broader words.
        ordered = sorted(
            fused,
            key=lambda item: (-fused[item], -bm25_by_id[item], item),
        )
        return [
            Hit(self._by_id[chunk_id], fused[chunk_id])
            for chunk_id in ordered[:k]
        ]

    def _query_vector(self, query: str) -> dict[int, float]:
        return self.embedder.embed([query])[0]

    def _ranked_ids(self, scores: list[float], candidates: int) -> list[str]:
        order = sorted(
            range(len(scores)),
            key=lambda index: (-scores[index], self.chunks[index].chunk_id),
        )
        ranked = []
        for index in order:
            if scores[index] <= 0:
                break
            ranked.append(self.chunks[index].chunk_id)
            if len(ranked) == candidates:
                break
        return ranked

    def _hits_from_ids(self, chunk_ids: list[str], scores: list[float]) -> list[Hit]:
        score_by_id = {
            chunk.chunk_id: scores[index] for index, chunk in enumerate(self.chunks)
        }
        return [Hit(self._by_id[chunk_id], score_by_id[chunk_id]) for chunk_id in chunk_ids]
