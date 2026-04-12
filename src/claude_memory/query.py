"""Multi-signal retrieval pipeline with recency weighting."""

import math
from dataclasses import dataclass
from datetime import datetime, timezone

from .config import DEFAULT_MAX_RESULTS, RECENCY_DECAY_ALPHA, RETRIEVAL_CANDIDATES
from .store import MemoryStore


@dataclass
class SearchResult:
    text: str
    score: float
    session_id: str
    project: str
    timestamp: datetime
    similarity: float
    recency_weight: float


def search(
    store: MemoryStore,
    query: str,
    project: str | None = None,
    max_results: int = DEFAULT_MAX_RESULTS,
    candidates: int = RETRIEVAL_CANDIDATES,
    decay_alpha: float = RECENCY_DECAY_ALPHA,
) -> list[SearchResult]:
    """Search memory with dense similarity + recency weighting."""
    where = None
    if project:
        where = {"project": project}

    raw = store.query(query, n_results=candidates, where=where)

    if not raw["ids"] or not raw["ids"][0]:
        return []

    now = datetime.now(timezone.utc)
    results = []

    for i, doc_id in enumerate(raw["ids"][0]):
        doc = raw["documents"][0][i]
        meta = raw["metadatas"][0][i]
        distance = raw["distances"][0][i]

        # ChromaDB cosine distance: 0 = identical, 2 = opposite
        # Convert to similarity: 1 - (distance / 2) gives [0, 1]
        similarity = 1.0 - (distance / 2.0)

        ts = datetime.fromisoformat(meta["timestamp"])
        days_ago = (now - ts).total_seconds() / 86400.0
        recency_weight = math.exp(-decay_alpha * days_ago)

        score = similarity * recency_weight

        results.append(SearchResult(
            text=doc,
            score=score,
            session_id=meta["session_id"],
            project=meta["project"],
            timestamp=ts,
            similarity=similarity,
            recency_weight=recency_weight,
        ))

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:max_results]
