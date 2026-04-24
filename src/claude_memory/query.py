"""Multi-signal retrieval pipeline with recency weighting and project boost."""

import math
from dataclasses import dataclass
from datetime import datetime, timezone

from .config import (
    DEFAULT_MAX_RESULTS,
    PROJECT_BOOST_FACTOR,
    RECENCY_DECAY_ALPHA,
    RETRIEVAL_CANDIDATES,
)
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
    project_boost: float = 1.0


def search(
    store: MemoryStore,
    query: str,
    current_project: str | None = None,
    strict_project: str | None = None,
    max_results: int = DEFAULT_MAX_RESULTS,
    candidates: int = RETRIEVAL_CANDIDATES,
    decay_alpha: float = RECENCY_DECAY_ALPHA,
    boost_factor: float = PROJECT_BOOST_FACTOR,
) -> list[SearchResult]:
    """Search memory with dense similarity + recency weighting + project boost.

    - `current_project`: soft signal — results in this project get score * boost_factor.
    - `strict_project`: hard filter — restrict the vector query to this project only.
      Mutually meaningful with current_project (boost still applies inside the filter
      but is a no-op if current_project == strict_project).
    """
    where = {"project": strict_project} if strict_project else None

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

        project = meta["project"]
        project_boost = boost_factor if (current_project and project == current_project) else 1.0
        score = similarity * recency_weight * project_boost

        results.append(SearchResult(
            text=doc,
            score=score,
            session_id=meta["session_id"],
            project=project,
            timestamp=ts,
            similarity=similarity,
            recency_weight=recency_weight,
            project_boost=project_boost,
        ))

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:max_results]
