"""ChromaDB embedding and storage for EDUs."""

import json
import logging
import time

import chromadb
from sentence_transformers import SentenceTransformer

from .config import CHROMA_HOST, CHROMA_PORT, COLLECTION_NAME, EMBEDDING_MODEL
from .extractor import EDU

log = logging.getLogger(__name__)


def _retry_on_lock(fn, *args, attempts: int = 5, base_delay: float = 0.05, **kwargs):
    """Retry a ChromaDB call on SQLite 'database is locked' errors.

    Concurrent reads from the MCP server and writes from the ingest subprocess
    can briefly contend on the underlying SQLite file. WAL mode makes this rare,
    but we still retry with exponential backoff to avoid surfacing transient
    lock errors to the caller.
    """
    delay = base_delay
    for i in range(attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            msg = str(e).lower()
            if "database is locked" in msg or "database table is locked" in msg:
                if i == attempts - 1:
                    raise
                log.debug("ChromaDB locked, retrying in %.3fs (attempt %d)", delay, i + 1)
                time.sleep(delay)
                delay *= 2
                continue
            raise


class MemoryStore:
    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        embedding_model: str | None = None,
    ):
        self._host = host or CHROMA_HOST
        self._port = port or CHROMA_PORT
        self._embedding_model_name = embedding_model or EMBEDDING_MODEL
        self._model: SentenceTransformer | None = None
        self._client: chromadb.ClientAPI | None = None
        self._collection: chromadb.Collection | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self._embedding_model_name, device="cpu")
        return self._model

    @property
    def collection(self) -> chromadb.Collection:
        if self._collection is None:
            self._client = chromadb.HttpClient(host=self._host, port=self._port)
            self._collection = self._client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts."""
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def add_edus(self, edus: list[EDU]) -> int:
        """Embed and store EDUs. Returns count added."""
        if not edus:
            return 0

        texts = [e.text for e in edus]
        embeddings = self.embed(texts)

        _retry_on_lock(
            self.collection.add,
            ids=[e.edu_id for e in edus],
            documents=texts,
            embeddings=embeddings,
            metadatas=[{
                "session_id": e.session_id,
                "project": e.project,
                "timestamp": e.timestamp.isoformat(),
                "speakers": ",".join(e.speakers),
                "source_turns": json.dumps(e.source_turn_ids),
            } for e in edus],
        )
        return len(edus)

    def query(
        self,
        query_text: str,
        n_results: int = 30,
        where: dict | None = None,
    ) -> dict:
        """Query the collection by text similarity."""
        embedding = self.embed([query_text])[0]
        kwargs = {
            "query_embeddings": [embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        return _retry_on_lock(self.collection.query, **kwargs)

    def delete_session(self, session_id: str) -> int:
        """Delete all EDUs for a session. Returns count deleted."""
        results = _retry_on_lock(
            self.collection.get,
            where={"session_id": session_id},
            include=[],
        )
        ids = results["ids"]
        if ids:
            _retry_on_lock(self.collection.delete, ids=ids)
        return len(ids)

    def get_session_edus(self, session_id: str) -> list[dict]:
        """Retrieve all stored EDUs for a session. Returns list of {text, metadata}."""
        results = _retry_on_lock(
            self.collection.get,
            where={"session_id": session_id},
            include=["documents", "metadatas"],
        )
        edus = []
        for doc, meta in zip(results["documents"], results["metadatas"]):
            edus.append({"text": doc, **meta})
        return edus

    def count(self) -> int:
        return _retry_on_lock(self.collection.count)
