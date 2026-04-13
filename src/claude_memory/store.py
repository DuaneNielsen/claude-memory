"""ChromaDB embedding and storage for EDUs."""

import json
import logging

import chromadb
from sentence_transformers import SentenceTransformer

from .config import CHROMADB_DIR, COLLECTION_NAME, EMBEDDING_MODEL
from .extractor import EDU

log = logging.getLogger(__name__)


class MemoryStore:
    def __init__(
        self,
        chromadb_dir: str | None = None,
        embedding_model: str | None = None,
    ):
        self._chromadb_dir = chromadb_dir or str(CHROMADB_DIR)
        self._embedding_model_name = embedding_model or EMBEDDING_MODEL
        self._model: SentenceTransformer | None = None
        self._client: chromadb.ClientAPI | None = None
        self._collection: chromadb.Collection | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self._embedding_model_name)
        return self._model

    @property
    def collection(self) -> chromadb.Collection:
        if self._collection is None:
            self._client = chromadb.PersistentClient(path=self._chromadb_dir)
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

        self.collection.add(
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
        return self.collection.query(**kwargs)

    def delete_session(self, session_id: str) -> int:
        """Delete all EDUs for a session. Returns count deleted."""
        results = self.collection.get(
            where={"session_id": session_id},
            include=[],
        )
        ids = results["ids"]
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)

    def get_session_edus(self, session_id: str) -> list[dict]:
        """Retrieve all stored EDUs for a session. Returns list of {text, metadata}."""
        results = self.collection.get(
            where={"session_id": session_id},
            include=["documents", "metadatas"],
        )
        edus = []
        for doc, meta in zip(results["documents"], results["metadatas"]):
            edus.append({"text": doc, **meta})
        return edus

    def count(self) -> int:
        return self.collection.count()
