from pathlib import Path

# Paths
CLAUDE_PROJECTS_DIR = Path.home() / ".claude" / "projects"
DATA_DIR = Path.home() / ".local" / "share" / "claude-memory"
CHROMADB_DIR = DATA_DIR / "chromadb"
INGESTION_STATE_FILE = DATA_DIR / "ingested_sessions.json"

# Claude model for EDU extraction
DEFAULT_MODEL = "haiku"

# Embedding model
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"

# EDU extraction
MAX_TURNS_PER_CHUNK = 30
CHUNK_OVERLAP_TURNS = 5

# Retrieval
RETRIEVAL_CANDIDATES = 30
RECENCY_DECAY_ALPHA = 0.007  # per day
DEFAULT_MAX_RESULTS = 10

# ChromaDB collection name
COLLECTION_NAME = "edus"
