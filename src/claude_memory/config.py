from pathlib import Path

# Paths
CLAUDE_PROJECTS_DIR = Path.home() / ".claude" / "projects"
DATA_DIR = Path.home() / ".local" / "share" / "claude-memory"
CHROMADB_DIR = DATA_DIR / "chromadb"
INGESTION_STATE_FILE = DATA_DIR / "ingested_sessions.json"
TRAJECTORIES_DB = DATA_DIR / "trajectories.db"
INDICES_DIR = DATA_DIR / "indices"
KEYWORD_FLAGS_LOG = DATA_DIR / "keyword_flags.jsonl"

# Trajectories
NEIGHBOR_WINDOW = 5  # EDUs padded either side of a hit during retrieval
KEYWORD_MERGE_THRESHOLD = 0.9  # embedding similarity for auto-merge
KEYWORD_FLAG_THRESHOLD = 0.75  # below this → accept as new keyword

# Claude model for EDU extraction
DEFAULT_MODEL = "sonnet"

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

# Data-format schema version. Bumped when state / trajectory / EDU formats
# change in a way that makes old code writing to this store produce corrupt
# data. If a store's stored version doesn't match this, new code refuses to
# read and asks the user to reset.
#   1 = pre-trajectory EDU-only pipeline (plugin ≤ 0.2.0)
#   2 = trajectory pipeline with topic decomposition
SCHEMA_VERSION = 2

# ChromaDB server (client-server mode — required for concurrent access
# from the MCP server and ingest CLI; PersistentClient segfaults when
# two processes open the same DB)
CHROMA_HOST = "127.0.0.1"
CHROMA_PORT = 8765
