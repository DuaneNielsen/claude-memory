from pathlib import Path

# Paths
CLAUDE_PROJECTS_DIR = Path.home() / ".claude" / "projects"
DATA_DIR = Path.home() / ".local" / "share" / "claude-memory"
CHROMADB_DIR = DATA_DIR / "chromadb"
INGESTION_STATE_FILE = DATA_DIR / "ingested_sessions.json"
IN_PROGRESS_DIR = DATA_DIR / "in_progress"
TRAJECTORIES_DB = DATA_DIR / "trajectories.db"
INDICES_DIR = DATA_DIR / "indices"
KEYWORD_FLAGS_LOG = DATA_DIR / "keyword_flags.jsonl"

# Trajectories
NEIGHBOR_WINDOW = 5  # EDUs padded either side of a hit during retrieval
KEYWORD_MERGE_THRESHOLD = 0.9  # embedding similarity for auto-merge
KEYWORD_FLAG_THRESHOLD = 0.75  # below this → accept as new keyword

# Claude model for EDU extraction
DEFAULT_MODEL = "sonnet"

# Claude model for the LLM-curated SessionStart index. The catalog framing
# requires synthesizing across all of a project's trajectories + EDUs, which
# is the kind of wide-context task Opus is meaningfully better at. Falls back
# to the deterministic builder if the call fails (see index_builder.build_index).
INDEX_CURATOR_MODEL = "opus"

# Embedding model
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"

# EDU extraction uses a sliding "core + context margin" scheme:
# each chunk has a CORE region of turns that EDUs are extracted from, bracketed
# by CONTEXT_MARGIN turns on each side that the LLM sees for understanding but
# does NOT extract from. This removes the cliff-edge that caused the LLM to
# hallucinate citations just past the chunk boundary.
CHUNK_CORE_SIZE = 200
CHUNK_CONTEXT_MARGIN = 10
# Kept as legacy aliases (used by old extract_edus_from_session code path).
MAX_TURNS_PER_CHUNK = CHUNK_CORE_SIZE
CHUNK_OVERLAP_TURNS = 5

# Boundary classification (stage 2 of trajectory extraction).
# For each adjacent EDU pair, a classifier sees this many EDUs total as a
# sliding window — default 3 means (previous, candidate, lookahead). Bigger
# windows give more context but more tokens per call.
BOUNDARY_WINDOW_SIZE = 5
BOUNDARY_CONCURRENCY = 10
BOUNDARY_MODEL = "sonnet"

# Retrieval
RETRIEVAL_CANDIDATES = 30
RECENCY_DECAY_ALPHA = 0.007  # per day
DEFAULT_MAX_RESULTS = 10

# Cross-project recall: the current project is a *boost signal* in ranking,
# not a hard filter. Hits whose project matches the caller's current project
# get their score multiplied by this factor; cross-project hits still surface.
# Use strict_project=... at the API/CLI layer when isolation is actually wanted.
PROJECT_BOOST_FACTOR = 1.5

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
