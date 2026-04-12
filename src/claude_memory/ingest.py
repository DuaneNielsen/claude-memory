"""Bulk and incremental ingestion of Claude Code sessions."""

import asyncio
import hashlib
import json
import logging
import time
from pathlib import Path

from tqdm import tqdm

from .config import DATA_DIR, INGESTION_STATE_FILE
from .extractor import EDU, count_chunks, extract_edus_from_session
from .parser import Session, discover_sessions, parse_session_file
from .store import MemoryStore

log = logging.getLogger(__name__)

CONCURRENCY = 2


def load_ingestion_state() -> dict:
    """Load the record of which sessions have been ingested."""
    if INGESTION_STATE_FILE.exists():
        return json.loads(INGESTION_STATE_FILE.read_text())
    return {}


def save_ingestion_state(state: dict) -> None:
    """Save ingestion state to disk."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    INGESTION_STATE_FILE.write_text(json.dumps(state, indent=2))


def file_hash(path: Path) -> str:
    """Fast hash of file contents."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def get_pending_sessions(projects_dir: Path | None = None, force: bool = False) -> list[Session]:
    """Find and parse sessions that are new or changed. Hash computed once here."""
    state = load_ingestion_state()
    pending = []
    for path in discover_sessions(projects_dir):
        session_id = path.stem
        h = file_hash(path)
        if not force and session_id in state and state[session_id].get("hash") == h:
            continue
        session = parse_session_file(path)
        if session:
            session.file_hash = h
            pending.append(session)
    return pending


async def ingest_session(
    session: Session,
    store: MemoryStore,
    base_url: str | None = None,
) -> list[EDU]:
    """Extract EDUs from a session and store them."""
    edus = await extract_edus_from_session(session, base_url)
    if edus:
        added = store.add_edus(edus)
        log.info(f"  Stored {added} EDUs from {session.project}/{session.session_id[:8]}")
    return edus


async def ingest_all(
    projects_dir: Path | None = None,
    base_url: str | None = None,
    force: bool = False,
    concurrency: int = CONCURRENCY,
) -> dict:
    """Run full ingestion pipeline with concurrent LLM calls."""
    store = MemoryStore()
    state = load_ingestion_state()

    sessions = get_pending_sessions(projects_dir, force=force)

    if not sessions:
        log.info("No new sessions to ingest.")
        return {"sessions": 0, "edus": 0}

    total_chunks = sum(count_chunks(s) for s in sessions)
    total_turns = sum(len(s.turns) for s in sessions)
    log.info(f"Ingesting {len(sessions)} sessions ({total_turns} turns, {total_chunks} chunks) with concurrency={concurrency}")

    total_edus = 0
    sessions_done = 0
    failed = 0
    t_start = time.time()
    sem = asyncio.Semaphore(concurrency)
    pbar = tqdm(total=total_chunks, unit="chunk", desc="Ingesting")

    def on_chunk_done():
        pbar.set_postfix(edus=total_edus, sessions=f"{sessions_done}/{len(sessions)}")
        pbar.update(1)

    async def process(session: Session):
        nonlocal total_edus, sessions_done, failed
        async with sem:
            label = f"{session.project}/{session.session_id[:8]}"
            try:
                # Delete old EDUs if re-ingesting a changed session
                if session.session_id in state:
                    store.delete_session(session.session_id)

                edus = await extract_edus_from_session(session, base_url, on_chunk_done=on_chunk_done)
                if edus:
                    store.add_edus(edus)
                total_edus += len(edus)
                sessions_done += 1

                state[session.session_id] = {
                    "file_path": str(session.file_path),
                    "hash": session.file_hash,
                    "edu_count": len(edus),
                    "project": session.project,
                    "timestamp": session.turns[0].timestamp.isoformat() if session.turns else None,
                }
                save_ingestion_state(state)
                pbar.set_postfix(edus=total_edus, sessions=f"{sessions_done}/{len(sessions)}")
            except Exception as e:
                failed += 1
                log.error(f"Failed {label}: {e}")

    await asyncio.gather(*[process(s) for s in sessions])
    pbar.close()

    elapsed = time.time() - t_start
    log.info(f"Done. {total_edus} EDUs from {len(sessions) - failed} sessions in {elapsed:.0f}s ({failed} failed)")
    return {"sessions": len(sessions) - failed, "edus": total_edus, "failed": failed, "elapsed": elapsed}
