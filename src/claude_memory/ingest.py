"""Bulk and incremental ingestion of Claude Code sessions."""

import asyncio
import hashlib
import json
import logging
import time
from pathlib import Path

from tqdm import tqdm

from .config import DATA_DIR, INGESTION_STATE_FILE
from .extractor import EDU, count_chunks, count_chunks_incremental, extract_edus_from_session, extract_edus_incremental
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


TAIL_BYTES = 128


def file_hash(path: Path) -> str:
    """Signature for append-only JSONL sessions: size + md5 of last TAIL_BYTES.

    Any append changes size; any rewrite changes the tail. Avoids reading the
    full file on every status check.
    """
    size = path.stat().st_size
    with open(path, "rb") as f:
        if size > TAIL_BYTES:
            f.seek(-TAIL_BYTES, 2)
        tail = f.read()
    return f"{size}:{hashlib.md5(tail).hexdigest()}"


def get_pending_sessions(
    projects_dir: Path | None = None, force: bool = False
) -> list[tuple[Path, str, int]]:
    """Find sessions that are new or changed without parsing them.

    Returns (path, file_hash, old_turn_count) tuples. old_turn_count=0 means
    new session (full extraction), >0 means continuation (incremental).
    Callers that need Session objects should parse_session_file() themselves.
    """
    state = load_ingestion_state()
    pending = []
    for path in discover_sessions(projects_dir):
        session_id = path.stem
        h = file_hash(path)
        if not force and session_id in state and state[session_id].get("hash") == h:
            continue
        old_turn_count = state.get(session_id, {}).get("turn_count", 0) if not force else 0
        pending.append((path, h, old_turn_count))
    return pending


async def ingest_session(
    session: Session,
    store: MemoryStore,
    model: str | None = None,
) -> list[EDU]:
    """Extract EDUs from a session and store them."""
    edus = await extract_edus_from_session(session, model)
    if edus:
        added = store.add_edus(edus)
        log.info(f"  Stored {added} EDUs from {session.project}/{session.session_id[:8]}")
    return edus


async def ingest_all(
    projects_dir: Path | None = None,
    model: str | None = None,
    force: bool = False,
    concurrency: int = CONCURRENCY,
) -> dict:
    """Run full ingestion pipeline with concurrent LLM calls."""
    store = MemoryStore()
    state = load_ingestion_state()

    pending_refs = get_pending_sessions(projects_dir, force=force)

    if not pending_refs:
        log.info("No new sessions to ingest.")
        return {"sessions": 0, "edus": 0}

    # Parse sessions now that we're actually ingesting
    pending: list[tuple[Session, int]] = []
    for path, h, old_turn_count in pending_refs:
        session = parse_session_file(path)
        if session:
            session.file_hash = h
            pending.append((session, old_turn_count))

    if not pending:
        log.info("No parseable sessions to ingest.")
        return {"sessions": 0, "edus": 0}

    # Count chunks — incremental sessions only count new turns
    total_chunks = 0
    new_count = 0
    incremental_count = 0
    for session, old_turn_count in pending:
        if old_turn_count > 0:
            total_chunks += count_chunks_incremental(session, old_turn_count)
            incremental_count += 1
        else:
            total_chunks += count_chunks(session)
            new_count += 1

    total_turns = sum(len(s.turns) for s, _ in pending)
    log.info(
        f"Ingesting {len(pending)} sessions ({new_count} new, {incremental_count} continuations, "
        f"{total_turns} turns, {total_chunks} chunks) with concurrency={concurrency}"
    )

    total_edus = 0
    sessions_done = 0
    failed = 0
    t_start = time.time()
    sem = asyncio.Semaphore(concurrency)
    pbar = tqdm(total=total_chunks, unit="chunk", desc="Ingesting")

    def on_chunk_done():
        pbar.set_postfix(edus=total_edus, sessions=f"{sessions_done}/{len(pending)}")
        pbar.update(1)

    async def process(session: Session, old_turn_count: int):
        nonlocal total_edus, sessions_done, failed
        async with sem:
            label = f"{session.project}/{session.session_id[:8]}"
            try:
                if old_turn_count > 0:
                    # Incremental: load existing EDUs, extract only from new turns
                    existing_edus = store.get_session_edus(session.session_id)
                    edus = await extract_edus_incremental(
                        session, existing_edus, old_turn_count,
                        model, on_chunk_done=on_chunk_done,
                    )
                    if edus:
                        store.add_edus(edus)
                    new_edu_count = len(edus)
                    old_edu_count = state.get(session.session_id, {}).get("edu_count", 0)
                    total_edu_count = old_edu_count + new_edu_count
                    partial = True
                else:
                    # Full extraction: delete old EDUs if re-ingesting
                    if session.session_id in state:
                        store.delete_session(session.session_id)
                    edus = await extract_edus_from_session(session, model, on_chunk_done=on_chunk_done)
                    if edus:
                        store.add_edus(edus)
                    total_edu_count = len(edus)
                    new_edu_count = len(edus)
                    partial = False

                total_edus += new_edu_count
                sessions_done += 1

                state[session.session_id] = {
                    "file_path": str(session.file_path),
                    "hash": session.file_hash,
                    "edu_count": total_edu_count,
                    "turn_count": len(session.turns),
                    "project": session.project,
                    "timestamp": session.turns[0].timestamp.isoformat() if session.turns else None,
                    "partial": partial,
                }
                save_ingestion_state(state)
                pbar.set_postfix(edus=total_edus, sessions=f"{sessions_done}/{len(pending)}")
            except Exception as e:
                failed += 1
                log.error(f"Failed {label}: {e}")

    await asyncio.gather(*[process(s, otc) for s, otc in pending])
    pbar.close()

    elapsed = time.time() - t_start
    log.info(f"Done. {total_edus} EDUs from {len(pending) - failed} sessions in {elapsed:.0f}s ({failed} failed)")
    return {"sessions": len(pending) - failed, "edus": total_edus, "failed": failed, "elapsed": elapsed}
