"""Bulk and incremental ingestion of Claude Code sessions (trajectory mode).

Per session:
  1. Snapshot the project keyword cloud (pre-extraction).
  2. Extract trajectories (with EDUs) via the LLM, passing the cloud as context.
  3. Canonicalize new keywords against the cloud; apply mapping to trajectories.
  4. If re-ingesting, delete old trajectories + EDUs for this session.
  5. Persist trajectories to SQLite, EDUs to ChromaDB (with trajectory metadata).
  6. Log flagged borderline keywords for later review.
  7. Update ingestion state.

Concurrent sessions see the cloud as of when they started, so two concurrent
sessions can independently introduce near-dup keywords — canonicalization only
catches this once one of them has persisted. A periodic `canonicalize_project`
cleanup pass (not implemented yet) would reconcile such drift.
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from pathlib import Path

from tqdm import tqdm

from .config import DATA_DIR, IN_PROGRESS_DIR, INGESTION_STATE_FILE, SCHEMA_VERSION
from .extractor import RateLimitError, count_chunks, count_chunks_incremental, extract_trajectories_from_session
from .index_builder import write_index
from .keyword_canonicalizer import append_flags, apply_mapping, canonicalize_keywords
from .parser import Session, discover_sessions, parse_session_file
from .store import MemoryStore
from .trajectories import Trajectory, TrajectoryStore

log = logging.getLogger(__name__)

CONCURRENCY = 2


class SchemaVersionMismatch(RuntimeError):
    """Raised when the on-disk state schema doesn't match the running code."""


def load_ingestion_state() -> dict:
    """Load the record of which sessions have been ingested.

    Raises SchemaVersionMismatch if the file was written by a different
    format version — this guards against old code corrupting new data
    (e.g. a stale MCP server running alongside a newer CLI).
    """
    if not INGESTION_STATE_FILE.exists():
        return {}
    data = json.loads(INGESTION_STATE_FILE.read_text())

    # New format: wrapped with schema_version
    if isinstance(data, dict) and "schema_version" in data:
        v = data["schema_version"]
        if v != SCHEMA_VERSION:
            raise SchemaVersionMismatch(
                f"State file schema version is {v}, code expects {SCHEMA_VERSION}. "
                f"Run `claude-memory reset --all` to clean state, then re-ingest."
            )
        return data.get("sessions", {})

    # Old flat format (dict-of-sessions with no schema_version). We can't
    # distinguish "legitimate old state from 0.2.0" from "empty file" reliably
    # other than this — and either way, old-format data is incompatible.
    if isinstance(data, dict) and data:
        raise SchemaVersionMismatch(
            "State file is in an unversioned (pre-v2) format — likely written by an "
            "older plugin version. Run `claude-memory reset --all` to clean state, "
            "then re-ingest."
        )

    return {}


def save_ingestion_state(state: dict) -> None:
    """Save ingestion state to disk, wrapped with a schema version tag."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    wrapped = {"schema_version": SCHEMA_VERSION, "sessions": state}
    INGESTION_STATE_FILE.write_text(json.dumps(wrapped, indent=2))


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


def _mark_in_progress(session_id: str) -> None:
    """Drop a marker file (named after session_id, contents = pid) so other
    processes can see this session is being ingested right now."""
    IN_PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
    (IN_PROGRESS_DIR / session_id).write_text(str(os.getpid()))


def _unmark_in_progress(session_id: str) -> None:
    (IN_PROGRESS_DIR / session_id).unlink(missing_ok=True)


def get_in_progress_sessions() -> set[str]:
    """Return session IDs whose marker file points to a still-alive pid.
    Cleans up marker files belonging to dead pids as a side effect."""
    if not IN_PROGRESS_DIR.exists():
        return set()
    alive = set()
    for marker in IN_PROGRESS_DIR.iterdir():
        try:
            pid = int(marker.read_text().strip())
        except (ValueError, FileNotFoundError, OSError):
            marker.unlink(missing_ok=True)
            continue
        try:
            os.kill(pid, 0)
            alive.add(marker.name)
        except ProcessLookupError:
            marker.unlink(missing_ok=True)
        except PermissionError:
            # Pid exists but isn't ours — still alive
            alive.add(marker.name)
    return alive


def get_pending_sessions(
    projects_dir: Path | None = None,
    force: bool = False,
    exclude: set[str] | None = None,
) -> list[tuple[Path, str, int]]:
    """Find sessions that are new or changed without parsing them."""
    state = load_ingestion_state()
    pending = []
    exclude = exclude or set()
    for path in discover_sessions(projects_dir):
        session_id = path.stem
        if session_id in exclude:
            continue
        h = file_hash(path)
        if not force and session_id in state and state[session_id].get("hash") == h:
            continue
        old_turn_count = state.get(session_id, {}).get("turn_count", 0) if not force else 0
        pending.append((path, h, old_turn_count))
    return pending


def _canonicalize_and_persist(
    trajectories_with_edus,
    project: str,
    traj_store: TrajectoryStore,
    mem_store: MemoryStore,
):
    """Canonicalize trajectory keywords, then persist to SQLite + ChromaDB.

    Returns (trajectory_count, edu_count).
    """
    if not trajectories_with_edus:
        return 0, 0

    # Snapshot existing cloud BEFORE any writes so canonicalization is deterministic
    existing_cloud = traj_store.get_keywords_for_project(project)

    raw_keywords: set[str] = set()
    for traj, _ in trajectories_with_edus:
        raw_keywords.update(traj.keywords)

    result = canonicalize_keywords(
        raw_keywords, existing_cloud, mem_store.embed,
    )
    if result.auto_merged:
        log.info(
            "Project %s: auto-merged %d keywords: %s",
            project, len(result.auto_merged),
            ", ".join(f"{n}->{c}" for n, c, _ in result.auto_merged),
        )
    if result.flagged:
        append_flags(result.flagged, project)
        log.info(
            "Project %s: flagged %d borderline keywords for review",
            project, len(result.flagged),
        )

    edu_count = 0
    for traj, edus in trajectories_with_edus:
        traj.keywords = apply_mapping(traj.keywords, result.mapping)
        traj_store.add_trajectory(traj)
        if edus:
            mem_store.add_edus(edus)
            edu_count += len(edus)

    return len(trajectories_with_edus), edu_count


async def ingest_all(
    projects_dir: Path | None = None,
    model: str | None = None,
    force: bool = False,
    concurrency: int = CONCURRENCY,
    exclude: set[str] | None = None,
) -> dict:
    """Run full ingestion pipeline with concurrent LLM calls."""
    mem_store = MemoryStore()
    traj_store = TrajectoryStore()
    state = load_ingestion_state()

    pending_refs = get_pending_sessions(projects_dir, force=force, exclude=exclude)

    if not pending_refs:
        log.info("No new sessions to ingest.")
        return {"sessions": 0, "trajectories": 0, "edus": 0}

    # Parse sessions now that we're actually ingesting
    pending: list[tuple[Session, int]] = []
    for path, h, old_turn_count in pending_refs:
        session = parse_session_file(path)
        if session:
            session.file_hash = h
            pending.append((session, old_turn_count))

    if not pending:
        log.info("No parseable sessions to ingest.")
        return {"sessions": 0, "trajectories": 0, "edus": 0}

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
    total_trajectories = 0
    sessions_done = 0
    failed = 0
    skipped_for_retry = 0
    touched_projects: set[str] = set()
    t_start = time.time()
    sem = asyncio.Semaphore(concurrency)
    pbar = tqdm(total=total_chunks, unit="chunk", desc="Ingesting")
    rate_limited = asyncio.Event()

    def on_chunk_done():
        pbar.set_postfix(
            edus=total_edus, trajs=total_trajectories,
            sessions=f"{sessions_done}/{len(pending)}",
        )
        pbar.update(1)

    async def process(session: Session, old_turn_count: int):
        nonlocal total_edus, total_trajectories, sessions_done, failed, skipped_for_retry
        async with sem:
            # If another task already hit the rate limit, don't start new work.
            if rate_limited.is_set():
                skipped_for_retry += 1
                return
            label = f"{session.project}/{session.session_id[:8]}"
            _mark_in_progress(session.session_id)
            try:
                # If re-ingesting (full), clear prior data for this session first
                if old_turn_count == 0 and session.session_id in state:
                    mem_store.delete_session(session.session_id)
                    traj_store.delete_session(session.session_id)

                # Snapshot the project keyword cloud for LLM context
                keyword_cloud = traj_store.get_keywords_for_project(session.project)

                # Existing trajectories for this session (incremental mode only)
                existing_trajectories = (
                    traj_store.get_by_session(session.session_id)
                    if old_turn_count > 0 else []
                )

                results, chunks_failed = await extract_trajectories_from_session(
                    session=session,
                    keyword_cloud=keyword_cloud,
                    existing_trajectories=existing_trajectories,
                    new_turn_start=old_turn_count,
                    model=model,
                    on_chunk_done=on_chunk_done,
                )

                # If any chunks failed (non-rate-limit), don't commit state:
                # leave the session pending so a future ingest picks it up.
                # Data already written to DB for successful chunks will be
                # cleared when the session is retried (old_turn_count==0 path).
                if chunks_failed > 0:
                    log.warning(
                        f"{label}: {chunks_failed} chunks failed — leaving session "
                        f"pending for retry (no state update)"
                    )
                    skipped_for_retry += 1
                    return

                new_traj_count, new_edu_count = _canonicalize_and_persist(
                    results, session.project, traj_store, mem_store,
                )

                total_edus += new_edu_count
                total_trajectories += new_traj_count
                sessions_done += 1
                if new_traj_count:
                    touched_projects.add(session.project)

                old_edu_count = state.get(session.session_id, {}).get("edu_count", 0) if old_turn_count > 0 else 0
                old_traj_count = state.get(session.session_id, {}).get("trajectory_count", 0) if old_turn_count > 0 else 0
                state[session.session_id] = {
                    "file_path": str(session.file_path),
                    "hash": session.file_hash,
                    "edu_count": old_edu_count + new_edu_count,
                    "trajectory_count": old_traj_count + new_traj_count,
                    "turn_count": len(session.turns),
                    "project": session.project,
                    "timestamp": session.turns[0].timestamp.isoformat() if session.turns else None,
                    "partial": old_turn_count > 0,
                }
                save_ingestion_state(state)
                pbar.set_postfix(
                    edus=total_edus, trajs=total_trajectories,
                    sessions=f"{sessions_done}/{len(pending)}",
                )
            except RateLimitError as e:
                # Tell in-flight and queued tasks to stop starting new work
                if not rate_limited.is_set():
                    log.error(f"Rate limit hit on {label}: {e}")
                    log.error("Stopping ingestion — re-run `claude-memory ingest` after the limit resets.")
                rate_limited.set()
                skipped_for_retry += 1
            except Exception as e:
                failed += 1
                log.exception(f"Failed {label}: {e}")
            finally:
                _unmark_in_progress(session.session_id)

    await asyncio.gather(*[process(s, otc) for s, otc in pending])
    pbar.close()

    # Rebuild indices for every project that got new data.
    # Runs serially after all extraction completes so each project sees the
    # final canonical cloud when its index is written.
    indices_written = 0
    for project in sorted(touched_projects):
        try:
            path = await write_index(project, traj_store=traj_store, mem_store=mem_store)
            indices_written += 1
            log.info(f"Rebuilt index for {project} -> {path}")
        except Exception as e:
            log.exception(f"Failed to build index for {project}: {e}")

    elapsed = time.time() - t_start
    completed_sessions = len(pending) - failed - skipped_for_retry
    status = "Rate-limited — stopped for later resume" if rate_limited.is_set() else "Done"
    log.info(
        f"{status}. {total_trajectories} trajectories / {total_edus} EDUs "
        f"from {completed_sessions} sessions in {elapsed:.0f}s "
        f"({failed} failed, {skipped_for_retry} left pending for retry); "
        f"rebuilt {indices_written} project indices"
    )
    if skipped_for_retry:
        log.info("Re-run `claude-memory ingest` to resume the pending sessions.")
    return {
        "sessions": completed_sessions,
        "trajectories": total_trajectories,
        "edus": total_edus,
        "indices": indices_written,
        "failed": failed,
        "pending": skipped_for_retry,
        "elapsed": elapsed,
        "rate_limited": rate_limited.is_set(),
    }
