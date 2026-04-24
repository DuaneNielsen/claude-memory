"""Recall retrieval pipeline: hit expansion, re-stitching, wall-of-text render.

Flow:
  1. Find trajectory hits — keyword lookup (SQLite) + vector search on the question (ChromaDB)
  2. Expand each hit: the full trajectory's EDUs + up to N neighbor EDUs on each side
     (the tail of the prior trajectory, the head of the next, within the same session)
  3. Re-stitch overlapping windows into contiguous blocks
  4. Render as a markdown "wall of text" capped at a context budget

The MCP tool returns the wall directly. Synthesis is the calling agent's
responsibility — typically dispatched into an Agent/Task subagent so the wall
never lands in the main context.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

from .config import DATA_DIR, NEIGHBOR_WINDOW, PROJECT_BOOST_FACTOR
from .store import MemoryStore
from .trajectories import Trajectory, TrajectoryStore

log = logging.getLogger(__name__)

# JSONL log of every recall call. One line per call, append-only.
# Used for trend analysis (latency drift, regression detection, A/B comparison
# across version changes). Failures to write are swallowed — never let a
# logging hiccup break the user's recall.
RECALL_LOG_PATH = DATA_DIR / "recall_log.jsonl"

# Wall-of-text cap. The calling agent's Agent-tool subagent has the full 200k
# context window; leave headroom for the question + system prompt + answer.
WALL_TOKEN_BUDGET = 120_000
CHARS_PER_TOKEN = 3.5


@dataclass
class EDURecord:
    """A lightweight EDU representation for retrieval, bundling metadata we need
    for ordering and rendering. Kept separate from the extractor's EDU dataclass
    so retrieval stays decoupled from extraction concerns."""
    edu_id: str
    text: str
    trajectory_id: str
    trajectory_index: int
    session_id: str
    tag: str


@dataclass
class StitchedBlock:
    session_id: str
    trajectories: list[Trajectory]  # trajectories whose EDUs appear in this block, in order
    edus: list[EDURecord]           # merged ordered EDUs


def _find_hit_trajectory_ids(
    search_terms: list[str],
    question: str,
    current_project: str | None,
    strict_project: str | None,
    traj_store: TrajectoryStore,
    mem_store: MemoryStore,
    vector_n: int = 30,
    boost_factor: float = PROJECT_BOOST_FACTOR,
) -> tuple[list[str], dict[str, float]]:
    """Return (trajectory_ids, timings_seconds). Trajectory_ids are ordered by
    boosted score across keyword + vector hits.

    - `current_project`: trajectories in this project get score * boost_factor.
    - `strict_project`: hard filter — only consider trajectories in this project.
      When set, the boost is effectively a no-op (everything is the same project).

    Default (both None): global search, no boost. When `current_project` is set
    without `strict_project`, cross-project hits are returned but ranked below
    same-project hits of comparable similarity.

    Timings dict keys: 'keyword' (SQLite keyword lookup), 'vector' (embed +
    ChromaDB query), 'total'.
    """
    scores: dict[str, float] = {}
    timings: dict[str, float] = {"keyword": 0.0, "vector": 0.0}

    def _bump(tid: str, score: float):
        if tid:
            scores[tid] = max(scores.get(tid, 0.0), score)

    t_start = time.perf_counter()

    if search_terms:
        normalized = [t.strip().lower() for t in search_terms if t and t.strip()]
        if normalized:
            t0 = time.perf_counter()
            if strict_project:
                for tid in traj_store.search_by_keywords(strict_project, normalized):
                    _bump(tid, 1.0)  # keyword hit baseline
            else:
                for tid, project in traj_store.search_by_keywords_global(normalized):
                    boost = boost_factor if project == current_project else 1.0
                    _bump(tid, 1.0 * boost)
            timings["keyword"] = time.perf_counter() - t0

    if question and question.strip():
        t0 = time.perf_counter()
        if strict_project:
            where = {"$and": [
                {"project": {"$eq": strict_project}},
                {"trajectory_id": {"$ne": ""}},
            ]}
        else:
            where = {"trajectory_id": {"$ne": ""}}
        results = mem_store.query(question, n_results=vector_n, where=where)
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        for i, meta in enumerate(metas):
            tid = meta.get("trajectory_id") or ""
            if not tid:
                continue
            similarity = 1.0 - (dists[i] / 2.0)
            project = meta.get("project") or ""
            boost = boost_factor if (current_project and project == current_project) else 1.0
            _bump(tid, similarity * boost)
        timings["vector"] = time.perf_counter() - t0

    timings["total"] = time.perf_counter() - t_start
    ranked = [tid for tid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
    return ranked, timings


def _group_trajectories_by_session(trajectories: list[Trajectory]) -> dict[str, list[Trajectory]]:
    by_sess: dict[str, list[Trajectory]] = {}
    for t in trajectories:
        by_sess.setdefault(t.session_id, []).append(t)
    for sess_id in by_sess:
        by_sess[sess_id].sort(key=lambda t: t.start_turn)
    return by_sess


def _collect_window_trajectory_ids(
    hit_ids: list[str],
    traj_store: TrajectoryStore,
    pad: int,
) -> list[str]:
    """Given hit trajectory_ids, return the expanded set including neighbors (same session)."""
    hits = traj_store.get_many_by_ids(hit_ids)
    sessions_needed = {h.session_id for h in hits}

    # Pull all trajectories for every session the hits belong to (cheap query)
    session_trajs: dict[str, list[Trajectory]] = {}
    for sid in sessions_needed:
        session_trajs[sid] = sorted(
            traj_store.get_by_session(sid),
            key=lambda t: t.start_turn,
        )

    expanded: set[str] = set(hit_ids)
    for hit in hits:
        seq = session_trajs.get(hit.session_id, [])
        idx = next((i for i, t in enumerate(seq) if t.id == hit.id), None)
        if idx is None:
            continue
        if idx > 0:
            expanded.add(seq[idx - 1].id)
        if idx < len(seq) - 1:
            expanded.add(seq[idx + 1].id)
    return sorted(expanded)


def _edus_to_records(raw_edus: list[dict]) -> list[EDURecord]:
    out: list[EDURecord] = []
    for r in raw_edus:
        out.append(EDURecord(
            edu_id=r.get("id", ""),
            text=r["text"],
            trajectory_id=r.get("trajectory_id", "") or "",
            trajectory_index=int(r.get("trajectory_index", 0) or 0),
            session_id=r.get("session_id", "") or "",
            tag=r.get("tag", ""),
        ))
    return out


def _trim_to_window(
    edus: list[EDURecord],
    hit_traj_ids: set[str],
    session_trajs: list[Trajectory],
    pad: int,
) -> list[EDURecord]:
    """Keep: all EDUs from hit trajectories; tail-N of each prev neighbor;
    head-N of each next neighbor. Session-scoped."""
    traj_pos = {t.id: i for i, t in enumerate(session_trajs)}
    hit_positions = {traj_pos[tid] for tid in hit_traj_ids if tid in traj_pos}

    keep_ids: set[str] = set(hit_traj_ids)
    # Mark neighbors
    neighbor_side: dict[str, str] = {}
    for pos in hit_positions:
        if pos > 0:
            prev_id = session_trajs[pos - 1].id
            if prev_id not in hit_traj_ids:
                keep_ids.add(prev_id)
                neighbor_side[prev_id] = "prev"
        if pos < len(session_trajs) - 1:
            next_id = session_trajs[pos + 1].id
            if next_id not in hit_traj_ids:
                keep_ids.add(next_id)
                neighbor_side[next_id] = "next"

    # Group EDUs by trajectory_id and apply pad
    by_traj: dict[str, list[EDURecord]] = {}
    for e in edus:
        if e.trajectory_id in keep_ids:
            by_traj.setdefault(e.trajectory_id, []).append(e)
    for tid in by_traj:
        by_traj[tid].sort(key=lambda e: e.trajectory_index)

    kept: list[EDURecord] = []
    for tid, edus_list in by_traj.items():
        side = neighbor_side.get(tid)
        if side == "prev":
            kept.extend(edus_list[-pad:])
        elif side == "next":
            kept.extend(edus_list[:pad])
        else:
            kept.extend(edus_list)
    return kept


def _stitch_session_edus(
    session_id: str,
    edus: list[EDURecord],
    session_trajs: list[Trajectory],
) -> StitchedBlock:
    """Order EDUs in session by trajectory position then by trajectory_index, wrap with block metadata."""
    pos = {t.id: t.start_turn for t in session_trajs}
    edus.sort(key=lambda e: (pos.get(e.trajectory_id, 0), e.trajectory_index))
    # Collect unique trajectories that actually appear, in order
    traj_by_id = {t.id: t for t in session_trajs}
    ordered_traj_ids: list[str] = []
    seen: set[str] = set()
    for e in edus:
        if e.trajectory_id not in seen and e.trajectory_id in traj_by_id:
            seen.add(e.trajectory_id)
            ordered_traj_ids.append(e.trajectory_id)
    return StitchedBlock(
        session_id=session_id,
        trajectories=[traj_by_id[tid] for tid in ordered_traj_ids],
        edus=edus,
    )


def gather_windows(
    hit_trajectory_ids: list[str],
    traj_store: TrajectoryStore,
    mem_store: MemoryStore,
    pad: int = NEIGHBOR_WINDOW,
) -> list[StitchedBlock]:
    """Given a list of hit trajectory_ids, return restitched per-session blocks."""
    if not hit_trajectory_ids:
        return []

    expanded_ids = _collect_window_trajectory_ids(hit_trajectory_ids, traj_store, pad)
    raw_edus = mem_store.get_edus_by_trajectories(expanded_ids)
    edus = _edus_to_records(raw_edus)

    # Group by session
    by_sess_edus: dict[str, list[EDURecord]] = {}
    for e in edus:
        by_sess_edus.setdefault(e.session_id, []).append(e)

    hit_set = set(hit_trajectory_ids)
    blocks: list[StitchedBlock] = []
    for sess_id, sess_edus in by_sess_edus.items():
        session_trajs = sorted(
            traj_store.get_by_session(sess_id),
            key=lambda t: t.start_turn,
        )
        trimmed = _trim_to_window(sess_edus, hit_set, session_trajs, pad)
        block = _stitch_session_edus(sess_id, trimmed, session_trajs)
        if block.edus:
            blocks.append(block)

    # Order blocks by the most recent trajectory's timestamp (freshest first)
    def block_sort_key(b: StitchedBlock):
        if b.trajectories:
            return max(t.created_at for t in b.trajectories)
        return None

    blocks.sort(key=block_sort_key, reverse=True)
    return blocks


def render_wall(
    blocks: list[StitchedBlock],
    budget_tokens: int = WALL_TOKEN_BUDGET,
) -> tuple[str, int]:
    """Render blocks as markdown. Returns (text, blocks_included)."""
    parts: list[str] = []
    blocks_included = 0
    running_chars = 0
    char_budget = int(budget_tokens * CHARS_PER_TOKEN)

    for i, block in enumerate(blocks):
        block_parts: list[str] = []
        if block.trajectories:
            dates = sorted({t.created_at.strftime("%Y-%m-%d") for t in block.trajectories})
            date_str = dates[0] if len(dates) == 1 else f"{dates[0]}—{dates[-1]}"
            block_parts.append(f"### Block {i+1} — {date_str} (session {block.session_id[:8]})")
            for t in block.trajectories:
                kw_str = ", ".join(t.keywords) if t.keywords else ""
                block_parts.append(f"- [trajectory] {t.summary}  (keywords: {kw_str})")
            block_parts.append("")
        for e in block.edus:
            block_parts.append(f"- [{e.tag}] {e.text}")
        block_parts.append("")

        block_text = "\n".join(block_parts)
        if running_chars + len(block_text) > char_budget and blocks_included > 0:
            parts.append(
                f"\n*(truncated: {len(blocks) - blocks_included} more block(s) omitted for space)*"
            )
            break
        parts.append(block_text)
        running_chars += len(block_text)
        blocks_included += 1

    return "\n".join(parts).strip(), blocks_included


@dataclass
class RecallContext:
    """Diagnostics for a build_recall_wall() call. The wall itself is returned
    alongside this — these are just the counts and phase timings useful for
    logging and the tool's footer line."""
    hit_count: int
    block_count: int
    blocks_in_wall: int
    wall_chars: int
    # Phase timings in seconds. `find_hits` splits into keyword + vector
    # contributions in `find_hits_breakdown`. Synthesis is no longer ours to
    # measure — the calling agent's Agent-tool dispatch owns that cost.
    t_find_hits: float = 0.0
    t_gather: float = 0.0
    t_render: float = 0.0
    t_total: float = 0.0
    find_hits_breakdown: dict = field(default_factory=dict)


def _package_version() -> str:
    try:
        from importlib.metadata import version
        return version("claude-memory")
    except Exception:
        return "unknown"


def _append_recall_log(
    *,
    ctx: RecallContext,
    question: str,
    search_terms: list[str],
    current_project: str | None,
    strict_project: str | None,
    mem_store: MemoryStore,
) -> None:
    """Append one JSON line per recall to RECALL_LOG_PATH. Best-effort —
    log failures must not break recall."""
    try:
        from datetime import datetime, timezone
        try:
            corpus_size = mem_store.count()
        except Exception:
            corpus_size = -1
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "version": _package_version(),
            "question": question[:500],
            "search_terms": search_terms,
            "current_project": current_project,
            "strict_project": strict_project,
            "corpus_size": corpus_size,
            "hits": ctx.hit_count,
            "blocks_total": ctx.block_count,
            "blocks_in_wall": ctx.blocks_in_wall,
            "wall_chars": ctx.wall_chars,
            "t_total": round(ctx.t_total, 4),
            "t_find_hits": round(ctx.t_find_hits, 4),
            "t_find_keyword": round(ctx.find_hits_breakdown.get("keyword", 0.0), 4),
            "t_find_vector": round(ctx.find_hits_breakdown.get("vector", 0.0), 4),
            "t_gather": round(ctx.t_gather, 4),
            "t_render": round(ctx.t_render, 4),
        }
        RECALL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with RECALL_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        log.debug("recall log write failed: %s", e)


def build_recall_wall(
    search_terms: list[str],
    question: str,
    current_project: str | None = None,
    strict_project: str | None = None,
    traj_store: TrajectoryStore | None = None,
    mem_store: MemoryStore | None = None,
    pad: int = NEIGHBOR_WINDOW,
    budget_tokens: int = WALL_TOKEN_BUDGET,
) -> tuple[str, RecallContext]:
    """Find hits, gather neighbor-padded blocks, render the wall-of-text.

    Returns (wall, ctx). The wall is markdown — empty string if nothing matched
    or no EDUs could be retrieved. `ctx` carries diagnostic counts + phase
    timings for logging.

    Recall is global by default. `current_project` is a soft boost — trajectories
    in that project rank higher than cross-project hits of comparable similarity.
    `strict_project` is a hard filter — restricts retrieval to that project only.

    Synthesis is *not* this function's job. The calling agent should dispatch an
    Agent/Task subagent with the wall + question and let it produce prose.
    """
    traj_store = traj_store or TrajectoryStore()
    mem_store = mem_store or MemoryStore()

    t_total_start = time.perf_counter()

    t0 = time.perf_counter()
    hit_ids, find_breakdown = _find_hit_trajectory_ids(
        search_terms, question,
        current_project=current_project,
        strict_project=strict_project,
        traj_store=traj_store,
        mem_store=mem_store,
    )
    t_find = time.perf_counter() - t0

    if not hit_ids:
        ctx = RecallContext(
            hit_count=0, block_count=0, blocks_in_wall=0, wall_chars=0,
            t_find_hits=t_find, t_total=time.perf_counter() - t_total_start,
            find_hits_breakdown=find_breakdown,
        )
        _append_recall_log(
            ctx=ctx, question=question, search_terms=search_terms,
            current_project=current_project, strict_project=strict_project,
            mem_store=mem_store,
        )
        return "", ctx

    t0 = time.perf_counter()
    blocks = gather_windows(hit_ids, traj_store, mem_store, pad=pad)
    t_gather = time.perf_counter() - t0

    t0 = time.perf_counter()
    wall, blocks_included = render_wall(blocks, budget_tokens=budget_tokens)
    t_render = time.perf_counter() - t0

    ctx = RecallContext(
        hit_count=len(hit_ids),
        block_count=len(blocks),
        blocks_in_wall=blocks_included,
        wall_chars=len(wall),
        t_find_hits=t_find,
        t_gather=t_gather,
        t_render=t_render,
        t_total=time.perf_counter() - t_total_start,
        find_hits_breakdown=find_breakdown,
    )
    _append_recall_log(
        ctx=ctx,
        question=question,
        search_terms=search_terms,
        current_project=current_project,
        strict_project=strict_project,
        mem_store=mem_store,
    )
    return wall, ctx
