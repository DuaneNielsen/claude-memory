"""Deep-recall retrieval pipeline: hit expansion, re-stitching, subagent synthesis.

Flow:
  1. Find trajectory hits — keyword lookup (SQLite) + vector search on the question (ChromaDB)
  2. Expand each hit: the full trajectory's EDUs + up to N neighbor EDUs on each side
     (the tail of the prior trajectory, the head of the next, within the same session)
  3. Re-stitch overlapping windows into contiguous blocks
  4. Render as a markdown "wall of text" capped at a context budget
  5. Call an Opus subagent with (question, wall-of-text) → synthesized answer

The caller (MCP tool) just receives the synthesized answer — the main agent's
context stays clean.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import DATA_DIR, NEIGHBOR_WINDOW, PROJECT_BOOST_FACTOR
from .store import MemoryStore
from .trajectories import Trajectory, TrajectoryStore

log = logging.getLogger(__name__)

# JSONL log of every recall_memory call. One line per call, append-only.
# Used for trend analysis (latency drift, regression detection, A/B comparison
# across model/version changes). Failures to write are swallowed — never let
# a logging hiccup break the user's recall.
RECALL_LOG_PATH = DATA_DIR / "recall_log.jsonl"

SUBAGENT_TOKEN_BUDGET = 120_000  # wall-of-text cap (Opus has 200k; leave headroom)
CHARS_PER_TOKEN = 3.5

SUBAGENT_MODEL = "opus"

SUBAGENT_SYSTEM_PROMPT = """\
You are a memory-retrieval assistant. The primary Claude agent has asked you to answer a question using excerpts from past conversations. Your job:

1. Read the provided memory excerpts carefully. They are organized into "blocks" — each block is a contiguous stretch of conversation trajectories that were extracted earlier.
2. Answer the question using ONLY information found in the excerpts. If the answer isn't there, say so plainly.
3. Be concise (target 150-400 words). Cite specific trajectories or dates when relevant (e.g. "On 2026-04-21, ...").
4. If excerpts contradict each other, note the discrepancy and prefer the most recent one.
5. If the question is vague or unanswerable from these excerpts, briefly explain what IS in the excerpts that's adjacent to the question.

Output plain prose, no markdown headers. The primary agent will quote or paraphrase your answer to the user."""


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
    budget_tokens: int = SUBAGENT_TOKEN_BUDGET,
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


async def call_subagent(
    question: str,
    wall: str,
    model: str = SUBAGENT_MODEL,
) -> tuple[str, dict[str, float]]:
    """Spawn `claude -p --model <model>` and return (answer, timings_seconds).

    Uses `--output-format stream-json` so we can timestamp the first event the
    subagent emits, separating CLI cold-start + first-token latency from the
    rest of generation. Timings dict keys: 'spawn' (process fork/exec),
    'first_event' (start → first stdout line; ~ CLI startup + first token),
    'complete' (start → process exit).
    """
    prompt = f"""## Question

{question}

## Memory Excerpts

{wall}
"""
    # Wall-clock milestones inside the subagent process. Together they carve
    # t_subagent into: boot (until system:init — CLI startup + SessionStart
    # hooks), api_request (init → first stream_event = model started
    # responding), gen (first stream_event → result event), cleanup (result →
    # exit). `api_ttft_ms` is the API's own self-reported TTFT, surfaced in
    # the message_start event.
    timings: dict[str, float] = {
        "spawn": 0.0,
        "first_event": 0.0,
        "system_init": 0.0,
        "first_stream_event": 0.0,
        "first_content_delta": 0.0,
        "result_event": 0.0,
        "complete": 0.0,
        "api_ttft_ms": 0.0,  # from API's message_start event, not wall clock
    }

    t_start = time.perf_counter()
    proc = await asyncio.create_subprocess_exec(
        "claude", "-p",
        "--model", model,
        "--tools", "",
        "--system-prompt", SUBAGENT_SYSTEM_PROMPT,
        "--output-format", "stream-json",
        "--verbose",  # required by claude CLI when stream-json output is used
        "--include-partial-messages",  # so first_assistant_event ≈ true TTFT
        "--no-session-persistence",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    timings["spawn"] = time.perf_counter() - t_start

    proc.stdin.write(prompt.encode())
    await proc.stdin.drain()
    proc.stdin.close()

    events: list[dict] = []
    first_event_at: float | None = None
    seen: set[str] = set()
    while True:
        line = await proc.stdout.readline()
        if not line:
            break
        now = time.perf_counter() - t_start
        if first_event_at is None:
            first_event_at = now
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            log.debug("subagent emitted non-JSON line: %r", line[:200])
            continue
        events.append(event)
        if not isinstance(event, dict):
            continue
        etype = event.get("type")
        if etype == "system" and event.get("subtype") == "init" and "system_init" not in seen:
            timings["system_init"] = now
            seen.add("system_init")
        elif etype == "stream_event":
            if "first_stream_event" not in seen:
                timings["first_stream_event"] = now
                seen.add("first_stream_event")
            inner = event.get("event") or {}
            if isinstance(inner, dict):
                # API self-reports TTFT on the message_start event — capture once.
                if inner.get("type") == "message_start" and "api_ttft" not in seen:
                    ttft = event.get("ttft_ms")
                    if isinstance(ttft, (int, float)):
                        timings["api_ttft_ms"] = float(ttft)
                        seen.add("api_ttft")
                if inner.get("type") == "content_block_delta" and "first_content_delta" not in seen:
                    timings["first_content_delta"] = now
                    seen.add("first_content_delta")
        elif etype == "result" and "result_event" not in seen:
            timings["result_event"] = now
            seen.add("result_event")

    rc = await proc.wait()
    timings["complete"] = time.perf_counter() - t_start
    if first_event_at is not None:
        timings["first_event"] = first_event_at

    if rc != 0:
        stderr = (await proc.stderr.read()).decode()
        raise RuntimeError(f"subagent failed (exit {rc}): {stderr}")

    # Find the result event (final one, type='result') — falling back to any
    # event with a 'result' field for forward-compatibility with format tweaks.
    for event in reversed(events):
        if isinstance(event, dict) and event.get("type") == "result":
            return event.get("result") or "", timings
    for event in reversed(events):
        if isinstance(event, dict) and "result" in event:
            return event["result"] or "", timings
    raise ValueError(
        f"Could not extract result from subagent stream "
        f"({len(events)} events, types: {[e.get('type') for e in events[:5]]}...)"
    )


@dataclass
class RecallResult:
    answer: str
    hit_count: int
    block_count: int
    blocks_in_wall: int
    wall_chars: int
    # Phase timings in seconds. `find_hits` further splits into keyword + vector
    # contributions in `find_hits_breakdown`; `subagent` further splits into
    # spawn / first_event / complete in `subagent_breakdown`. The split lets us
    # tell how much of subagent cost is CLI cold-start (replaceable by an
    # in-process SDK call) vs actual inference.
    t_find_hits: float = 0.0
    t_gather: float = 0.0
    t_render: float = 0.0
    t_subagent: float = 0.0
    t_total: float = 0.0
    find_hits_breakdown: dict = field(default_factory=dict)
    subagent_breakdown: dict = field(default_factory=dict)


def _package_version() -> str:
    try:
        from importlib.metadata import version
        return version("claude-memory")
    except Exception:
        return "unknown"


def _append_recall_log(
    *,
    result: RecallResult,
    question: str,
    search_terms: list[str],
    current_project: str | None,
    strict_project: str | None,
    model: str,
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
            "model": model,
            "question": question[:500],
            "search_terms": search_terms,
            "current_project": current_project,
            "strict_project": strict_project,
            "corpus_size": corpus_size,
            "hits": result.hit_count,
            "blocks_total": result.block_count,
            "blocks_in_wall": result.blocks_in_wall,
            "wall_chars": result.wall_chars,
            "t_total": round(result.t_total, 4),
            "t_find_hits": round(result.t_find_hits, 4),
            "t_find_keyword": round(result.find_hits_breakdown.get("keyword", 0.0), 4),
            "t_find_vector": round(result.find_hits_breakdown.get("vector", 0.0), 4),
            "t_gather": round(result.t_gather, 4),
            "t_render": round(result.t_render, 4),
            "t_subagent": round(result.t_subagent, 4),
            "t_subagent_spawn": round(result.subagent_breakdown.get("spawn", 0.0), 4),
            "t_subagent_first_event": round(result.subagent_breakdown.get("first_event", 0.0), 4),
            "t_subagent_system_init": round(result.subagent_breakdown.get("system_init", 0.0), 4),
            "t_subagent_first_stream_event": round(result.subagent_breakdown.get("first_stream_event", 0.0), 4),
            "t_subagent_first_content_delta": round(result.subagent_breakdown.get("first_content_delta", 0.0), 4),
            "t_subagent_result_event": round(result.subagent_breakdown.get("result_event", 0.0), 4),
            "t_subagent_complete": round(result.subagent_breakdown.get("complete", 0.0), 4),
            "api_ttft_ms": round(result.subagent_breakdown.get("api_ttft_ms", 0.0), 1),
        }
        RECALL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with RECALL_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        log.debug("recall log write failed: %s", e)


async def recall_memory(
    search_terms: list[str],
    question: str,
    current_project: str | None = None,
    strict_project: str | None = None,
    traj_store: TrajectoryStore | None = None,
    mem_store: MemoryStore | None = None,
    pad: int = NEIGHBOR_WINDOW,
    budget_tokens: int = SUBAGENT_TOKEN_BUDGET,
    model: str = SUBAGENT_MODEL,
) -> RecallResult:
    """End-to-end deep-recall. Returns the subagent's synthesized answer + diagnostics.

    Recall is global by default. `current_project` is a soft boost — trajectories
    in that project rank higher than cross-project hits of comparable similarity.
    `strict_project` is a hard filter — restricts retrieval to that project only.
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
        scope = f"project '{strict_project}'" if strict_project else "any project"
        return RecallResult(
            answer=f"No memories matched the search terms or question across {scope}.",
            hit_count=0, block_count=0, blocks_in_wall=0, wall_chars=0,
            t_find_hits=t_find, t_total=time.perf_counter() - t_total_start,
            find_hits_breakdown=find_breakdown,
        )

    t0 = time.perf_counter()
    blocks = gather_windows(hit_ids, traj_store, mem_store, pad=pad)
    t_gather = time.perf_counter() - t0

    t0 = time.perf_counter()
    wall, blocks_included = render_wall(blocks, budget_tokens=budget_tokens)
    t_render = time.perf_counter() - t0

    if not wall.strip():
        return RecallResult(
            answer="Hits were found but their EDUs could not be retrieved from the memory store.",
            hit_count=len(hit_ids), block_count=len(blocks), blocks_in_wall=0, wall_chars=0,
            t_find_hits=t_find, t_gather=t_gather, t_render=t_render,
            t_total=time.perf_counter() - t_total_start,
            find_hits_breakdown=find_breakdown,
        )

    t0 = time.perf_counter()
    answer, subagent_breakdown = await call_subagent(question, wall, model=model)
    t_subagent = time.perf_counter() - t0

    result = RecallResult(
        answer=answer,
        hit_count=len(hit_ids),
        block_count=len(blocks),
        blocks_in_wall=blocks_included,
        wall_chars=len(wall),
        t_find_hits=t_find,
        t_gather=t_gather,
        t_render=t_render,
        t_subagent=t_subagent,
        t_total=time.perf_counter() - t_total_start,
        find_hits_breakdown=find_breakdown,
        subagent_breakdown=subagent_breakdown,
    )
    _append_recall_log(
        result=result,
        question=question,
        search_terms=search_terms,
        current_project=current_project,
        strict_project=strict_project,
        model=model,
        mem_store=mem_store,
    )
    return result
