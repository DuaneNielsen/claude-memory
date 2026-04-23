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
from dataclasses import dataclass, field
from pathlib import Path

from .config import NEIGHBOR_WINDOW
from .store import MemoryStore
from .trajectories import Trajectory, TrajectoryStore

log = logging.getLogger(__name__)

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
    project: str,
    traj_store: TrajectoryStore,
    mem_store: MemoryStore,
    vector_n: int = 30,
) -> list[str]:
    """Return the union of keyword-hit and vector-hit trajectory_ids."""
    hit_ids: set[str] = set()

    if search_terms:
        normalized = [t.strip().lower() for t in search_terms if t and t.strip()]
        if normalized:
            hit_ids.update(traj_store.search_by_keywords(project, normalized))

    if question and question.strip():
        where = {"$and": [
            {"project": {"$eq": project}},
            {"trajectory_id": {"$ne": ""}},
        ]}
        results = mem_store.query(question, n_results=vector_n, where=where)
        for meta in results.get("metadatas", [[]])[0]:
            tid = meta.get("trajectory_id") or ""
            if tid:
                hit_ids.add(tid)

    return sorted(hit_ids)


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
) -> str:
    """Spawn `claude -p --model <model>` with the subagent system prompt and return its text answer."""
    prompt = f"""## Question

{question}

## Memory Excerpts

{wall}
"""
    proc = await asyncio.create_subprocess_exec(
        "claude", "-p",
        "--model", model,
        "--tools", "",
        "--system-prompt", SUBAGENT_SYSTEM_PROMPT,
        "--output-format", "json",
        "--no-session-persistence",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate(prompt.encode())
    if proc.returncode != 0:
        raise RuntimeError(f"subagent failed (exit {proc.returncode}): {stderr.decode()}")

    content = stdout.decode().strip()
    if not content:
        raise ValueError("subagent returned empty output")

    # `claude -p --output-format json` streams events; the final one has the assistant response
    events = json.loads(content)
    if isinstance(events, list):
        for event in reversed(events):
            if isinstance(event, dict) and event.get("type") == "result":
                return event.get("result") or ""
            if isinstance(event, dict) and "result" in event:
                return event["result"] or ""
    # Fallback: if we got a single dict with a text field
    if isinstance(events, dict) and "result" in events:
        return events["result"] or ""
    raise ValueError(f"Could not extract result from subagent response")


@dataclass
class RecallResult:
    answer: str
    hit_count: int
    block_count: int
    blocks_in_wall: int
    wall_chars: int


async def recall_memory(
    search_terms: list[str],
    question: str,
    project: str,
    traj_store: TrajectoryStore | None = None,
    mem_store: MemoryStore | None = None,
    pad: int = NEIGHBOR_WINDOW,
    budget_tokens: int = SUBAGENT_TOKEN_BUDGET,
    model: str = SUBAGENT_MODEL,
) -> RecallResult:
    """End-to-end deep-recall. Returns the subagent's synthesized answer + diagnostics."""
    traj_store = traj_store or TrajectoryStore()
    mem_store = mem_store or MemoryStore()

    hit_ids = _find_hit_trajectory_ids(
        search_terms, question, project, traj_store, mem_store,
    )
    if not hit_ids:
        return RecallResult(
            answer="No memories matched the search terms or question for this project.",
            hit_count=0, block_count=0, blocks_in_wall=0, wall_chars=0,
        )

    blocks = gather_windows(hit_ids, traj_store, mem_store, pad=pad)
    wall, blocks_included = render_wall(blocks, budget_tokens=budget_tokens)

    if not wall.strip():
        return RecallResult(
            answer="Hits were found but their EDUs could not be retrieved from the memory store.",
            hit_count=len(hit_ids), block_count=len(blocks), blocks_in_wall=0, wall_chars=0,
        )

    answer = await call_subagent(question, wall, model=model)
    return RecallResult(
        answer=answer,
        hit_count=len(hit_ids),
        block_count=len(blocks),
        blocks_in_wall=blocks_included,
        wall_chars=len(wall),
    )
