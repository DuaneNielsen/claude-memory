"""Build the per-project memory index injected at session start.

Injected content is the "top" of the hierarchical memory: orient Claude
without dumping detail. The MCP recall tool is the drill-down path for
anything not in the index.

Structure (all sections project-scoped):
    ## Preferences          — EDUs tagged PREFERENCE, deduped
    ## Recent Activity      — top-N trajectory summaries, recency-ordered
    ## Key Decisions/Gotchas — EDUs tagged DECISION or GOTCHA,
                              ranked by keyword-frequency × recency
    ## Keyword Cloud        — alphabetized topic tags for this project

Empty sections are omitted. Output is deterministic: byte-identical when
nothing has changed, so prompt-cache hits work across sessions.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .config import INDICES_DIR, RECENCY_DECAY_ALPHA
from .extractor import EDUTag
from .store import MemoryStore
from .trajectories import Trajectory, TrajectoryStore

log = logging.getLogger(__name__)

MAX_PREFERENCES = 8
MAX_RECENT_ACTIVITY = 8
MAX_KEY_DECISIONS = 8
MAX_KEYWORDS_IN_CLOUD = 40

# Token budget enforcement — rough char-to-token ratio for English markdown
MAX_INDEX_TOKENS = 600
CHARS_PER_TOKEN = 3.5


@dataclass
class IndexEDU:
    text: str
    tag: str
    trajectory_id: str
    timestamp: datetime
    score: float = 0.0


@dataclass
class ProjectIndex:
    project: str
    generated_at: datetime
    preferences: list[IndexEDU] = field(default_factory=list)
    recent_activity: list[Trajectory] = field(default_factory=list)
    key_decisions: list[IndexEDU] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)


def _as_utc(ts: datetime) -> datetime:
    """Normalize a datetime to UTC-aware so we can safely subtract across sources."""
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _recency_weight(ts: datetime, now: datetime, alpha: float = RECENCY_DECAY_ALPHA) -> float:
    days = max(0.0, (_as_utc(now) - _as_utc(ts)).total_seconds() / 86400.0)
    return math.exp(-alpha * days)


def _edu_from_metadata(doc: str, meta: dict) -> IndexEDU | None:
    try:
        ts = datetime.fromisoformat(meta["timestamp"])
    except (KeyError, ValueError):
        return None
    return IndexEDU(
        text=doc,
        tag=meta.get("tag", EDUTag.PROJECT.value),
        trajectory_id=meta.get("trajectory_id", "") or "",
        timestamp=ts,
    )


def build_index(
    project: str,
    traj_store: TrajectoryStore,
    mem_store: MemoryStore,
    now: datetime | None = None,
) -> ProjectIndex:
    """Build the ProjectIndex for a single project."""
    now = now or datetime.now(timezone.utc)

    trajectories = traj_store.get_all_by_project(project)
    index = ProjectIndex(project=project, generated_at=now)

    if not trajectories:
        # No data yet for this project — return empty index
        index.keywords = traj_store.get_keywords_for_project(project)
        return index

    # --- Fetch all EDUs attached to these trajectories ---
    traj_ids = [t.id for t in trajectories]
    raw_edus = mem_store.get_edus_by_trajectories(traj_ids)
    edus: list[IndexEDU] = []
    for r in raw_edus:
        edu = _edu_from_metadata(r["text"], r)
        if edu:
            edus.append(edu)

    # --- Keyword frequency map (for ranking decisions) ---
    kw_freq = traj_store.get_keyword_frequencies(project)
    traj_by_id = {t.id: t for t in trajectories}

    # --- Section: Preferences ---
    seen_pref_texts: set[str] = set()
    prefs = []
    # Most-recent preferences first
    for edu in sorted(edus, key=lambda e: e.timestamp, reverse=True):
        if edu.tag != EDUTag.PREFERENCE.value:
            continue
        norm = edu.text.strip().lower()
        if norm in seen_pref_texts:
            continue
        seen_pref_texts.add(norm)
        prefs.append(edu)
        if len(prefs) >= MAX_PREFERENCES:
            break
    index.preferences = prefs

    # --- Section: Recent Activity (trajectory summaries) ---
    index.recent_activity = trajectories[:MAX_RECENT_ACTIVITY]

    # --- Section: Key Decisions / Gotchas ---
    # Score each decision/gotcha EDU by: max keyword freq in its trajectory × recency weight
    def score_edu(edu: IndexEDU) -> float:
        traj = traj_by_id.get(edu.trajectory_id)
        if traj is None or not traj.keywords:
            max_freq = 1
        else:
            max_freq = max((kw_freq.get(kw, 1) for kw in traj.keywords), default=1)
        return max_freq * _recency_weight(edu.timestamp, now)

    for edu in edus:
        edu.score = score_edu(edu)

    decisions = [
        e for e in edus
        if e.tag in (EDUTag.DECISION.value, EDUTag.GOTCHA.value)
    ]
    decisions.sort(key=lambda e: e.score, reverse=True)

    # Dedup by text, keep highest-scoring variant
    seen_dec_texts: set[str] = set()
    top_decisions = []
    for edu in decisions:
        norm = edu.text.strip().lower()
        if norm in seen_dec_texts:
            continue
        seen_dec_texts.add(norm)
        top_decisions.append(edu)
        if len(top_decisions) >= MAX_KEY_DECISIONS:
            break
    index.key_decisions = top_decisions

    # --- Section: Keyword Cloud ---
    all_keywords = traj_store.get_keywords_for_project(project)
    if len(all_keywords) > MAX_KEYWORDS_IN_CLOUD:
        # Keep the most frequent ones, then alphabetize
        top_by_freq = sorted(
            all_keywords,
            key=lambda k: (-kw_freq.get(k, 0), k),
        )[:MAX_KEYWORDS_IN_CLOUD]
        index.keywords = sorted(top_by_freq)
    else:
        index.keywords = all_keywords

    return index


def render_markdown(index: ProjectIndex) -> str:
    """Render a ProjectIndex as a stable markdown string (empty sections omitted)."""
    parts: list[str] = []

    if index.preferences:
        parts.append("## Preferences")
        for p in index.preferences:
            parts.append(f"- {p.text}")
        parts.append("")

    if index.recent_activity:
        parts.append(f"## Recent Activity ({index.project})")
        for t in index.recent_activity:
            date = t.created_at.strftime("%Y-%m-%d")
            parts.append(f"- [{date}] {t.summary}")
        parts.append("")

    if index.key_decisions:
        parts.append("## Key Decisions / Gotchas")
        for d in index.key_decisions:
            parts.append(f"- [{d.tag}] {d.text}")
        parts.append("")

    if index.keywords:
        parts.append("## Keyword Cloud")
        parts.append(", ".join(index.keywords))
        parts.append("")

    return "\n".join(parts).rstrip() + "\n"


def enforce_token_budget(
    index: ProjectIndex,
    max_tokens: int = MAX_INDEX_TOKENS,
) -> ProjectIndex:
    """Shrink an index in-place until it fits the token budget.

    Truncates sections proportionally; keeps at least 1 item per non-empty
    section when possible. Returns the same index (mutated).
    """
    while True:
        rendered = render_markdown(index)
        est_tokens = len(rendered) / CHARS_PER_TOKEN
        if est_tokens <= max_tokens:
            return index

        # Pick the largest section to trim from
        sizes = [
            (len(index.preferences), "preferences"),
            (len(index.recent_activity), "recent_activity"),
            (len(index.key_decisions), "key_decisions"),
        ]
        sizes.sort(reverse=True)
        trimmed = False
        for size, name in sizes:
            if size > 1:
                section = getattr(index, name)
                section.pop()
                trimmed = True
                break

        if not trimmed:
            # All sections at size ≤ 1 — last resort, trim keyword cloud
            if len(index.keywords) > 10:
                index.keywords = index.keywords[: len(index.keywords) // 2]
                continue
            log.warning(
                "Index for %s still over budget after trimming; returning as-is (%.0f tokens)",
                index.project, est_tokens,
            )
            return index


def write_index(
    project: str,
    traj_store: TrajectoryStore | None = None,
    mem_store: MemoryStore | None = None,
    indices_dir: Path | None = None,
    now: datetime | None = None,
) -> Path:
    """Build the index for a project and write it to disk. Returns the path."""
    traj_store = traj_store or TrajectoryStore()
    mem_store = mem_store or MemoryStore()
    indices_dir = Path(indices_dir) if indices_dir else INDICES_DIR
    indices_dir.mkdir(parents=True, exist_ok=True)

    index = build_index(project, traj_store, mem_store, now=now)
    index = enforce_token_budget(index)
    rendered = render_markdown(index)

    path = indices_dir / f"{project}.md"
    # Write atomically so a concurrent SessionStart hook never sees a partial file
    tmp = path.with_suffix(".md.tmp")
    tmp.write_text(rendered)
    tmp.replace(path)
    return path


def _project_to_filename(project: str) -> str:
    """Preserve the existing project naming convention on disk."""
    return project  # projects already use safe names from CLAUDE_PROJECTS_DIR
