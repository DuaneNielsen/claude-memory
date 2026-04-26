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

import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .config import INDEX_CURATOR_MODEL, INDICES_DIR, RECENCY_DECAY_ALPHA
from .extractor import (
    INDEX_CURATOR_JSON_SCHEMA,
    INDEX_CURATOR_ONE_SHOT_INPUT,
    INDEX_CURATOR_ONE_SHOT_OUTPUT,
    INDEX_CURATOR_SYSTEM_PROMPT,
    EDUTag,
    call_claude,
)
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


# --- LLM-curated index types (input + output) ---------------------------- #

# Tags whose EDUs should be summarized as catalog entries (count + topic
# clusters). Config/project are excluded by default — they tend to be
# high-volume noise where a count is more useful than a clustering. If the
# project_overview ever feels impoverished we revisit.
CATALOG_TAGS: tuple[str, ...] = ("decision", "gotcha", "architecture")
# Trajectory digest cap (most recent first, subsampled if larger).
INDEX_INPUT_TRAJECTORY_CAP = 80
# Per-tag cap for catalog-only EDUs (most recent first).
INDEX_INPUT_EDUS_PER_TAG_CAP = 80
# Cap for preference EDUs (verbatim, all included if under).
INDEX_INPUT_PREFERENCE_CAP = 30


@dataclass
class IndexInputTrajectory:
    id: str
    date: str          # ISO date string (just the day) for display
    summary: str
    keywords: list[str]
    edu_count: int


@dataclass
class IndexInputEDU:
    edu_id: str
    tag: str
    text: str
    trajectory_keywords: list[str]
    date: str


@dataclass
class IndexInput:
    project: str
    total_trajectories: int
    total_edus: int
    time_range_start: str  # ISO date
    time_range_end: str    # ISO date
    trajectories: list[IndexInputTrajectory]
    preference_edus: list[IndexInputEDU]              # verbatim
    catalog_edus_by_tag: dict[str, list[IndexInputEDU]]  # tag → EDUs (capped)
    catalog_counts_by_tag: dict[str, int]             # tag → full count (uncapped)
    keyword_frequencies: dict[str, int]               # canonical keyword → count


@dataclass
class CuratedIndexAvailable:
    tag: str
    count: int
    topics: list[str]


@dataclass
class CuratedIndexPreference:
    text: str
    source_edu_ids: list[str] = field(default_factory=list)


@dataclass
class CuratedIndexThread:
    summary: str
    keywords: list[str] = field(default_factory=list)


@dataclass
class CuratedIndexRecent:
    date: str
    summary: str


@dataclass
class CuratedIndex:
    project: str
    project_overview: str = ""
    preferences: list[CuratedIndexPreference] = field(default_factory=list)
    available_memories: list[CuratedIndexAvailable] = field(default_factory=list)
    active_threads: list[CuratedIndexThread] = field(default_factory=list)
    recent_activity: list[CuratedIndexRecent] = field(default_factory=list)
    keyword_cloud: list[str] = field(default_factory=list)


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


def _gather_index_input(
    project: str,
    traj_store: TrajectoryStore,
    mem_store: MemoryStore,
) -> IndexInput:
    """Gather the wide-input slice the LLM curator needs.

    Returns an IndexInput with the trajectory digest, preference EDUs (verbatim,
    capped), per-tag catalog EDUs (capped), full per-tag counts, and the
    keyword frequency map. EDUs are sorted newest-first within each cap.
    """
    trajectories = traj_store.get_all_by_project(project)
    keyword_frequencies = traj_store.get_keyword_frequencies(project)

    if not trajectories:
        return IndexInput(
            project=project,
            total_trajectories=0,
            total_edus=0,
            time_range_start="",
            time_range_end="",
            trajectories=[],
            preference_edus=[],
            catalog_edus_by_tag={tag: [] for tag in CATALOG_TAGS},
            catalog_counts_by_tag={tag: 0 for tag in CATALOG_TAGS},
            keyword_frequencies=keyword_frequencies,
        )

    traj_ids = [t.id for t in trajectories]
    raw_edus = mem_store.get_edus_by_trajectories(traj_ids)
    traj_by_id = {t.id: t for t in trajectories}

    # Bucket EDUs by trajectory id and by tag
    edus_by_traj: dict[str, list[dict]] = {}
    for r in raw_edus:
        edus_by_traj.setdefault(r.get("trajectory_id", ""), []).append(r)

    # --- Trajectory digest (newest-first, capped) ---
    digest: list[IndexInputTrajectory] = []
    for t in trajectories:  # already DESC by created_at
        digest.append(IndexInputTrajectory(
            id=t.id,
            date=t.created_at.strftime("%Y-%m-%d"),
            summary=t.summary,
            keywords=list(t.keywords),
            edu_count=len(edus_by_traj.get(t.id, [])),
        ))
    if len(digest) > INDEX_INPUT_TRAJECTORY_CAP:
        digest = digest[:INDEX_INPUT_TRAJECTORY_CAP]

    # --- Per-tag EDU buckets ---
    preference_pool: list[IndexInputEDU] = []
    catalog_pool: dict[str, list[IndexInputEDU]] = {tag: [] for tag in CATALOG_TAGS}
    catalog_counts: dict[str, int] = {tag: 0 for tag in CATALOG_TAGS}

    for r in raw_edus:
        tag = r.get("tag", "")
        traj_id = r.get("trajectory_id", "") or ""
        traj = traj_by_id.get(traj_id)
        ts_raw = r.get("timestamp", "")
        try:
            ts = datetime.fromisoformat(ts_raw) if ts_raw else None
        except ValueError:
            ts = None
        date_str = ts.strftime("%Y-%m-%d") if ts else ""
        item = IndexInputEDU(
            edu_id=r.get("id", ""),
            tag=tag,
            text=r.get("text", ""),
            trajectory_keywords=list(traj.keywords) if traj else [],
            date=date_str,
        )
        if tag == EDUTag.PREFERENCE.value:
            preference_pool.append((ts, item))  # type: ignore[arg-type]
        elif tag in CATALOG_TAGS:
            catalog_counts[tag] += 1
            catalog_pool[tag].append((ts, item))  # type: ignore[arg-type]

    def _sort_and_strip(pool: list, cap: int) -> list[IndexInputEDU]:
        pool.sort(key=lambda pair: pair[0] or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
        return [item for (_ts, item) in pool[:cap]]

    preference_edus = _sort_and_strip(preference_pool, INDEX_INPUT_PREFERENCE_CAP)
    catalog_edus_by_tag = {
        tag: _sort_and_strip(catalog_pool[tag], INDEX_INPUT_EDUS_PER_TAG_CAP)
        for tag in CATALOG_TAGS
    }

    # --- Time range across trajectories ---
    starts = [t.created_at for t in trajectories]
    time_range_start = min(starts).strftime("%Y-%m-%d")
    time_range_end = max(starts).strftime("%Y-%m-%d")

    return IndexInput(
        project=project,
        total_trajectories=len(trajectories),
        total_edus=len(raw_edus),
        time_range_start=time_range_start,
        time_range_end=time_range_end,
        trajectories=digest,
        preference_edus=preference_edus,
        catalog_edus_by_tag=catalog_edus_by_tag,
        catalog_counts_by_tag=catalog_counts,
        keyword_frequencies=keyword_frequencies,
    )


def _format_curator_input(inp: IndexInput) -> str:
    """Render an IndexInput into the text the LLM curator sees."""
    lines: list[str] = [
        f"Project: {inp.project}",
        (
            f"Total trajectories: {inp.total_trajectories}    "
            f"Total EDUs: {inp.total_edus}    "
            f"Time range: {inp.time_range_start or 'n/a'} to {inp.time_range_end or 'n/a'}"
        ),
        "",
        "Per-tag full counts (these are the totals; some EDUs below may be subsamples):",
    ]
    counts_str = "    ".join(f"{tag}: {inp.catalog_counts_by_tag.get(tag, 0)}" for tag in CATALOG_TAGS)
    lines.append(f"  {counts_str}")
    lines.append("")

    if inp.keyword_frequencies:
        kw_pairs = sorted(inp.keyword_frequencies.items(), key=lambda kv: (-kv[1], kv[0]))
        kw_str = ", ".join(f"{k}: {v}" for k, v in kw_pairs)
    else:
        kw_str = "(none yet)"
    lines.append("Keyword frequencies (canonical cloud):")
    lines.append(f"  {kw_str}")
    lines.append("")

    lines.append("Trajectory digest (newest-first; id, date, summary, keywords, edu_count):")
    if not inp.trajectories:
        lines.append("  (none)")
    for i, t in enumerate(inp.trajectories, 1):
        kws = ", ".join(t.keywords) if t.keywords else "—"
        lines.append(f"{i}. [{t.id}, {t.date}] {t.summary} | [{kws}] | {t.edu_count} EDUs")
    lines.append("")

    lines.append("Preference EDUs (verbatim — eligible for `preferences` output):")
    if not inp.preference_edus:
        lines.append("  (none)")
    for e in inp.preference_edus:
        lines.append(f"- [{e.edu_id}, {e.date}] {e.text}")
    lines.append("")

    lines.append("Catalog EDUs (text + tag + per-trajectory keywords; capped per tag):")
    for tag in CATALOG_TAGS:
        bucket = inp.catalog_edus_by_tag.get(tag, [])
        lines.append(f"[{tag}]")
        if not bucket:
            lines.append("  (none)")
            continue
        for e in bucket:
            kws = ", ".join(e.trajectory_keywords) if e.trajectory_keywords else "—"
            lines.append(f"- [{e.edu_id}, {e.date}] {e.text} | traj_keywords: [{kws}]")
    return "\n".join(lines)


def _parse_curated_index(
    raw: dict,
    inp: IndexInput,
    project: str,
) -> CuratedIndex:
    """Coerce the LLM's JSON response into a CuratedIndex.

    Drops malformed entries; warns on `source_edu_ids` that don't reference an
    EDU we actually showed the model.
    """
    valid_edu_ids = {e.edu_id for e in inp.preference_edus}
    valid_edu_ids.update(
        e.edu_id for bucket in inp.catalog_edus_by_tag.values() for e in bucket
    )

    out = CuratedIndex(project=project)
    out.project_overview = str(raw.get("project_overview", "")).strip()

    for p in raw.get("preferences", []) or []:
        text = str(p.get("text", "")).strip()
        if not text:
            continue
        ids = [str(s) for s in (p.get("source_edu_ids") or []) if isinstance(s, str)]
        bad = [s for s in ids if s not in valid_edu_ids]
        if bad:
            log.warning(
                "Curator preference cited unknown source_edu_ids %s — keeping entry but stripping bad ids",
                bad,
            )
            ids = [s for s in ids if s in valid_edu_ids]
        out.preferences.append(CuratedIndexPreference(text=text, source_edu_ids=ids))

    for am in raw.get("available_memories", []) or []:
        tag = str(am.get("tag", "")).strip().lower()
        if tag not in CATALOG_TAGS:
            continue
        try:
            count = int(am.get("count", 0))
        except (TypeError, ValueError):
            count = 0
        topics = [str(t).strip() for t in (am.get("topics") or []) if isinstance(t, str) and str(t).strip()]
        out.available_memories.append(CuratedIndexAvailable(tag=tag, count=count, topics=topics))

    for at in raw.get("active_threads", []) or []:
        summary = str(at.get("summary", "")).strip()
        if not summary:
            continue
        keywords = [str(k).strip() for k in (at.get("keywords") or []) if isinstance(k, str) and str(k).strip()]
        out.active_threads.append(CuratedIndexThread(summary=summary, keywords=keywords))

    for ra in raw.get("recent_activity", []) or []:
        date = str(ra.get("date", "")).strip()
        summary = str(ra.get("summary", "")).strip()
        if not summary:
            continue
        out.recent_activity.append(CuratedIndexRecent(date=date, summary=summary))

    out.keyword_cloud = [
        str(k).strip()
        for k in (raw.get("keyword_cloud") or [])
        if isinstance(k, str) and str(k).strip()
    ]
    return out


async def curate_index_with_llm(
    inp: IndexInput,
    model: str | None = None,
) -> CuratedIndex:
    """Call the LLM curator and return a CuratedIndex.

    Errors (RateLimitError, malformed JSON, claude-CLI failures) propagate to
    the caller; `build_index` catches them and falls back to the deterministic
    builder.
    """
    text = _format_curator_input(inp)
    raw = await call_claude(
        text,
        INDEX_CURATOR_SYSTEM_PROMPT,
        model or INDEX_CURATOR_MODEL,
        json_schema=INDEX_CURATOR_JSON_SCHEMA,
        one_shot_input=INDEX_CURATOR_ONE_SHOT_INPUT,
        one_shot_output=INDEX_CURATOR_ONE_SHOT_OUTPUT,
    )
    return _parse_curated_index(raw, inp, inp.project)


def build_index(
    project: str,
    traj_store: TrajectoryStore,
    mem_store: MemoryStore,
    now: datetime | None = None,
) -> ProjectIndex:
    """Build a ProjectIndex for a project.

    Thin wrapper that delegates to the deterministic builder. Kept as the
    public entrypoint so the LLM-curation path can plug in here (see
    `write_index`) without rippling through the rest of the codebase.
    """
    return _build_index_deterministic(project, traj_store, mem_store, now=now)


def _build_index_deterministic(
    project: str,
    traj_store: TrajectoryStore,
    mem_store: MemoryStore,
    now: datetime | None = None,
) -> ProjectIndex:
    """Build the ProjectIndex for a single project from raw EDUs (no LLM)."""
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


def _pluralize_tag(tag: str, count: int) -> str:
    """Render a tag label for the available_memories section."""
    if tag == "decision":
        return "decisions" if count != 1 else "decision"
    if tag == "gotcha":
        return "gotchas" if count != 1 else "gotcha"
    if tag == "architecture":
        return "architecture notes" if count != 1 else "architecture note"
    return f"{tag}s" if count != 1 else tag


def render_curated_markdown(curated: CuratedIndex) -> str:
    """Render a CuratedIndex as markdown for the SessionStart wrapper.

    No leading h1 — that's added by `hooks/session_start_index.py`. Empty
    sections are dropped (except keyword_cloud, which is always emitted when
    non-empty since it doubles as the topic-coverage hint).
    """
    parts: list[str] = []

    if curated.project_overview:
        parts.append("## Project overview")
        parts.append(curated.project_overview)
        parts.append("")

    if curated.preferences:
        parts.append("## Preferences")
        for p in curated.preferences:
            parts.append(f"- {p.text}")
        parts.append("")

    if curated.available_memories:
        parts.append("## Memories available (call recall_get_context to retrieve)")
        for am in curated.available_memories:
            label = _pluralize_tag(am.tag, am.count)
            topics_str = ", ".join(am.topics) if am.topics else "(no clusters)"
            parts.append(f"- **{am.count} {label}** across topics: {topics_str}")
        parts.append("")

    if curated.active_threads:
        parts.append("## Active threads")
        for thread in curated.active_threads:
            kws = f" ({', '.join(thread.keywords)})" if thread.keywords else ""
            parts.append(f"- {thread.summary}{kws}")
        parts.append("")

    if curated.recent_activity:
        parts.append("## Recent activity")
        for ra in curated.recent_activity:
            prefix = f"[{ra.date}] " if ra.date else ""
            parts.append(f"- {prefix}{ra.summary}")
        parts.append("")

    if curated.keyword_cloud:
        parts.append("## Keyword cloud")
        parts.append(", ".join(curated.keyword_cloud))
        parts.append("")

    return "\n".join(parts).rstrip() + "\n"


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
    """Shrink a deterministic index in-place until it fits the token budget.

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


def enforce_token_budget_curated(
    curated: CuratedIndex,
    max_tokens: int = MAX_INDEX_TOKENS,
) -> CuratedIndex:
    """Shrink a CuratedIndex in-place until it fits the token budget.

    The LLM should respect the budget upstream, but trim defensively.
    Sections are trimmed in reverse priority order (least important first):
        keyword_cloud → recent_activity → available_memories → active_threads
        → preferences → project_overview
    Within each list section, drop entries from the tail (oldest / lowest-rank).
    """
    priority_pop = (
        "keyword_cloud",
        "recent_activity",
        "available_memories",
        "active_threads",
        "preferences",
    )
    while True:
        rendered = render_curated_markdown(curated)
        est_tokens = len(rendered) / CHARS_PER_TOKEN
        if est_tokens <= max_tokens:
            return curated

        trimmed = False
        for name in priority_pop:
            section = getattr(curated, name)
            if len(section) > 1:
                section.pop()
                trimmed = True
                break
        if trimmed:
            continue

        # All list sections at size ≤ 1. Truncate project_overview as last resort.
        if len(curated.project_overview) > 200:
            curated.project_overview = curated.project_overview[:200].rsplit(" ", 1)[0] + "…"
            continue
        log.warning(
            "Curated index for %s still over budget after trimming; returning as-is (%.0f tokens)",
            curated.project, est_tokens,
        )
        return curated


def _hash_index_input(inp: IndexInput) -> str:
    """Stable content hash of an IndexInput for cache-gating writes.

    Captures: trajectory IDs + summaries + EDU IDs + EDU texts. If nothing
    that the LLM would see changed, the hash is byte-identical and we can
    skip the LLM call.
    """
    h = hashlib.sha256()
    h.update(inp.project.encode())
    h.update(b"|trajs|")
    for t in sorted(inp.trajectories, key=lambda x: x.id):
        h.update(t.id.encode())
        h.update(b"\x1f")
        h.update(t.summary.encode())
        h.update(b"\x1f")
        h.update(",".join(t.keywords).encode())
        h.update(b"\x1e")
    h.update(b"|prefs|")
    for e in sorted(inp.preference_edus, key=lambda x: x.edu_id):
        h.update(e.edu_id.encode())
        h.update(b"\x1f")
        h.update(e.text.encode())
        h.update(b"\x1e")
    h.update(b"|catalog|")
    for tag in CATALOG_TAGS:
        h.update(tag.encode())
        h.update(b":")
        h.update(str(inp.catalog_counts_by_tag.get(tag, 0)).encode())
        h.update(b":")
        for e in sorted(inp.catalog_edus_by_tag.get(tag, []), key=lambda x: x.edu_id):
            h.update(e.edu_id.encode())
            h.update(b"\x1f")
            h.update(e.text.encode())
            h.update(b"\x1e")
    h.update(b"|kw|")
    for kw, freq in sorted(inp.keyword_frequencies.items()):
        h.update(f"{kw}:{freq}".encode())
        h.update(b"\x1e")
    return h.hexdigest()


def _meta_path(indices_dir: Path, project: str) -> Path:
    return indices_dir / f"{project}.md.meta.json"


def _read_meta_hash(meta_path: Path) -> str | None:
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    h = meta.get("input_hash")
    return h if isinstance(h, str) else None


def _write_meta(meta_path: Path, input_hash: str, source: str) -> None:
    meta = {
        "input_hash": input_hash,
        "source": source,  # "llm" | "deterministic"
        "written_at": datetime.now(timezone.utc).isoformat(),
    }
    tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
    tmp.write_text(json.dumps(meta, indent=2))
    tmp.replace(meta_path)


async def write_index(
    project: str,
    traj_store: TrajectoryStore | None = None,
    mem_store: MemoryStore | None = None,
    indices_dir: Path | None = None,
    now: datetime | None = None,
    use_llm: bool = True,
) -> Path:
    """Build the index for a project and write it to disk. Returns the path.

    Pipeline:
        1. Gather IndexInput.
        2. Compute hash of the input. If it matches the stored meta hash,
           skip everything and return the existing index path.
        3. Otherwise: try `curate_index_with_llm` (Opus). On any failure
           (RateLimitError, JSON error, missing CLI, etc.), fall back to
           the deterministic builder so the system stays usable offline.
        4. Render and write atomically. Update `<project>.md.meta.json`.
    """
    traj_store = traj_store or TrajectoryStore()
    mem_store = mem_store or MemoryStore()
    indices_dir = Path(indices_dir) if indices_dir else INDICES_DIR
    indices_dir.mkdir(parents=True, exist_ok=True)

    path = indices_dir / f"{project}.md"
    meta_path = _meta_path(indices_dir, project)

    inp = _gather_index_input(project, traj_store, mem_store)
    input_hash = _hash_index_input(inp)

    if path.exists() and _read_meta_hash(meta_path) == input_hash:
        log.debug("Index for %s unchanged (hash %s); skipping rebuild", project, input_hash[:8])
        return path

    rendered: str | None = None
    source = "deterministic"

    if use_llm:
        try:
            curated = await curate_index_with_llm(inp)
            curated = enforce_token_budget_curated(curated)
            rendered = render_curated_markdown(curated)
            source = "llm"
        except Exception as e:
            log.warning(
                "LLM index curation failed for %s (%s); falling back to deterministic builder",
                project, type(e).__name__,
            )

    if rendered is None:
        det = _build_index_deterministic(project, traj_store, mem_store, now=now)
        det = enforce_token_budget(det)
        rendered = render_markdown(det)

    tmp = path.with_suffix(".md.tmp")
    tmp.write_text(rendered)
    tmp.replace(path)
    _write_meta(meta_path, input_hash, source)
    return path


def _project_to_filename(project: str) -> str:
    """Preserve the existing project naming convention on disk."""
    return project  # projects already use safe names from CLAUDE_PROJECTS_DIR
