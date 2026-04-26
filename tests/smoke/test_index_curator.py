"""Layer 3 smoke test for the LLM-curated index builder.

Keyless: never calls the `claude` CLI. Exercises the pure-function path —
input formatting, JSON parsing, markdown rendering, token-budget trimming,
and hash-gated atomic writes. The actual LLM call (`curate_index_with_llm`)
lives in a separate keyed workflow we run on demand.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from claude_memory.index_builder import (
    CATALOG_TAGS,
    CuratedIndex,
    CuratedIndexAvailable,
    CuratedIndexPreference,
    CuratedIndexRecent,
    CuratedIndexThread,
    IndexInput,
    IndexInputEDU,
    IndexInputTrajectory,
    _format_curator_input,
    _hash_index_input,
    _parse_curated_index,
    enforce_token_budget_curated,
    render_curated_markdown,
    write_index,
)


def _fake_input() -> IndexInput:
    return IndexInput(
        project="projects-claude-memory",
        total_trajectories=3,
        total_edus=8,
        time_range_start="2026-04-20",
        time_range_end="2026-04-26",
        trajectories=[
            IndexInputTrajectory(
                id="t-1",
                date="2026-04-26",
                summary="Designed the LLM-curated index pipeline",
                keywords=["claude-memory", "index-builder", "planning"],
                edu_count=4,
            ),
            IndexInputTrajectory(
                id="t-2",
                date="2026-04-25",
                summary="Wired up the CI matrix",
                keywords=["ci", "claude-memory", "docker"],
                edu_count=3,
            ),
        ],
        preference_edus=[
            IndexInputEDU(
                edu_id="edu-pref-1",
                tag="preference",
                text="For 'why did you decide X' questions, dispatch recall_get_context via an Agent.",
                trajectory_keywords=["claude-memory"],
                date="2026-04-22",
            ),
        ],
        catalog_edus_by_tag={
            "decision": [
                IndexInputEDU(
                    edu_id="edu-d-1",
                    tag="decision",
                    text="Reframed index as catalog (advertisement), not digest.",
                    trajectory_keywords=["claude-memory", "index-builder"],
                    date="2026-04-26",
                ),
            ],
            "gotcha": [],
            "architecture": [
                IndexInputEDU(
                    edu_id="edu-a-1",
                    tag="architecture",
                    text="Hash-gated index writes use <project>.md.meta.json.",
                    trajectory_keywords=["claude-memory", "index-builder"],
                    date="2026-04-26",
                ),
            ],
        },
        catalog_counts_by_tag={"decision": 5, "gotcha": 2, "architecture": 3},
        keyword_frequencies={
            "claude-memory": 3,
            "index-builder": 2,
            "ci": 1,
            "docker": 1,
            "planning": 1,
        },
    )


def _fake_curated() -> CuratedIndex:
    return CuratedIndex(
        project="projects-claude-memory",
        project_overview=(
            "claude-memory is a local conversation-memory plugin for Claude Code. "
            "Currently focused on the LLM-curated SessionStart index."
        ),
        preferences=[
            CuratedIndexPreference(
                text="For 'why did you decide X' questions, dispatch recall_get_context via an Agent.",
                source_edu_ids=["edu-pref-1"],
            ),
        ],
        available_memories=[
            CuratedIndexAvailable(tag="decision", count=5, topics=["claude-memory", "index-builder"]),
            CuratedIndexAvailable(tag="gotcha", count=2, topics=["ci"]),
            CuratedIndexAvailable(tag="architecture", count=3, topics=["claude-memory", "index-builder"]),
        ],
        active_threads=[
            CuratedIndexThread(
                summary="LLM-curated index design",
                keywords=["claude-memory", "index-builder", "planning"],
            ),
        ],
        recent_activity=[
            CuratedIndexRecent(date="2026-04-26", summary="Designed the LLM-curated index pipeline"),
            CuratedIndexRecent(date="2026-04-25", summary="Wired up the CI matrix"),
        ],
        keyword_cloud=["ci", "claude-memory", "docker", "index-builder", "planning"],
    )


def test_format_curator_input_includes_required_sections():
    text = _format_curator_input(_fake_input())
    assert "Project: projects-claude-memory" in text
    assert "Total trajectories: 3" in text
    assert "Per-tag full counts" in text
    assert "decision: 5" in text
    assert "Keyword frequencies" in text
    assert "claude-memory: 3" in text
    assert "Trajectory digest" in text
    assert "Designed the LLM-curated index pipeline" in text
    assert "Preference EDUs" in text
    assert "edu-pref-1" in text
    assert "[decision]" in text
    assert "[gotcha]" in text
    assert "[architecture]" in text


def test_render_curated_markdown_shape():
    rendered = render_curated_markdown(_fake_curated())
    assert "## Project overview" in rendered
    assert "## Preferences" in rendered
    assert "## Memories available (call recall_get_context to retrieve)" in rendered
    assert "**5 decisions**" in rendered
    assert "**2 gotchas**" in rendered
    assert "**3 architecture notes**" in rendered
    assert "## Active threads" in rendered
    assert "## Recent activity" in rendered
    assert "[2026-04-26] Designed" in rendered
    assert "## Keyword cloud" in rendered
    # No leading h1 — the SessionStart hook adds that
    assert not rendered.lstrip().startswith("# ")


def test_render_curated_markdown_drops_empty_sections():
    minimal = CuratedIndex(project="p", keyword_cloud=["a", "b"])
    rendered = render_curated_markdown(minimal)
    assert "## Keyword cloud" in rendered
    assert "## Project overview" not in rendered
    assert "## Preferences" not in rendered
    assert "## Memories available" not in rendered
    assert "## Active threads" not in rendered
    assert "## Recent activity" not in rendered


def test_parse_curated_index_strips_unknown_source_edu_ids(caplog):
    raw = {
        "project_overview": "ok",
        "preferences": [
            {
                "text": "good pref",
                "source_edu_ids": ["edu-pref-1", "ghost-edu"],
            },
        ],
        "available_memories": [
            {"tag": "decision", "count": 5, "topics": ["claude-memory"]},
            {"tag": "bogus", "count": 99, "topics": ["x"]},  # unknown tag dropped
        ],
        "active_threads": [
            {"summary": "thread", "keywords": ["a"]},
        ],
        "recent_activity": [
            {"date": "2026-04-26", "summary": "did stuff"},
        ],
        "keyword_cloud": ["claude-memory", "ci"],
    }
    out = _parse_curated_index(raw, _fake_input(), "projects-claude-memory")
    assert len(out.preferences) == 1
    assert out.preferences[0].source_edu_ids == ["edu-pref-1"]
    # The unknown "bogus" tag should be dropped.
    assert {am.tag for am in out.available_memories} == {"decision"}
    assert out.active_threads[0].summary == "thread"
    assert out.recent_activity[0].date == "2026-04-26"


def test_hash_input_stable_and_change_detecting():
    a = _hash_index_input(_fake_input())
    b = _hash_index_input(_fake_input())
    assert a == b, "hash should be deterministic for identical input"

    inp2 = _fake_input()
    inp2.catalog_counts_by_tag["decision"] += 1
    c = _hash_index_input(inp2)
    assert a != c, "hash should change when a per-tag count changes"


def test_enforce_token_budget_curated_trims_to_fit():
    # Build a deliberately oversized curated index by repeating sections.
    big = CuratedIndex(
        project="p",
        project_overview="P" * 400,
        preferences=[CuratedIndexPreference(text="a pref " * 20) for _ in range(20)],
        available_memories=[
            CuratedIndexAvailable(tag="decision", count=10, topics=["x"] * 30),
        ] * 3,
        active_threads=[
            CuratedIndexThread(summary="t " * 30, keywords=["x"] * 5)
            for _ in range(10)
        ],
        recent_activity=[
            CuratedIndexRecent(date="2026-04-26", summary="did some work " * 10)
            for _ in range(20)
        ],
        keyword_cloud=["kw" + str(i) for i in range(60)],
    )
    out = enforce_token_budget_curated(big, max_tokens=400)
    rendered = render_curated_markdown(out)
    # Char/token ≈ 3.5 — give it some headroom in the assert
    assert len(rendered) / 3.5 <= 420, f"render still too big: {len(rendered)} chars"


def test_write_index_is_async():
    # Quick sanity check that write_index is exposed as a coroutine — calling
    # it synchronously would raise TypeError. (We don't actually run it here:
    # that would require a real TrajectoryStore + MemoryStore.)
    import inspect

    assert inspect.iscoroutinefunction(write_index)


def test_meta_json_round_trip(tmp_path: Path):
    # _write_meta + _read_meta_hash form a tiny round-trip pair.
    from claude_memory.index_builder import _read_meta_hash, _write_meta

    meta_path = tmp_path / "p.md.meta.json"
    _write_meta(meta_path, "abc123", source="llm")
    assert _read_meta_hash(meta_path) == "abc123"
    payload = json.loads(meta_path.read_text())
    assert payload["source"] == "llm"
    assert "written_at" in payload


def test_catalog_tags_constant_excludes_config_and_project():
    # Plan §1 default: try without config/project first; revisit if the
    # project_overview looks impoverished. This test pins that decision.
    assert "config" not in CATALOG_TAGS
    assert "project" not in CATALOG_TAGS
    assert set(CATALOG_TAGS) == {"decision", "gotcha", "architecture"}
