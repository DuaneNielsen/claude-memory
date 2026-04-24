"""Extract Elementary Discourse Units (EDUs) from conversation turns via LLM."""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .config import (
    BOUNDARY_CONCURRENCY,
    BOUNDARY_MODEL,
    BOUNDARY_WINDOW_SIZE,
    CHUNK_CONTEXT_MARGIN,
    CHUNK_CORE_SIZE,
    CHUNK_OVERLAP_TURNS,
    DEFAULT_MODEL,
    MAX_TURNS_PER_CHUNK,
)
from .parser import Session, Turn

log = logging.getLogger(__name__)


class EDUTag(str, Enum):
    DECISION = "decision"          # architectural/design choices and their rationale
    PREFERENCE = "preference"      # user preferences, communication style, workflow
    GOTCHA = "gotcha"              # bugs, surprises, things to avoid, non-obvious caveats
    CONFIG = "config"              # config values, paths, ports, flags, command invocations
    ARCHITECTURE = "architecture"  # how systems are structured, data flow, component relationships
    PROJECT = "project"            # project status, goals, ongoing work

    @classmethod
    def coerce(cls, value: str | None) -> "EDUTag":
        """Parse a tag string, defaulting to PROJECT on unknown values."""
        if not value:
            return cls.PROJECT
        try:
            return cls(value.strip().lower())
        except ValueError:
            return cls.PROJECT

SYSTEM_PROMPT = """\
You are a memory extraction system. Given a conversation session between a user and an AI assistant (Claude Code), decompose it into Elementary Discourse Units (EDUs) — short, self-contained statements that each express a single fact, event, decision, or preference.

CRITICAL SOURCE BOUNDARY: Use ONLY the numbered turns listed in the user message below as your source material. Do NOT extract facts from any other context you can see — no CLAUDE.md, no tool descriptions, no skill or plugin registry, no general knowledge, no prior conversations. If a fact isn't in the numbered turns provided, do not include it. Every EDU's `source_turn_ids` must reference turn numbers that actually appear in the input.

CORE vs CONTEXT SECTIONS: The input may be divided into "=== CONTEXT ===" and "=== CORE ===" sections. Extract EDUs ONLY from turns in CORE sections. CONTEXT turns are provided so you can understand the surrounding conversation, but facts stated only in a CONTEXT turn must be skipped. Every EDU's `source_turn_ids` must contain only turn numbers that appear under a "=== CORE ===" header.

Requirements:
1. Each EDU must be independently understandable without any other EDU or the original conversation for context.
2. Replace all pronouns and ambiguous references with specific names, tools, paths, or details. Use the most informative identifier consistently (e.g. "PipeWire filter-chain" not "it", "the primesignal project" not "the project").
3. Preserve ALL substantive information — no detail should be lost. Technical details (paths, configs, commands, parameters) are especially important.
4. Infer and include temporal context. Convert relative dates to absolute where possible (e.g. "yesterday" → "2026-03-19" if the conversation date is 2026-03-20).
5. Separate distinct facts into distinct EDUs, even if they appear in the same message. One fact per EDU.
6. Skip conversational filler ("sure", "let me check", "here's what I found") — extract only substantive content.
7. Capture:
   - Technical decisions and their reasoning
   - Configuration values and paths
   - Problems encountered and their solutions
   - User preferences and corrections
   - Project context and goals
   - Commands that were run and their outcomes (briefly)
8. For each EDU, include the turn numbers it was derived from.
9. Assign each EDU exactly one `tag` from this fixed set (pick the best single fit):
   - "decision": architectural/design choices and their rationale (why X was chosen over Y)
   - "preference": user preferences, communication style, workflow conventions
   - "gotcha": bugs, surprises, things to avoid, non-obvious caveats, brittle workarounds
   - "config": concrete config values, paths, ports, flags, command invocations
   - "architecture": how systems are structured, data flow, component relationships
   - "project": project status, goals, ongoing work, who's doing what

Output as JSON: {"edus": [{"text": "...", "source_turn_ids": [1, 2], "tag": "config"}, ...]}"""

ONE_SHOT_INPUT = """\
Session date: 2026-03-20
Project: home

Turn 1 [user]: EasyEffects is messing up my audio routing. Everything is going through its virtual sink even when I don't want it to
Turn 2 [assistant]: I can see EasyEffects has inserted itself as the default sink. Let me check what's routing through it. Looking at wpctl status, all your playback streams are being captured by the EasyEffects sink. Since you mentioned you only use it for mic processing, we could replace it with a native PipeWire filter-chain that only processes the mic input without touching playback routing.
Turn 3 [user]: yeah do that. rip it out
Turn 4 [assistant]: Done. I've removed EasyEffects and created a PipeWire filter-chain config at ~/.config/pipewire/pipewire.conf.d/20-mic-chain.conf. The chain is: Yeti Nano → highpass 80Hz → LSP compressor (-26dB threshold, 4:1) → LSP de-esser (-24dB, 6:1, SC HPF 4kHz/LPF 9kHz) → LSP limiter (-3dB). The processed mic appears as "mic_chain.output". Your playback now goes directly to HDMI without any intermediary.
Turn 5 [user]: what about the LV2 plugins, does pipewire support those natively?
Turn 6 [assistant]: PipeWire has an LV2 module but the Ubuntu PPA build doesn't include it. I built it from source — cloned PipeWire tag 1.0.5 to ~/builds/pipewire/ and compiled libpipewire-module-filter-chain-lv2.so. Installed it to /usr/lib/x86_64-linux-gnu/pipewire-0.3/. Note: if PipeWire gets updated via apt, this custom module may need rebuilding."""

ONE_SHOT_OUTPUT = json.dumps({"edus": [
    {"text": "EasyEffects was removed from Duane's system on 2026-03-20 because it was hijacking all playback audio into its virtual sink, breaking audio routing", "source_turn_ids": [1, 3], "tag": "decision"},
    {"text": "EasyEffects was replaced with a native PipeWire filter-chain that only processes mic input without affecting playback routing", "source_turn_ids": [2, 3], "tag": "decision"},
    {"text": "The PipeWire mic processing filter-chain config is located at ~/.config/pipewire/pipewire.conf.d/20-mic-chain.conf", "source_turn_ids": [4], "tag": "config"},
    {"text": "The PipeWire mic chain order is: Yeti Nano → highpass 80Hz (2x biquad) → LSP compressor (leveling, -26dB threshold, 4:1 ratio) → LSP compressor (de-esser, -24dB threshold, 6:1 ratio, sidechain HPF 4kHz/LPF 9kHz) → LSP limiter (Herm Thin, -3dB)", "source_turn_ids": [4], "tag": "architecture"},
    {"text": "The processed mic output appears as 'mic_chain.output' (named 'Processed Mic') in PipeWire", "source_turn_ids": [4], "tag": "config"},
    {"text": "After removing EasyEffects, playback audio routes directly to HDMI output (Navi 31) without any intermediary sink", "source_turn_ids": [4], "tag": "architecture"},
    {"text": "PipeWire's LV2 filter-chain module (libpipewire-module-filter-chain-lv2.so) was built from source because the Ubuntu PPA build did not include it", "source_turn_ids": [5, 6], "tag": "gotcha"},
    {"text": "PipeWire was built from source at ~/builds/pipewire/ using tag 1.0.5, and the LV2 module was installed to /usr/lib/x86_64-linux-gnu/pipewire-0.3/", "source_turn_ids": [6], "tag": "config"},
    {"text": "If PipeWire is updated via apt, the custom-built LV2 module may need to be rebuilt from ~/builds/pipewire/", "source_turn_ids": [6], "tag": "gotcha"},
]})


@dataclass
class EDU:
    edu_id: str
    text: str
    source_turn_ids: list[int]
    session_id: str
    project: str
    timestamp: datetime
    speakers: list[str] = field(default_factory=list)
    tag: EDUTag = EDUTag.PROJECT
    trajectory_id: str | None = None
    trajectory_index: int = 0


def format_turns_for_extraction(turns: list[Turn], session_date: str, project: str) -> str:
    """Format turns into the input text for the extraction prompt."""
    lines = [f"Session date: {session_date}", f"Project: {project}", ""]
    for t in turns:
        lines.append(f"Turn {t.turn_id} [{t.speaker}]: {t.text}")
    return "\n".join(lines)


def chunk_turns(turns: list[Turn]) -> list[list[Turn]]:
    """Split long sessions into overlapping chunks (legacy flat-EDU path only)."""
    if len(turns) <= MAX_TURNS_PER_CHUNK:
        return [turns]

    chunks = []
    start = 0
    while start < len(turns):
        end = start + MAX_TURNS_PER_CHUNK
        chunks.append(turns[start:end])
        start = end - CHUNK_OVERLAP_TURNS
        if start + CHUNK_OVERLAP_TURNS >= len(turns):
            break
    return chunks


def chunk_turns_with_context(turns: list[Turn]) -> list[tuple[list[Turn], set[int]]]:
    """Split turns into non-overlapping CORE regions with CONTEXT margins.

    Each returned tuple is (window_turns, core_turn_ids) where:
      - window_turns is the contiguous slice of turns the LLM will see
        (core + up to CHUNK_CONTEXT_MARGIN turns on each side)
      - core_turn_ids is the set of turn IDs the LLM should extract EDUs from;
        turns outside this set are context-only

    Cores are non-overlapping so the same EDU isn't extracted twice.
    """
    if not turns:
        return []
    if len(turns) <= CHUNK_CORE_SIZE:
        core_ids = {t.turn_id for t in turns}
        return [(turns, core_ids)]

    chunks: list[tuple[list[Turn], set[int]]] = []
    core_start = 0
    while core_start < len(turns):
        core_end = min(core_start + CHUNK_CORE_SIZE, len(turns))
        context_start = max(0, core_start - CHUNK_CONTEXT_MARGIN)
        context_end = min(len(turns), core_end + CHUNK_CONTEXT_MARGIN)
        window_turns = turns[context_start:context_end]
        core_ids = {turns[i].turn_id for i in range(core_start, core_end)}
        chunks.append((window_turns, core_ids))
        core_start = core_end
    return chunks


EDU_JSON_SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "edus": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "source_turn_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                    },
                    "tag": {
                        "type": "string",
                        "enum": [t.value for t in EDUTag],
                    },
                },
                "required": ["text", "source_turn_ids", "tag"],
            },
        },
    },
    "required": ["edus"],
})


async def call_claude(
    text: str,
    system_prompt: str,
    model: str | None = None,
    json_schema: str | None = None,
    one_shot_input: str | None = None,
    one_shot_output: str | None = None,
) -> dict:
    """Call claude CLI for extraction with an enforced JSON schema.

    Defaults reproduce the flat-EDU extraction path. Pass `json_schema`,
    `one_shot_input`, and `one_shot_output` to drive a different extraction
    mode (e.g. trajectory extraction).
    """
    model = model or DEFAULT_MODEL
    schema = json_schema if json_schema is not None else EDU_JSON_SCHEMA
    example_in = one_shot_input if one_shot_input is not None else ONE_SHOT_INPUT
    example_out = one_shot_output if one_shot_output is not None else ONE_SHOT_OUTPUT

    if example_in and example_out:
        prompt = f"""Here is an example of the expected input and output:

INPUT:
{example_in}

OUTPUT:
{example_out}

Now process this input:

{text}"""
    else:
        # Explicit empty one-shot → no example block (simple binary classifiers etc.)
        prompt = text

    t_start = time.monotonic()
    # Pass the prompt as a positional arg rather than via stdin to avoid the
    # claude CLI's "no stdin data received in 3s" race under concurrent-process
    # load (multiple subprocesses fighting for I/O during stage 2).
    proc = await asyncio.create_subprocess_exec(
        "claude", "-p",
        "--model", model,
        "--tools", "",
        "--system-prompt", system_prompt,
        "--json-schema", schema,
        "--output-format", "json",
        "--no-session-persistence",
        prompt,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        # Run from /tmp so cwd-traversal doesn't find ~/CLAUDE.md and inject
        # it into the LLM's context — the model would otherwise hallucinate
        # "session" trajectories whose facts actually came from CLAUDE.md.
        cwd="/tmp",
    )
    stdout, stderr = await proc.communicate()
    elapsed = time.monotonic() - t_start
    stdout_str = stdout.decode()
    stderr_str = stderr.decode()

    _check_rate_limited(stdout_str, stderr_str)

    # Defense-in-depth: a silent sub-5s exit != 0 is the fingerprint of a rate-limit
    # refusal. Treat as rate limit so ingest aborts cleanly instead of hammering
    # every remaining chunk.
    if proc.returncode != 0 and elapsed < 5.0 and not stdout_str.strip() and not stderr_str.strip():
        raise RateLimitError(
            f"claude CLI exited {proc.returncode} in {elapsed:.1f}s with no output — suspected rate limit"
        )

    if proc.returncode != 0:
        err = stderr_str.strip() or stdout_str.strip()[:1000] or "(no output on stderr or stdout)"
        raise RuntimeError(f"claude CLI failed (exit {proc.returncode}): {err}")

    content = stdout_str.strip()
    if not content:
        raise ValueError("claude CLI returned empty output")

    events = json.loads(content)
    if isinstance(events, list):
        for event in reversed(events):
            if isinstance(event, dict) and "structured_output" in event:
                return event["structured_output"]

    raise ValueError("No structured_output found in claude CLI response")


class RateLimitError(RuntimeError):
    """Raised when claude CLI rejects a call due to subscription rate limit."""


def _check_rate_limited(stdout_str: str, stderr_str: str) -> None:
    """Parse claude CLI output for rate-limit signals. Raises RateLimitError if hit."""
    # Cheap substring screen first
    combined = stdout_str + "\n" + stderr_str
    if "rate_limit" not in combined.lower() and "overagestatus" not in combined.lower():
        return
    # Parse stdout JSON if present and look for overageStatus=rejected or is_error
    try:
        events = json.loads(stdout_str) if stdout_str.strip().startswith("[") else None
    except json.JSONDecodeError:
        events = None
    if isinstance(events, list):
        for event in events:
            if not isinstance(event, dict):
                continue
            rl = event.get("rate_limit_info") or event.get("rate_limit_event", {}).get("rate_limit_info")
            if isinstance(rl, dict) and rl.get("overageStatus") == "rejected" and rl.get("status") != "allowed":
                raise RateLimitError(f"claude CLI rate-limited: {rl}")
            if event.get("type") == "result" and event.get("is_error") and "rate" in str(event).lower():
                raise RateLimitError(f"claude CLI rate-limited: {str(event)[:300]}")
    # Fallback: substring match in stderr (less structured errors)
    if "rate limit" in stderr_str.lower() or "rate_limit" in stderr_str.lower():
        raise RateLimitError(f"claude CLI rate-limited (stderr): {stderr_str[:300]}")


INCREMENTAL_SYSTEM_PROMPT = """\
You are a memory extraction system. You are given:
1. Previously extracted facts (EDUs) from earlier turns of this conversation
2. New turns that have been added since the last extraction

Your job: extract NEW Elementary Discourse Units (EDUs) from ONLY the new turns. Do NOT repeat any fact already captured in the existing EDUs.

Requirements:
1. Each EDU must be independently understandable without any other EDU or the original conversation for context.
2. Replace all pronouns and ambiguous references with specific names, tools, paths, or details.
3. Preserve ALL substantive information from the new turns — no detail should be lost.
4. Infer and include temporal context. Convert relative dates to absolute where possible.
5. Separate distinct facts into distinct EDUs. One fact per EDU.
6. Skip conversational filler — extract only substantive content.
7. Include the turn numbers each EDU was derived from.
8. If a new turn merely confirms or restates something already in the existing EDUs, skip it.
9. Assign each EDU exactly one `tag` from this fixed set (pick the best single fit):
   - "decision": architectural/design choices and their rationale
   - "preference": user preferences, communication style, workflow conventions
   - "gotcha": bugs, surprises, things to avoid, non-obvious caveats
   - "config": concrete config values, paths, ports, flags, command invocations
   - "architecture": how systems are structured, data flow, component relationships
   - "project": project status, goals, ongoing work

Output as JSON: {"edus": [{"text": "...", "source_turn_ids": [1, 2], "tag": "config"}, ...]}
If there are no new facts, output: {"edus": []}"""


def format_incremental_input(
    existing_edus: list[dict],
    new_turns: list[Turn],
    session_date: str,
    project: str,
) -> str:
    """Format existing EDUs + new turns for incremental extraction."""
    lines = [f"Session date: {session_date}", f"Project: {project}", ""]

    lines.append("=== Previously extracted facts ===")
    for i, edu in enumerate(existing_edus, 1):
        lines.append(f"  {i}. {edu['text']}")
    lines.append("")

    lines.append("=== New turns (extract from these only) ===")
    for t in new_turns:
        lines.append(f"Turn {t.turn_id} [{t.speaker}]: {t.text}")

    return "\n".join(lines)


async def extract_edus_incremental(
    session: Session,
    existing_edus: list[dict],
    old_turn_count: int,
    model: str | None = None,
    on_chunk_done: callable = None,
) -> list[EDU]:
    """Extract EDUs from only the new turns, using existing EDUs as context."""
    new_turns = session.turns[old_turn_count:]
    if not new_turns:
        if on_chunk_done:
            on_chunk_done()
        return []

    session_date = session.turns[0].timestamp.strftime("%Y-%m-%d")
    chunks = chunk_turns(new_turns)

    all_edus: list[EDU] = []
    seen_texts: set[str] = set()
    # Include existing EDU texts in seen set to avoid duplicates
    for edu in existing_edus:
        seen_texts.add(edu["text"].lower())

    for chunk in chunks:
        text = format_incremental_input(existing_edus, chunk, session_date, session.project)
        try:
            result = await call_claude(text, INCREMENTAL_SYSTEM_PROMPT, model)
        except Exception as e:
            log.error(f"Incremental extraction failed for session {session.session_id}: {e}")
            if on_chunk_done:
                on_chunk_done()
            continue

        raw_edus = result.get("edus", [])
        for raw in raw_edus:
            edu_text = raw.get("text", "").strip()
            if not edu_text:
                continue
            normalized = edu_text.lower()
            if normalized in seen_texts:
                continue
            seen_texts.add(normalized)

            source_ids = raw.get("source_turn_ids", [])
            speakers = list({t.speaker for t in chunk if t.turn_id in source_ids})
            source_turns = [t for t in chunk if t.turn_id in source_ids]
            ts = source_turns[0].timestamp if source_turns else new_turns[0].timestamp

            all_edus.append(EDU(
                edu_id=str(uuid.uuid4()),
                text=edu_text,
                source_turn_ids=source_ids,
                session_id=session.session_id,
                project=session.project,
                timestamp=ts,
                speakers=speakers,
                tag=EDUTag.coerce(raw.get("tag")),
            ))

        if on_chunk_done:
            on_chunk_done()

    return all_edus


def count_chunks_incremental(session: Session, old_turn_count: int) -> int:
    """Count stage-1 chunks for incremental extraction."""
    new_turns = session.turns[old_turn_count:]
    if not new_turns:
        return 0
    return len(chunk_turns_with_context(new_turns))


def count_chunks(session: Session) -> int:
    """Count how many stage-1 LLM calls a session will need."""
    return len(chunk_turns_with_context(session.turns))


# --------------------------------------------------------------------------- #
# Trajectory-mode extraction (3-stage pipeline)                                #
# --------------------------------------------------------------------------- #
#
# Historical: a single LLM call was asked to "decompose this chunk into
# trajectories with EDUs". The LLM hallucinated turn ranges beyond the
# shown chunk, fabricating summaries for turns it had never seen.
#
# Current: three separate, tightly-bounded LLM calls per session:
#   1. Extract EDUs from turns (Sonnet, per chunk). Hard-bounded: only
#      source_turn_ids within shown turns are valid.
#   2. Classify topic boundaries pairwise (Haiku, per adjacent EDU pair).
#      Binary output — cannot hallucinate structure.
#   3. Label each trajectory with summary + keywords (Sonnet, per group).
#      Operates on concrete EDUs, not session reconstruction.
#
# Stage 2 uses a sliding window of BOUNDARY_WINDOW_SIZE EDUs centered on
# each pair — the candidate pair + extra EDUs as lookahead/lookback context.


BOUNDARY_SYSTEM_PROMPT = """\
You are a topic-segmentation classifier. Given a short window of consecutive EDUs (Elementary Discourse Units — atomic facts extracted from a conversation), decide whether the CANDIDATE EDU continues the same topic as the PREVIOUS EDU, or shifts to a meaningfully different topic.

Rules:
- "Same topic" means the two EDUs discuss the same subject, system, problem, or activity.
- Continuity examples: debugging → fixing the same bug; designing a feature → implementing it; asking a question → receiving the answer; exploring different facets of one system.
- Topic-shift examples: moving from audio config to git workflow; finishing a design discussion and starting a new build task; closing one debugging thread and starting another.
- A topic does NOT need to conclude before it can continue — ongoing exploration is still the same topic.
- Ignore surface differences (one EDU is a fact, next is a decision about that fact) if the underlying subject is the same.
- Focus on semantic topic, not author or EDU tag.

STRONG SIGNAL — turn provenance:
- If the PREVIOUS and CANDIDATE EDUs were derived from the SAME turn or from IMMEDIATELY ADJACENT turns in the original conversation, they almost certainly share a topic. A single long assistant turn typically covers one topic in depth, not a dozen unrelated ones. Default to "same_topic: true" for same-turn / adjacent-turn EDUs unless the content is obviously unrelated.
- If the PREVIOUS and CANDIDATE EDUs come from turns many positions apart with no turns between them represented, that is weak evidence of a shift but not conclusive.

Extra context EDUs in the window (marked "context") help you disambiguate — use them to understand what the conversation was about — but base your decision on the PREVIOUS/CANDIDATE pair.

Output: {"same_topic": true} if candidate continues previous's topic; {"same_topic": false} if it shifts."""


BOUNDARY_JSON_SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "same_topic": {"type": "boolean"},
    },
    "required": ["same_topic"],
})


LABEL_SYSTEM_PROMPT = """\
You are a trajectory labeler. Given a group of EDUs (Elementary Discourse Units) that ALL discuss a single topic, produce a concise summary and canonical keywords for that topic.

CRITICAL: Use ONLY the EDUs provided below. Do not invent content, do not extract from CLAUDE.md or other context you can see.

Requirements:
- `summary`: a single sentence describing what the trajectory was ABOUT — the subject matter discussed. Phrasings like "Discussed X", "Explored Y", "Debugged Z", "Decided W" are all fine. Past tense, action-oriented. This is the "chapter title" that will appear in a recent-activity index.
- `keywords`: 2-5 short, lowercase, canonical topic labels. PREFER reusing keywords from the provided project cloud; introduce a new keyword ONLY when no existing one fits. Keywords must be normalized — prefer common short forms (e.g. "pipewire", not "PipeWire" or "pipewire-1.0"). No spaces; use hyphens (e.g. "build-from-source"). Singular over plural.

Output: {"summary": "...", "keywords": ["...", "..."]}"""


LABEL_ONE_SHOT_INPUT = """\
Project: home
Existing project keyword cloud: audio, claude-memory, niri, pipewire, wayland

EDUs in this trajectory:
1. [gotcha] PipeWire's LV2 filter-chain module (libpipewire-module-filter-chain-lv2.so) was built from source because the Ubuntu PPA build did not include it
2. [config] PipeWire was built from source at ~/builds/pipewire/ using tag 1.0.5, and the LV2 module was installed to /usr/lib/x86_64-linux-gnu/pipewire-0.3/
3. [gotcha] If PipeWire is updated via apt, the custom-built LV2 module may need to be rebuilt from ~/builds/pipewire/"""


LABEL_ONE_SHOT_OUTPUT = json.dumps({
    "summary": "Built PipeWire's LV2 filter-chain module from source because the Ubuntu PPA build omitted it",
    "keywords": ["pipewire", "lv2", "build-from-source"],
})


LABEL_JSON_SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "keywords": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        },
    },
    "required": ["summary", "keywords"],
})


def _normalize_keyword(kw: str) -> str:
    """Lowercase, trim, collapse whitespace to hyphens."""
    return "-".join(kw.lower().strip().split())


def _format_edu_extraction_input(
    turns: list[Turn],
    session_date: str,
    project: str,
    core_turn_ids: set[int] | None = None,
) -> str:
    """Format turns into the stage-1 EDU-extraction prompt input.

    If core_turn_ids is provided, turns are grouped into CONTEXT (not for
    extraction) and CORE (extract EDUs from these) sections so the LLM can
    see surrounding conversation without being tempted to extract from it.

    The input is bracketed by explicit BEGIN/END markers with a random nonce
    so the LLM cannot mentally "continue past" the visible turns.
    """
    all_turn_ids = sorted({t.turn_id for t in turns})
    nonce = uuid.uuid4().hex[:8]

    lines = [f"=== BEGIN INPUT [nonce: {nonce}] ===",
             f"Session date: {session_date}",
             f"Project: {project}"]

    # Explicit turn-range bound: tell the LLM the valid citation range.
    # This reduces hallucination of fabricated "continuation" turns.
    if all_turn_ids:
        lines.append(
            f"Turns in this input range from {all_turn_ids[0]} to {all_turn_ids[-1]} "
            f"(total {len(all_turn_ids)}). Every source_turn_ids value MUST be within this range."
        )
        if core_turn_ids is not None:
            core_list = sorted(core_turn_ids)
            lines.append(
                f"Of those, CORE turns (the only ones you may extract EDUs from) are "
                f"{core_list[0]} to {core_list[-1]} (total {len(core_list)})."
            )
    lines.append("")

    if core_turn_ids is None or all(t.turn_id in core_turn_ids for t in turns):
        for t in turns:
            lines.append(f"Turn {t.turn_id} [{t.speaker}]: {t.text}")
    else:
        # Split turns into pre-context / core / post-context segments
        sections: list[tuple[str, list[Turn]]] = []
        current_section = None
        current_turns: list[Turn] = []
        for t in turns:
            section = "core" if t.turn_id in core_turn_ids else "context"
            if section != current_section:
                if current_turns:
                    sections.append((current_section, current_turns))
                current_section = section
                current_turns = [t]
            else:
                current_turns.append(t)
        if current_turns:
            sections.append((current_section, current_turns))

        for sec_name, sec_turns in sections:
            if sec_name == "context":
                lines.append("=== CONTEXT (for understanding only — do NOT extract EDUs from these turns) ===")
            else:
                lines.append("=== CORE (extract EDUs ONLY from these turns) ===")
            for t in sec_turns:
                lines.append(f"Turn {t.turn_id} [{t.speaker}]: {t.text}")
            lines.append("")

    # Unambiguous end sentinel. No turns exist past this marker; the nonce
    # makes the boundary token impossible to pattern-match from training data.
    if all_turn_ids:
        lines.append(
            f"=== END INPUT [nonce: {nonce}] — no turns exist beyond this point "
            f"(session had {len(all_turn_ids)} turns: {all_turn_ids[0]}-{all_turn_ids[-1]}) ==="
        )
    else:
        lines.append(f"=== END INPUT [nonce: {nonce}] ===")
    return "\n".join(lines)


def _format_boundary_input(
    window_edus: list[EDU],
    prev_rel_idx: int,
    curr_rel_idx: int,
) -> str:
    """Format a sliding-window boundary-classification prompt."""
    lines = ["EDUs in order (oldest at top):"]
    for i, edu in enumerate(window_edus):
        role = "context"
        if i == prev_rel_idx:
            role = "PREVIOUS"
        elif i == curr_rel_idx:
            role = "CANDIDATE"
        turns_str = ",".join(str(s) for s in edu.source_turn_ids) if edu.source_turn_ids else "?"
        lines.append(f"[{i}] ({role}) from turn {turns_str} [{edu.tag.value}] {edu.text}")
    lines.append("")
    prev_turns = set(window_edus[prev_rel_idx].source_turn_ids) if prev_rel_idx < len(window_edus) else set()
    curr_turns = set(window_edus[curr_rel_idx].source_turn_ids) if curr_rel_idx < len(window_edus) else set()
    if prev_turns and curr_turns:
        shared = prev_turns & curr_turns
        prev_max = max(prev_turns)
        curr_min = min(curr_turns)
        gap = curr_min - prev_max
        hints = []
        if shared:
            hints.append(f"PREVIOUS and CANDIDATE share source turn(s) {sorted(shared)} — very strong signal of same topic")
        elif gap <= 1:
            hints.append(f"PREVIOUS ends at turn {prev_max}, CANDIDATE starts at turn {curr_min} (adjacent) — strong signal of same topic")
        else:
            hints.append(f"PREVIOUS ends at turn {prev_max}, CANDIDATE starts at turn {curr_min} ({gap} turns apart)")
        lines.append("Turn-provenance hints:")
        for h in hints:
            lines.append(f"  - {h}")
    lines.append("")
    lines.append("Does the CANDIDATE EDU continue the same topic as the PREVIOUS EDU?")
    return "\n".join(lines)


def _format_label_input(
    edus: list[EDU],
    project: str,
    keyword_cloud: list[str],
) -> str:
    """Format the stage-3 labeler prompt input."""
    cloud_str = ", ".join(sorted(set(keyword_cloud))) if keyword_cloud else "(none yet — this is the first session in this project)"
    lines = [
        f"Project: {project}",
        f"Existing project keyword cloud: {cloud_str}",
        "",
        "EDUs in this trajectory:",
    ]
    for i, edu in enumerate(edus, 1):
        lines.append(f"{i}. [{edu.tag.value}] {edu.text}")
    return "\n".join(lines)


def _build_boundary_window(
    edus: list[EDU],
    candidate_idx: int,
    window_size: int,
) -> tuple[list[EDU], int, int]:
    """Return a sliding window around the boundary between candidate_idx-1 and candidate_idx.

    The returned tuple is (window_edus, prev_rel_idx, curr_rel_idx) where the
    relative indices point to the previous/candidate pair within window_edus.

    Extra EDUs (window_size - 2) fill the window as context, split floor/ceil
    between lookback (before the previous EDU) and lookahead (after the candidate).
    At list edges the window is truncated.
    """
    assert candidate_idx >= 1
    extras = max(0, window_size - 2)
    lookback = extras // 2
    lookahead = extras - lookback

    start = max(0, candidate_idx - 1 - lookback)
    end = min(len(edus), candidate_idx + 1 + lookahead)

    window_edus = edus[start:end]
    prev_rel = (candidate_idx - 1) - start
    curr_rel = candidate_idx - start
    return window_edus, prev_rel, curr_rel


def _parse_edus_from_chunk_result(
    result: dict,
    chunk: list[Turn],
    session: Session,
    seen_texts: set[str],
    core_turn_ids: set[int] | None = None,
) -> list[EDU]:
    """Turn a stage-1 LLM response into EDU objects; drop invalid / duplicate / out-of-range.

    If core_turn_ids is provided, EDUs must cite at least one turn in the CORE
    (context-only citations get clipped and drops if none remain).
    """
    window_turn_ids = {t.turn_id for t in chunk}
    # Valid citation targets: core if specified, else the whole window
    valid_ids = core_turn_ids if core_turn_ids is not None else window_turn_ids
    turn_by_id = {t.turn_id: t for t in chunk}
    edus: list[EDU] = []
    dropped_count = 0
    clipped_count = 0
    sample_drops: list[list[int]] = []
    sample_clips: list[list[int]] = []
    for raw in result.get("edus", []) or []:
        text = str(raw.get("text", "")).strip()
        if not text:
            continue
        normalized = text.lower()
        if normalized in seen_texts:
            continue

        source_ids_raw = raw.get("source_turn_ids", []) or []
        source_ids_all = [int(s) for s in source_ids_raw if isinstance(s, (int, float))]
        # Clip cited turns to the CORE (if specified) or the window. Keep the
        # EDU if any cited turn is valid; drop only if ALL citations were either
        # hallucinated past the window or only referenced context-only turns.
        source_ids = [s for s in source_ids_all if s in valid_ids]
        if not source_ids:
            dropped_count += 1
            if len(sample_drops) < 3:
                sample_drops.append(source_ids_all)
            continue
        if len(source_ids) < len(source_ids_all):
            clipped_count += 1
            if len(sample_clips) < 3:
                sample_clips.append([s for s in source_ids_all if s not in valid_ids])

        seen_texts.add(normalized)
        source_turns = [turn_by_id[s] for s in source_ids if s in turn_by_id]
        speakers = list({t.speaker for t in source_turns})
        ts = source_turns[0].timestamp if source_turns else chunk[0].timestamp

        edus.append(EDU(
            edu_id=str(uuid.uuid4()),
            text=text,
            source_turn_ids=source_ids,
            session_id=session.session_id,
            project=session.project,
            timestamp=ts,
            speakers=speakers,
            tag=EDUTag.coerce(raw.get("tag")),
        ))

    if dropped_count or clipped_count:
        core_range = f"{min(valid_ids)}-{max(valid_ids)}" if valid_ids else "empty"
        parts = []
        if dropped_count:
            parts.append(f"dropped {dropped_count} (sample cites: {sample_drops})")
        if clipped_count:
            parts.append(f"clipped context-only turns from {clipped_count} (sample: {sample_clips})")
        log.warning(
            f"Session {session.session_id[:8]} core={core_range}: {'; '.join(parts)}"
        )
    return edus


async def _stage1_extract_edus(
    turns: list[Turn],
    session: Session,
    model: str | None,
    on_chunk_done: callable,
) -> tuple[list[EDU], int]:
    """Stage 1: extract EDUs from turns, chunked. Returns (edus, chunks_failed).

    RateLimitError bubbles up so the caller can abort cleanly.
    """
    if not turns:
        return [], 0

    session_date = turns[0].timestamp.strftime("%Y-%m-%d")
    chunks = chunk_turns_with_context(turns)

    all_edus: list[EDU] = []
    seen_texts: set[str] = set()
    chunks_failed = 0

    for window_turns, core_ids in chunks:
        text = _format_edu_extraction_input(window_turns, session_date, session.project, core_ids)
        try:
            result = await call_claude(text, SYSTEM_PROMPT, model)
        except RateLimitError:
            raise
        except Exception as e:
            log.error(f"Stage-1 EDU extraction failed for session {session.session_id[:8]}: {e}")
            chunks_failed += 1
            if on_chunk_done:
                on_chunk_done()
            continue

        all_edus.extend(_parse_edus_from_chunk_result(result, window_turns, session, seen_texts, core_ids))
        if on_chunk_done:
            on_chunk_done()

    # Preserve chronological order via source_turn_ids (first turn id).
    all_edus.sort(key=lambda e: (min(e.source_turn_ids) if e.source_turn_ids else 0))
    return all_edus, chunks_failed


async def _stage2_classify_boundaries(
    edus: list[EDU],
    window_size: int = BOUNDARY_WINDOW_SIZE,
    concurrency: int = BOUNDARY_CONCURRENCY,
    model: str = BOUNDARY_MODEL,
) -> list[bool]:
    """Stage 2: for each adjacent EDU pair, classify same-topic yes/no.

    Returns a list of (N-1) booleans where decisions[i] is True iff EDU[i+1]
    continues the topic of EDU[i]. On failure for an individual pair, defaults
    to True (conservatively link rather than over-segment).

    RateLimitError bubbles up.
    """
    if len(edus) <= 1:
        return []

    sem = asyncio.Semaphore(concurrency)

    async def classify_one(i: int) -> bool:
        """Decide whether EDU[i] continues EDU[i-1]."""
        window, prev_rel, curr_rel = _build_boundary_window(edus, i, window_size)
        text = _format_boundary_input(window, prev_rel, curr_rel)
        async with sem:
            try:
                result = await call_claude(
                    text,
                    BOUNDARY_SYSTEM_PROMPT,
                    model,
                    json_schema=BOUNDARY_JSON_SCHEMA,
                    # Binary classification — no one-shot needed, prompt is self-contained
                    one_shot_input="",
                    one_shot_output="",
                )
            except RateLimitError:
                raise
            except Exception as e:
                log.warning(f"Boundary classification failed at pair ({i-1},{i}): {e} — defaulting to same topic")
                return True
        val = result.get("same_topic")
        if isinstance(val, bool):
            return val
        log.warning(f"Boundary classifier returned non-bool {val!r} at pair ({i-1},{i}) — defaulting to same topic")
        return True

    return await asyncio.gather(*[classify_one(i) for i in range(1, len(edus))])


def _segment_edus_by_boundaries(
    edus: list[EDU],
    same_topic: list[bool],
) -> list[list[EDU]]:
    """Group EDUs into trajectories given boundary decisions (length == len(edus) - 1)."""
    if not edus:
        return []
    groups: list[list[EDU]] = [[edus[0]]]
    for i in range(1, len(edus)):
        decision = same_topic[i - 1] if i - 1 < len(same_topic) else True
        if decision:
            groups[-1].append(edus[i])
        else:
            groups.append([edus[i]])
    return groups


async def _stage3_label_trajectory(
    trajectory_edus: list[EDU],
    project: str,
    keyword_cloud: list[str],
    model: str | None,
    sem: asyncio.Semaphore,
) -> tuple[str, list[str]]:
    """Stage 3: label a single trajectory (summary + keywords). Returns ('', []) on failure."""
    text = _format_label_input(trajectory_edus, project, keyword_cloud)
    async with sem:
        try:
            result = await call_claude(
                text,
                LABEL_SYSTEM_PROMPT,
                model,
                json_schema=LABEL_JSON_SCHEMA,
                one_shot_input=LABEL_ONE_SHOT_INPUT,
                one_shot_output=LABEL_ONE_SHOT_OUTPUT,
            )
        except RateLimitError:
            raise
        except Exception as e:
            log.warning(f"Labeling failed for trajectory of {len(trajectory_edus)} EDUs: {e}")
            return "", []
    summary = str(result.get("summary", "")).strip()
    raw_keywords = result.get("keywords", []) or []
    keywords: list[str] = []
    for kw in raw_keywords:
        if not isinstance(kw, str):
            continue
        norm = _normalize_keyword(kw)
        if norm and norm not in keywords:
            keywords.append(norm)
    return summary, keywords


async def extract_trajectories_from_session(
    session: Session,
    keyword_cloud: list[str],
    existing_trajectories: list | None = None,
    new_turn_start: int = 0,
    model: str | None = None,
    on_chunk_done: callable = None,
):
    """Extract topic trajectories (with attached EDUs) from a session.

    Pipeline (3 stages):
      1. Extract EDUs from turns (Sonnet, chunked).
      2. Classify same-topic boundaries pairwise with sliding window (Haiku).
      3. Label each trajectory with summary + keywords (Sonnet).

    Args:
        session: the full Session.
        keyword_cloud: current canonical keywords for this project (pre-ingest snapshot).
        existing_trajectories: not used by this pipeline (kept for signature
            compatibility). Incremental mode still re-extracts from new_turn_start.
        new_turn_start: index into session.turns; extract from this turn onward.
        model: override model for Sonnet stages (1 and 3). Stage 2 always uses
            BOUNDARY_MODEL regardless of this parameter.
        on_chunk_done: progress callback, fired per stage-1 chunk.

    Returns:
        (results, chunks_failed): list of (Trajectory, [EDU]) tuples plus the
        count of stage-1 chunks that errored. On any stage-1 failure (count > 0),
        stages 2 and 3 are skipped and ([], chunks_failed) is returned so the
        session is retried on the next ingest. On RateLimitError (any stage),
        the exception bubbles up so the caller can abort cleanly.
    """
    from .trajectories import Trajectory

    if not session.turns:
        return [], 0

    turns_to_process = session.turns[new_turn_start:]
    if not turns_to_process:
        return [], 0

    # Stage 1: extract EDUs
    edus, chunks_failed = await _stage1_extract_edus(
        turns_to_process, session, model, on_chunk_done
    )
    if chunks_failed > 0:
        return [], chunks_failed
    if not edus:
        return [], 0

    # Stage 2: pairwise boundary classification
    same_topic = await _stage2_classify_boundaries(edus)

    # Segment EDUs into trajectory groups (pure function)
    groups = _segment_edus_by_boundaries(edus, same_topic)

    # Stage 3: label each trajectory in parallel
    label_sem = asyncio.Semaphore(BOUNDARY_CONCURRENCY)
    labels = await asyncio.gather(*[
        _stage3_label_trajectory(g, session.project, keyword_cloud, model, label_sem)
        for g in groups
    ])

    # Assemble Trajectory records
    results: list[tuple[Trajectory, list[EDU]]] = []
    for group, (summary, keywords) in zip(groups, labels):
        if not summary:
            summary = f"(unlabeled trajectory of {len(group)} EDUs)"
        if not keywords:
            keywords = ["uncategorized"]

        start_turn = min(min(e.source_turn_ids) for e in group if e.source_turn_ids)
        end_turn = max(max(e.source_turn_ids) for e in group if e.source_turn_ids)
        start_ts = min(e.timestamp for e in group)
        traj_id = str(uuid.uuid4())
        trajectory = Trajectory(
            id=traj_id,
            session_id=session.session_id,
            project=session.project,
            start_turn=start_turn,
            end_turn=end_turn,
            summary=summary,
            created_at=start_ts,
            keywords=keywords,
        )
        # Attach trajectory_id + trajectory_index to each EDU for downstream metadata
        for idx, edu in enumerate(group):
            edu.trajectory_id = traj_id
            edu.trajectory_index = idx

        results.append((trajectory, group))

    return results, 0


async def extract_edus_from_session(
    session: Session,
    model: str | None = None,
    on_chunk_done: callable = None,
) -> list[EDU]:
    """Extract EDUs from a single session."""
    if not session.turns:
        return []

    session_date = session.turns[0].timestamp.strftime("%Y-%m-%d")
    chunks = chunk_turns(session.turns)

    all_edus: list[EDU] = []
    seen_texts: set[str] = set()

    for chunk in chunks:
        text = format_turns_for_extraction(chunk, session_date, session.project)
        try:
            result = await call_claude(text, SYSTEM_PROMPT, model)
        except Exception as e:
            log.error(f"LLM call failed for session {session.session_id}: {e}")
            if on_chunk_done:
                on_chunk_done()
            continue

        raw_edus = result.get("edus", [])
        for raw in raw_edus:
            edu_text = raw.get("text", "").strip()
            if not edu_text:
                continue
            # Deduplicate across chunks (overlap can produce duplicates)
            normalized = edu_text.lower()
            if normalized in seen_texts:
                continue
            seen_texts.add(normalized)

            source_ids = raw.get("source_turn_ids", [])
            # Figure out speakers from source turns
            speakers = list({
                t.speaker for t in chunk
                if t.turn_id in source_ids
            })
            # Timestamp from earliest source turn
            source_turns = [t for t in chunk if t.turn_id in source_ids]
            ts = source_turns[0].timestamp if source_turns else session.turns[0].timestamp

            all_edus.append(EDU(
                edu_id=str(uuid.uuid4()),
                text=edu_text,
                source_turn_ids=source_ids,
                session_id=session.session_id,
                project=session.project,
                timestamp=ts,
                speakers=speakers,
                tag=EDUTag.coerce(raw.get("tag")),
            ))

        if on_chunk_done:
            on_chunk_done()

    return all_edus
