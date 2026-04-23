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
    """Split long sessions into overlapping chunks."""
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

    prompt = f"""Here is an example of the expected input and output:

INPUT:
{example_in}

OUTPUT:
{example_out}

Now process this input:

{text}"""

    t_start = time.monotonic()
    proc = await asyncio.create_subprocess_exec(
        "claude", "-p",
        "--model", model,
        "--tools", "",
        "--system-prompt", system_prompt,
        "--json-schema", schema,
        "--output-format", "json",
        "--no-session-persistence",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate(prompt.encode())
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
    """Count chunks for incremental extraction."""
    new_turns = session.turns[old_turn_count:]
    if not new_turns:
        return 0
    return len(chunk_turns(new_turns))


def count_chunks(session: Session) -> int:
    """Count how many LLM calls a session will need."""
    return len(chunk_turns(session.turns))


# --------------------------------------------------------------------------- #
# Trajectory-mode extraction                                                  #
# --------------------------------------------------------------------------- #

TRAJECTORY_SYSTEM_PROMPT = """\
You are a memory extraction system. Given a conversation session between a user and an AI assistant (Claude Code), decompose it into TOPIC TRAJECTORIES — contiguous ranges of turns covering a single coherent topic — and extract Elementary Discourse Units (EDUs) from each.

A trajectory is one topic discussed across a block of adjacent turns. All exchanges within the trajectory must be on the same topic. When the conversation shifts to a meaningfully different topic, start a new trajectory. A session may contain one or many trajectories.

A trajectory does NOT need to reach a conclusion, a decision, or a completed action. Open-ended discussion, brainstorming, exploration, design thinking, questions without definitive answers — these are all valid trajectories as long as the exchanges are coherent on one topic. Even a single user question and assistant reply is a trajectory if there is any substantive content (a fact, detail, consideration, or idea worth remembering).

For each trajectory, produce:
- `start_turn`, `end_turn`: the contiguous turn range (inclusive). Every substantive turn in that range belongs to this trajectory. Conversational filler at the boundary MAY be excluded.
- `summary`: a single sentence describing what the trajectory was ABOUT — the subject matter discussed. Do not require a completed action; "Discussed X" or "Explored approaches for Y" is fine when no decision was reached. This is the "chapter title" — it will be surfaced in a recent-activity index.
- `keywords`: 2-5 short, lowercase, canonical topic labels. PREFER reusing keywords from the provided project cloud; introduce a new keyword ONLY when no existing one fits. Keywords must be normalized — prefer common short forms (e.g. "pipewire", not "PipeWire" or "pipewire-1.0"). No spaces; use hyphens (e.g. "build-from-source"). Singular over plural.
- `edus`: Elementary Discourse Units extracted from the trajectory's turns.

Trajectories together should cover all substantive turns in the session. Gaps of pure filler between trajectories are fine. If the session contains any substantive content at all, you MUST return at least one trajectory — only return an empty trajectory list when the session is literally 100% filler (e.g. only `/clear` commands, session boilerplate, or acknowledgements with no information).

Requirements for EDUs:
1. Each EDU must be independently understandable without any other EDU or the original conversation for context.
2. Replace pronouns and ambiguous references with specific names, tools, paths, or details (e.g. "PipeWire filter-chain" not "it").
3. Preserve ALL substantive information — no detail should be lost. Technical details (paths, configs, commands, parameters) are especially important.
4. Infer and include temporal context. Convert relative dates to absolute where possible.
5. One fact per EDU — split distinct facts into distinct EDUs.
6. Skip conversational filler ("sure", "let me check"). Extract only substantive content.
7. `source_turn_ids` must cite the turn numbers the EDU was derived from (within the trajectory's range).
8. Assign each EDU exactly one `tag` from this fixed set:
   - "decision": architectural/design choices and their rationale
   - "preference": user preferences, communication style, workflow conventions
   - "gotcha": bugs, surprises, non-obvious caveats, brittle workarounds
   - "config": concrete config values, paths, ports, flags, command invocations
   - "architecture": how systems are structured, data flow, component relationships
   - "project": project status, goals, ongoing work

Output JSON shape: {"trajectories": [{"start_turn": int, "end_turn": int, "summary": "...", "keywords": ["...", "..."], "edus": [{"text": "...", "source_turn_ids": [n, ...], "tag": "..."}, ...]}, ...]}"""


TRAJECTORY_ONE_SHOT_INPUT = """\
Session date: 2026-03-20
Project: home
Existing project keyword cloud: audio, claude-memory, niri, pipewire, wayland

Turn 1 [user]: EasyEffects is messing up my audio routing. Everything is going through its virtual sink even when I don't want it to
Turn 2 [assistant]: I can see EasyEffects has inserted itself as the default sink. Let me check what's routing through it. Looking at wpctl status, all your playback streams are being captured by the EasyEffects sink. Since you mentioned you only use it for mic processing, we could replace it with a native PipeWire filter-chain that only processes the mic input without touching playback routing.
Turn 3 [user]: yeah do that. rip it out
Turn 4 [assistant]: Done. I've removed EasyEffects and created a PipeWire filter-chain config at ~/.config/pipewire/pipewire.conf.d/20-mic-chain.conf. The chain is: Yeti Nano → highpass 80Hz → LSP compressor (-26dB threshold, 4:1) → LSP de-esser (-24dB, 6:1, SC HPF 4kHz/LPF 9kHz) → LSP limiter (-3dB). The processed mic appears as "mic_chain.output". Your playback now goes directly to HDMI without any intermediary.
Turn 5 [user]: what about the LV2 plugins, does pipewire support those natively?
Turn 6 [assistant]: PipeWire has an LV2 module but the Ubuntu PPA build doesn't include it. I built it from source — cloned PipeWire tag 1.0.5 to ~/builds/pipewire/ and compiled libpipewire-module-filter-chain-lv2.so. Installed it to /usr/lib/x86_64-linux-gnu/pipewire-0.3/. Note: if PipeWire gets updated via apt, this custom module may need rebuilding."""


TRAJECTORY_ONE_SHOT_OUTPUT = json.dumps({"trajectories": [
    {
        "start_turn": 1,
        "end_turn": 4,
        "summary": "Replaced EasyEffects with a native PipeWire filter-chain for mic processing, fixing playback routing",
        "keywords": ["pipewire", "audio", "easyeffects"],
        "edus": [
            {"text": "EasyEffects was removed from Duane's system on 2026-03-20 because it was hijacking all playback audio into its virtual sink, breaking audio routing", "source_turn_ids": [1, 3], "tag": "decision"},
            {"text": "EasyEffects was replaced with a native PipeWire filter-chain that only processes mic input without affecting playback routing", "source_turn_ids": [2, 3], "tag": "decision"},
            {"text": "The PipeWire mic processing filter-chain config is located at ~/.config/pipewire/pipewire.conf.d/20-mic-chain.conf", "source_turn_ids": [4], "tag": "config"},
            {"text": "The PipeWire mic chain order is: Yeti Nano → highpass 80Hz → LSP compressor (leveling, -26dB threshold, 4:1 ratio) → LSP compressor (de-esser, -24dB threshold, 6:1 ratio, sidechain HPF 4kHz/LPF 9kHz) → LSP limiter (Herm Thin, -3dB)", "source_turn_ids": [4], "tag": "architecture"},
            {"text": "The processed mic output appears as 'mic_chain.output' (named 'Processed Mic') in PipeWire", "source_turn_ids": [4], "tag": "config"},
            {"text": "After removing EasyEffects, playback audio routes directly to HDMI output (Navi 31) without any intermediary sink", "source_turn_ids": [4], "tag": "architecture"}
        ]
    },
    {
        "start_turn": 5,
        "end_turn": 6,
        "summary": "Built PipeWire's LV2 filter-chain module from source because the Ubuntu PPA build omitted it",
        "keywords": ["pipewire", "lv2", "build-from-source"],
        "edus": [
            {"text": "PipeWire's LV2 filter-chain module (libpipewire-module-filter-chain-lv2.so) was built from source because the Ubuntu PPA build did not include it", "source_turn_ids": [5, 6], "tag": "gotcha"},
            {"text": "PipeWire was built from source at ~/builds/pipewire/ using tag 1.0.5, and the LV2 module was installed to /usr/lib/x86_64-linux-gnu/pipewire-0.3/", "source_turn_ids": [6], "tag": "config"},
            {"text": "If PipeWire is updated via apt, the custom-built LV2 module may need to be rebuilt from ~/builds/pipewire/", "source_turn_ids": [6], "tag": "gotcha"}
        ]
    }
]})


TRAJECTORY_JSON_SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "trajectories": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start_turn": {"type": "integer"},
                    "end_turn": {"type": "integer"},
                    "summary": {"type": "string"},
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
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
                "required": ["start_turn", "end_turn", "summary", "keywords", "edus"],
            },
        },
    },
    "required": ["trajectories"],
})


def _normalize_keyword(kw: str) -> str:
    """Lowercase, trim, collapse whitespace to hyphens."""
    return "-".join(kw.lower().strip().split())


def format_trajectory_input(
    turns: list[Turn],
    session_date: str,
    project: str,
    keyword_cloud: list[str],
    existing_trajectories: list | None = None,
) -> str:
    """Format a chunk of turns + context for trajectory extraction."""
    lines = [f"Session date: {session_date}", f"Project: {project}"]
    cloud_str = ", ".join(sorted(set(keyword_cloud))) if keyword_cloud else "(none yet — this is the first session in this project)"
    lines.append(f"Existing project keyword cloud: {cloud_str}")

    if existing_trajectories:
        lines.append("")
        lines.append("Existing trajectories already extracted from earlier turns of this session:")
        for t in existing_trajectories:
            kw = ", ".join(t.keywords) if t.keywords else "(none)"
            lines.append(f"  - [turns {t.start_turn}-{t.end_turn}] {t.summary}  [keywords: {kw}]")
        lines.append("Extract trajectories ONLY from the new turns below. If the new turns continue a topic already captured above, still return a new trajectory for the new turns (the downstream pipeline will link them by shared keywords).")

    lines.append("")
    for t in turns:
        lines.append(f"Turn {t.turn_id} [{t.speaker}]: {t.text}")
    return "\n".join(lines)


async def extract_trajectories_from_session(
    session: Session,
    keyword_cloud: list[str],
    existing_trajectories: list | None = None,
    new_turn_start: int = 0,
    model: str | None = None,
    on_chunk_done: callable = None,
):
    """Extract topic trajectories (with attached EDUs) from a session.

    Args:
        session: the full Session.
        keyword_cloud: current canonical keywords for this project (pre-ingest snapshot).
        existing_trajectories: Trajectory records already extracted from earlier
            turns of the same session (incremental mode). Their summaries and
            keywords are shown to the LLM as context.
        new_turn_start: index into session.turns. Turns at or after this index
            are the "new" turns to extract from. 0 means extract the whole
            session. Used for incremental ingestion.
        model: override the extraction model.
        on_chunk_done: progress callback.

    Returns:
        (results, chunks_failed): results is a list of (Trajectory, [EDU]) tuples
        in the order the LLM produced them. chunks_failed is the count of chunks
        whose extraction threw a non-rate-limit error (those chunks' content is
        missing from results). On RateLimitError, the exception bubbles up so the
        caller can abort the whole ingest for later resume.
    """
    # Deferred imports to avoid cycles
    from .trajectories import Trajectory

    if not session.turns:
        return [], 0

    existing_trajectories = existing_trajectories or []
    turns_to_process = session.turns[new_turn_start:]
    if not turns_to_process:
        return [], 0

    session_date = session.turns[0].timestamp.strftime("%Y-%m-%d")
    chunks = chunk_turns(turns_to_process)

    results: list[tuple[Trajectory, list[EDU]]] = []
    seen_edu_texts: set[str] = set()
    chunks_failed = 0

    for chunk in chunks:
        text = format_trajectory_input(
            chunk, session_date, session.project, keyword_cloud, existing_trajectories
        )
        try:
            result = await call_claude(
                text,
                TRAJECTORY_SYSTEM_PROMPT,
                model,
                json_schema=TRAJECTORY_JSON_SCHEMA,
                one_shot_input=TRAJECTORY_ONE_SHOT_INPUT,
                one_shot_output=TRAJECTORY_ONE_SHOT_OUTPUT,
            )
        except RateLimitError:
            # Bubble up so ingest can abort cleanly for later resume
            raise
        except Exception as e:
            log.error(f"Trajectory extraction failed for session {session.session_id}: {e}")
            chunks_failed += 1
            if on_chunk_done:
                on_chunk_done()
            continue

        for raw_traj in result.get("trajectories", []):
            try:
                start_turn = int(raw_traj["start_turn"])
                end_turn = int(raw_traj["end_turn"])
            except (KeyError, ValueError, TypeError) as e:
                log.warning(f"Skipping trajectory with invalid turn range: {e}")
                continue

            summary = str(raw_traj.get("summary", "")).strip()
            if not summary:
                log.warning(f"Skipping trajectory with empty summary (turns {start_turn}-{end_turn})")
                continue

            raw_keywords = raw_traj.get("keywords", []) or []
            keywords = []
            for kw in raw_keywords:
                if not isinstance(kw, str):
                    continue
                norm = _normalize_keyword(kw)
                if norm and norm not in keywords:
                    keywords.append(norm)
            if not keywords:
                log.warning(f"Trajectory turns {start_turn}-{end_turn} has no keywords; using ['uncategorized']")
                keywords = ["uncategorized"]

            # Timestamp of the trajectory = timestamp of its start turn
            start_ts = next(
                (t.timestamp for t in chunk if t.turn_id == start_turn),
                chunk[0].timestamp,
            )

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

            # Build EDUs for this trajectory
            edus: list[EDU] = []
            for raw_edu in raw_traj.get("edus", []) or []:
                edu_text = str(raw_edu.get("text", "")).strip()
                if not edu_text:
                    continue
                normalized = edu_text.lower()
                if normalized in seen_edu_texts:
                    continue
                seen_edu_texts.add(normalized)

                source_ids = raw_edu.get("source_turn_ids", []) or []
                speakers = list({t.speaker for t in chunk if t.turn_id in source_ids})
                source_turns = [t for t in chunk if t.turn_id in source_ids]
                ts = source_turns[0].timestamp if source_turns else start_ts

                edus.append(EDU(
                    edu_id=str(uuid.uuid4()),
                    text=edu_text,
                    source_turn_ids=source_ids,
                    session_id=session.session_id,
                    project=session.project,
                    timestamp=ts,
                    speakers=speakers,
                    tag=EDUTag.coerce(raw_edu.get("tag")),
                    trajectory_id=traj_id,
                    trajectory_index=len(edus),
                ))

            if not edus:
                log.warning(f"Trajectory turns {start_turn}-{end_turn} produced no EDUs; keeping trajectory record anyway")

            results.append((trajectory, edus))

        if on_chunk_done:
            on_chunk_done()

    return results, chunks_failed


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
