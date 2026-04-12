"""Extract Elementary Discourse Units (EDUs) from conversation turns via LLM."""

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime

import httpx

from .config import (
    CHUNK_OVERLAP_TURNS,
    LLM_BASE_URL,
    MAX_TURNS_PER_CHUNK,
)
from .parser import Session, Turn

log = logging.getLogger(__name__)

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

Output as JSON: {"edus": [{"text": "...", "source_turn_ids": [1, 2]}, ...]}"""

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
    {"text": "EasyEffects was removed from Duane's system on 2026-03-20 because it was hijacking all playback audio into its virtual sink, breaking audio routing", "source_turn_ids": [1, 3]},
    {"text": "EasyEffects was replaced with a native PipeWire filter-chain that only processes mic input without affecting playback routing", "source_turn_ids": [2, 3]},
    {"text": "The PipeWire mic processing filter-chain config is located at ~/.config/pipewire/pipewire.conf.d/20-mic-chain.conf", "source_turn_ids": [4]},
    {"text": "The PipeWire mic chain order is: Yeti Nano → highpass 80Hz (2x biquad) → LSP compressor (leveling, -26dB threshold, 4:1 ratio) → LSP compressor (de-esser, -24dB threshold, 6:1 ratio, sidechain HPF 4kHz/LPF 9kHz) → LSP limiter (Herm Thin, -3dB)", "source_turn_ids": [4]},
    {"text": "The processed mic output appears as 'mic_chain.output' (named 'Processed Mic') in PipeWire", "source_turn_ids": [4]},
    {"text": "After removing EasyEffects, playback audio routes directly to HDMI output (Navi 31) without any intermediary sink", "source_turn_ids": [4]},
    {"text": "PipeWire's LV2 filter-chain module (libpipewire-module-filter-chain-lv2.so) was built from source because the Ubuntu PPA build did not include it", "source_turn_ids": [5, 6]},
    {"text": "PipeWire was built from source at ~/builds/pipewire/ using tag 1.0.5, and the LV2 module was installed to /usr/lib/x86_64-linux-gnu/pipewire-0.3/", "source_turn_ids": [6]},
    {"text": "If PipeWire is updated via apt, the custom-built LV2 module may need to be rebuilt from ~/builds/pipewire/", "source_turn_ids": [6]},
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


async def call_llm(text: str, base_url: str | None = None) -> dict:
    """Call the local LLM with the EDU extraction prompt."""
    url = (base_url or LLM_BASE_URL).rstrip("/") + "/v1/chat/completions"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": ONE_SHOT_INPUT},
        {"role": "assistant", "content": ONE_SHOT_OUTPUT},
        {"role": "user", "content": text},
    ]

    async with httpx.AsyncClient(timeout=600.0) as client:
        resp = await client.post(url, json={
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 16384,
        })
        resp.raise_for_status()
        data = resp.json()

    msg = data["choices"][0]["message"]
    content = msg.get("content") or ""
    # Strip <think>...</think> blocks if present (Qwen thinking mode)
    import re
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    if not content:
        raise ValueError(f"LLM returned empty content. Keys: {list(msg.keys())}")
    return json.loads(content)


def count_chunks(session: Session) -> int:
    """Count how many LLM calls a session will need."""
    return len(chunk_turns(session.turns))


async def extract_edus_from_session(
    session: Session,
    base_url: str | None = None,
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
            result = await call_llm(text, base_url)
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
            ))

        if on_chunk_done:
            on_chunk_done()

    return all_edus
