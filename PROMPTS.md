# Claude Memory System — Prompts

All LLM prompts used by the system. Adapted from EMem (arXiv:2511.17208), Memoria (arXiv:2512.12686), and A-Mem (arXiv:2502.12110).

---

## 1. EDU Extraction Prompt

Used during ingestion to decompose conversations into atomic facts.

### System Prompt

```
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

Output as JSON: {"edus": [{"text": "...", "source_turn_ids": [1, 2]}, ...]}
```

### One-Shot Example

**User message (input):**
```
Session date: 2026-03-20
Project: /home/duane

Turn 1 [user]: EasyEffects is messing up my audio routing. Everything is going through its virtual sink even when I don't want it to
Turn 2 [assistant]: I can see EasyEffects has inserted itself as the default sink. Let me check what's routing through it. Looking at wpctl status, all your playback streams are being captured by the EasyEffects sink. Since you mentioned you only use it for mic processing, we could replace it with a native PipeWire filter-chain that only processes the mic input without touching playback routing.
Turn 3 [user]: yeah do that. rip it out
Turn 4 [assistant]: Done. I've removed EasyEffects and created a PipeWire filter-chain config at ~/.config/pipewire/pipewire.conf.d/20-mic-chain.conf. The chain is: Yeti Nano → highpass 80Hz → LSP compressor (-26dB threshold, 4:1) → LSP de-esser (-24dB, 6:1, SC HPF 4kHz/LPF 9kHz) → LSP limiter (-3dB). The processed mic appears as "mic_chain.output". Your playback now goes directly to HDMI without any intermediary.
Turn 5 [user]: what about the LV2 plugins, does pipewire support those natively?
Turn 6 [assistant]: PipeWire has an LV2 module but the Ubuntu PPA build doesn't include it. I built it from source — cloned PipeWire tag 1.0.5 to ~/builds/pipewire/ and compiled libpipewire-module-filter-chain-lv2.so. Installed it to /usr/lib/x86_64-linux-gnu/pipewire-0.3/. Note: if PipeWire gets updated via apt, this custom module may need rebuilding.
```

**Expected output:**
```json
{"edus": [
  {"text": "EasyEffects was removed from Duane's system on 2026-03-20 because it was hijacking all playback audio into its virtual sink, breaking audio routing", "source_turn_ids": [1, 3]},
  {"text": "EasyEffects was replaced with a native PipeWire filter-chain that only processes mic input without affecting playback routing", "source_turn_ids": [2, 3]},
  {"text": "The PipeWire mic processing filter-chain config is located at ~/.config/pipewire/pipewire.conf.d/20-mic-chain.conf", "source_turn_ids": [4]},
  {"text": "The PipeWire mic chain order is: Yeti Nano → highpass 80Hz (2x biquad) → LSP compressor (leveling, -26dB threshold, 4:1 ratio) → LSP compressor (de-esser, -24dB threshold, 6:1 ratio, sidechain HPF 4kHz/LPF 9kHz) → LSP limiter (Herm Thin, -3dB)", "source_turn_ids": [4]},
  {"text": "The processed mic output appears as 'mic_chain.output' (named 'Processed Mic') in PipeWire", "source_turn_ids": [4]},
  {"text": "After removing EasyEffects, playback audio routes directly to HDMI output (Navi 31) without any intermediary sink", "source_turn_ids": [4]},
  {"text": "PipeWire's LV2 filter-chain module (libpipewire-module-filter-chain-lv2.so) was built from source because the Ubuntu PPA build did not include it", "source_turn_ids": [5, 6]},
  {"text": "PipeWire was built from source at ~/builds/pipewire/ using tag 1.0.5, and the LV2 module was installed to /usr/lib/x86_64-linux-gnu/pipewire-0.3/", "source_turn_ids": [6]},
  {"text": "If PipeWire is updated via apt, the custom-built LV2 module may need to be rebuilt from ~/builds/pipewire/", "source_turn_ids": [6]}
]}
```

---

## 2. EDU Relevance Filter Prompt

Used at query time to prune retrieved candidates. Adapted from EMem's filter prompt.

### System Prompt

```
You are a memory relevance filter for a conversational memory system. Given a user's query and a list of candidate memory facts (EDUs), select ALL facts that could be relevant to answering the query.

Rules:
- Be MAXIMALLY INCLUSIVE — err on the side of keeping too many rather than too few
- Keep facts with embedded information that might indirectly answer the query
- Keep facts with temporal context that helps establish when things happened
- Keep facts with quantitative data (versions, sizes, ports, paths)
- Keep facts that mention entities related to the query, even tangentially
- Keep facts that support multi-hop reasoning (A relates to B, B relates to query)
- Keep historical context that helps understand current state
- Prefer false positives over false negatives — it's cheap for the caller to ignore irrelevant facts, but expensive to miss relevant ones

Output the selected facts as JSON: {"selected": [0, 3, 7, ...]} using the index numbers of the candidate facts.
```

### Example

```
Query: "What audio issues have we dealt with?"

Candidate facts:
[0] EasyEffects was removed from Duane's system on 2026-03-20 because it was hijacking all playback audio
[1] Duane's Niri compositor config is at ~/.config/niri/config.kdl
[2] DeepFilterNet was removed from the PipeWire mic chain on 2026-03-23 because it crashes under CPU load
[3] PipeWire quantum is locked at 2048 because the HDMI driver drops to 1024 if min-quantum is lower, causing xruns
[4] The primesignal project uses a master branch
[5] spotify_player outputs at 44100 Hz which gets resampled to 48000 Hz by PipeWire
[6] Duane set up DHCP reservation for thor at 192.168.1.204

Selected: {"selected": [0, 2, 3, 5]}
```

---

## 3. Session Summary Prompt (optional, for session-level index)

Used to generate a brief topic summary per session, enabling coarse-grained browsing.

```
Given the following extracted facts (EDUs) from a single conversation session, write a 1-2 sentence summary of what was discussed. Focus on the topics and outcomes, not the process.

EDUs:
${edu_list}

Summary:
```

---

## 4. Query Expansion Prompt (optional, for ambiguous queries)

If the initial query is too vague, expand it into multiple specific sub-queries.

```
The user is searching their conversation history with this query: "${query}"

Generate 2-3 specific sub-queries that would help find relevant information. Each sub-query should target a different angle or aspect of the original query.

Output as JSON: {"queries": ["...", "...", "..."]}
```

---

## Prompt Design Principles (from papers)

### From EMem (arXiv:2511.17208)
- One-shot prompting with a worked example outperforms zero-shot for EDU extraction
- EDUs should be "minimal yet complete in meaning" — atomic but self-contained
- "Consistently use the most informative name for each entity in all EDUs"
- Source turn attribution enables traceability back to original conversation

### From Memoria (arXiv:2512.12686)
- Extract from user messages primarily — they contain intent and preferences
- Recency weighting via exponential decay: `w = e^(-α * minutes_elapsed)`
- Normalize weights to prevent total suppression of old memories

### From A-Mem (arXiv:2502.12110)
- Extract keywords and categorical tags alongside the fact itself
- Link new memories to existing ones by semantic similarity
- Evolution actions: strengthen (reinforce existing), update_neighbor (propagate changes)

### From the survey papers
- "Visible ≠ usable" — models can't use information buried mid-context (Lost in the Middle)
- Structure beats embedding quality — organized retrieval >> better vectors
- Hybrid retrieval (dense + metadata + temporal) consistently outperforms single-signal
