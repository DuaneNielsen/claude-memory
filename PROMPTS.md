# Claude Memory System — Prompts

Every LLM prompt the system uses, plus prompts from the research it draws on but does not yet implement. Designs adapted from EMem ([arXiv:2511.17208](https://arxiv.org/abs/2511.17208)), Memoria ([arXiv:2512.12686](https://arxiv.org/abs/2512.12686)), and A-Mem ([arXiv:2502.12110](https://arxiv.org/abs/2502.12110)).

> ⚙️ **Source of truth:** all production prompts live in `src/claude_memory/extractor.py`. The strings below are pulled from that file — if you edit one, edit the other in the same change so this doc doesn't drift.

---

## Part 1 — Production prompts (in active use)

The ingestion pipeline runs in three stages. Stage 1 extracts EDUs from raw turns. Stage 2 segments those EDUs into topical trajectories by classifying adjacent-pair boundaries. Stage 3 labels each trajectory. There are four prompts total — one per stage, plus a separate incremental variant of stage 1 for resumed sessions.

### 1.1 — Initial EDU extraction (`SYSTEM_PROMPT`)

Used the first time a session is ingested. Sees a chunk of conversation turns (with optional adjacent-turn context margins) and emits atomic facts.

`extractor.py: SYSTEM_PROMPT` (line 47)

```
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

Output as JSON: {"edus": [{"text": "...", "source_turn_ids": [1, 2], "tag": "config"}, ...]}
```

**Why this shape:**
- The "CRITICAL SOURCE BOUNDARY" stanza was added after the model started fabricating EDUs sourced from `~/CLAUDE.md` (visible to the cwd-traversing `claude` CLI). The fix is two-pronged: this prompt rule, plus running the subprocess from `cwd="/tmp"` so there's no CLAUDE.md to find. See `extractor.py:257`.
- "CORE vs CONTEXT" supports overlap-free chunking with adjacent-turn lookahead/lookback. The model sees the surrounding turns for context but is forbidden from extracting facts from them — that role belongs to the chunk where they're CORE.
- Tags are a closed set so downstream filters (`tag in {"decision", "gotcha"}`) work without normalization.

#### One-shot example

`extractor.py: ONE_SHOT_INPUT / ONE_SHOT_OUTPUT` (line 79). The example is paste-prepended to every extraction call by `call_claude`:

```
Here is an example of the expected input and output:

INPUT:
<ONE_SHOT_INPUT>

OUTPUT:
<ONE_SHOT_OUTPUT>

Now process this input:

<actual chunk>
```

The example uses a real conversation about removing EasyEffects + replacing it with a PipeWire filter-chain. It's deliberately rich in concrete detail (paths, version numbers, parameter values) so the model learns to preserve that level of specificity.

### 1.2 — Incremental EDU extraction (`INCREMENTAL_SYSTEM_PROMPT`)

Used when a session is *resumed* — the JSONL file has grown since the last ingest run. Rather than re-extract from scratch, this variant feeds the model the previously-extracted EDUs as context and asks it to extract only what's *new*.

`extractor.py: INCREMENTAL_SYSTEM_PROMPT` (line 325)

```
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
If there are no new facts, output: {"edus": []}
```

**Why this shape:**
- Items 1-7 mirror the initial prompt so the output schema is identical and downstream code doesn't branch.
- Item 8 is the key behavioral difference — without it the model re-emits ~30% of the existing EDUs in slightly different words, polluting the store with near-duplicates.
- The empty-output sentinel at the end (`{"edus": []}`) is explicit because the model otherwise tends to manufacture filler when nothing genuinely new exists.

### 1.3 — Trajectory boundary detection (`BOUNDARY_SYSTEM_PROMPT`)

Used by stage 2 of the trajectory pipeline. After EDUs are extracted, the system needs to group them into "trajectories" — coherent runs of EDUs that discuss the same topic. Rather than ask one big LLM call to do the segmentation (which hallucinated trajectory ranges in early prototypes), the pipeline classifies each *adjacent EDU pair* with a binary same-topic / different-topic decision and links runs of `same_topic=true` into trajectories.

`extractor.py: BOUNDARY_SYSTEM_PROMPT` (line 472)

```
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

Output: {"same_topic": true} if candidate continues previous's topic; {"same_topic": false} if it shifts.
```

**Why this shape:**
- Binary output cannot hallucinate structure. The earlier prototype that asked for "list of trajectories with EDU ranges" emitted ranges past the end of the input ~10% of the time. A binary classifier physically cannot.
- The "STRONG SIGNAL — turn provenance" stanza was added after observing the model declared topic shifts between EDUs that came from the same single assistant turn. The provenance signal is essentially free (already in the EDU's `source_turn_ids`) and dramatically improves precision.
- The window includes lookahead/lookback context EDUs marked "context" so the model can disambiguate without those EDUs being eligible decisions themselves — same CORE/CONTEXT pattern as stage 1.
- Uses Haiku rather than Sonnet because the task is bounded enough that the cheaper/faster model performs equivalently. See `BOUNDARY_DEFAULT_MODEL` in extractor.py.

### 1.4 — Trajectory labeling (`LABEL_SYSTEM_PROMPT`)

Stage 3. Once boundaries have grouped EDUs into trajectories, this prompt produces the human-readable summary + canonical keywords that show up in the per-project memory index.

`extractor.py: LABEL_SYSTEM_PROMPT` (line 501)

```
You are a trajectory labeler. Given a group of EDUs (Elementary Discourse Units) that ALL discuss a single topic, produce a concise summary and canonical keywords for that topic.

CRITICAL: Use ONLY the EDUs provided below. Do not invent content, do not extract from CLAUDE.md or other context you can see.

Requirements:
- `summary`: a single sentence describing what the trajectory was ABOUT — the subject matter discussed. Phrasings like "Discussed X", "Explored Y", "Debugged Z", "Decided W" are all fine. Past tense, action-oriented. This is the "chapter title" that will appear in a recent-activity index.
- `keywords`: 2-5 short, lowercase, canonical topic labels. PREFER reusing keywords from the provided project cloud; introduce a new keyword ONLY when no existing one fits. Keywords must be normalized — prefer common short forms (e.g. "pipewire", not "PipeWire" or "pipewire-1.0"). No spaces; use hyphens (e.g. "build-from-source"). Singular over plural.

Output: {"summary": "...", "keywords": ["...", "..."]}
```

**Why this shape:**
- The "CRITICAL" stanza is the same anti-CLAUDE.md guardrail as stage 1 — labeling a trajectory while seeing CLAUDE.md tempts the model to drift toward filename-style summaries.
- Past-tense, action-oriented summaries make the resulting index read like a changelog ("Debugged xrun on HDMI sink", "Decided to drop EasyEffects") rather than a topic taxonomy. Concrete verbs make the index scannable.
- The "PREFER reusing keywords from the provided project cloud" rule keeps the keyword vocabulary tight. Without it, the same topic shows up under three near-synonyms ("pipewire", "audio-server", "pipewire-1.0") and keyword search degrades. The cloud is computed from past trajectories of the same project.

#### One-shot example

`extractor.py: LABEL_ONE_SHOT_INPUT / LABEL_ONE_SHOT_OUTPUT` (line 513). Same wrap pattern as stage 1.

```
Project: home
Existing project keyword cloud: audio, claude-memory, niri, pipewire, wayland

EDUs in this trajectory:
1. [gotcha] PipeWire's LV2 filter-chain module (libpipewire-module-filter-chain-lv2.so) was built from source because the Ubuntu PPA build did not include it
2. [config] PipeWire was built from source at ~/builds/pipewire/ using tag 1.0.5, and the LV2 module was installed to /usr/lib/x86_64-linux-gnu/pipewire-0.3/
3. [gotcha] If PipeWire is updated via apt, the custom-built LV2 module may need to be rebuilt from ~/builds/pipewire/

→ {"summary": "Built PipeWire's LV2 filter-chain module from source because the Ubuntu PPA build omitted it", "keywords": ["pipewire", "lv2", "build-from-source"]}
```

---

## Part 2 — Research-paper prompts (designed but not yet implemented)

These appear in the literature and the original design notes. They're documented here because the system is structured to add them later — but at the moment retrieval bypasses an LLM entirely (pure dense + recency-decay scoring in `retrieval.py`). If/when these get wired in, they should be moved up to Part 1.

### 2.1 — EDU relevance filter (from EMem)

A second pass that prunes the dense-retrieval candidate set to query-relevant ones. EMem reports this filter step measurably improves answer accuracy by removing tangentially-similar EDUs that crowd out the true matches.

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

### 2.2 — Session summary (coarse-grained browsing)

Produces a 1-2 sentence per-session summary as an alternative entry point to per-trajectory labeling. The current system uses trajectory-level labels instead, but a session-level rollup could surface "what was this whole session about" at a glance.

```
Given the following extracted facts (EDUs) from a single conversation session, write a 1-2 sentence summary of what was discussed. Focus on the topics and outcomes, not the process.

EDUs:
${edu_list}

Summary:
```

### 2.3 — Query expansion (for ambiguous queries)

When the user query is too vague to match anything cleanly, expand it into multiple targeted sub-queries and merge the result sets. Standard IR technique; useful when the search index is small enough that recall trumps precision.

```
The user is searching their conversation history with this query: "${query}"

Generate 2-3 specific sub-queries that would help find relevant information. Each sub-query should target a different angle or aspect of the original query.

Output as JSON: {"queries": ["...", "...", "..."]}
```

---

## Part 3 — Design principles drawn from the literature

The prompts above embody decisions that came from these papers. Notes here for reviewers (and future-me) to audit whether the implementation is still tracking the source.

### From EMem ([arXiv:2511.17208](https://arxiv.org/abs/2511.17208))
- One-shot prompting with a worked example outperforms zero-shot for EDU extraction → implemented in `call_claude`'s example wrap.
- EDUs should be "minimal yet complete in meaning" — atomic but self-contained → enforced by requirement 1 of `SYSTEM_PROMPT`.
- "Consistently use the most informative name for each entity in all EDUs" → requirement 2 of `SYSTEM_PROMPT`.
- Source turn attribution enables traceability back to original conversation → enforced by `source_turn_ids` field in the JSON schema; further enforced by the "must reference turn numbers that actually appear in the input" rule.

### From Memoria ([arXiv:2512.12686](https://arxiv.org/abs/2512.12686))
- Extract from user messages primarily — they contain intent and preferences. *Partial implementation:* current system extracts from both, with EDUs tagged by speaker via `speakers: list[str]`.
- Recency weighting via exponential decay → implemented in `retrieval.py` as `score = similarity * exp(-0.007 * days_ago)`.
- Normalize weights to prevent total suppression of old memories → handled implicitly by the decay constant being small enough (`0.007` ≈ half-life of ~99 days).

### From A-Mem ([arXiv:2502.12110](https://arxiv.org/abs/2502.12110))
- Extract keywords and categorical tags alongside the fact itself → tags via the closed `EDUTag` set in stage 1; keywords via `LABEL_SYSTEM_PROMPT` in stage 3.
- Link new memories to existing ones by semantic similarity → *not yet implemented*; would require an evolution-action pass after stage 3.
- Evolution actions: strengthen (reinforce existing), update_neighbor (propagate changes) → *not yet implemented*.

### From the survey papers
- "Visible ≠ usable" — models can't use information buried mid-context (Lost in the Middle) → motivates the per-project SessionStart index (one-line summaries up front) instead of relying on the user to find old conversations.
- Structure beats embedding quality — organized retrieval >> better vectors → the trajectory grouping in stage 2/3 is itself a structural retrieval signal: queries that match a trajectory's keywords return its EDUs even when the EDU text doesn't match the query.
- Hybrid retrieval (dense + metadata + temporal) consistently outperforms single-signal → implemented as similarity * recency-decay in `retrieval.py`; tag-filtered retrieval is available but not yet exposed as a query-time knob.
