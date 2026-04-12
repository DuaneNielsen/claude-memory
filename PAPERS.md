# Key Implementation Details from Papers

Extracted findings that directly inform implementation decisions.

---

## EMem — "A Simple Yet Strong Baseline" (arXiv:2511.17208)

### Architecture
- **Input**: Multi-session conversation logs
- **Processing**: LLM extracts Elementary Discourse Units (EDUs) per session using one-shot prompting
- **Storage**: Heterogeneous graph with 3 node types: sessions → EDUs → arguments (entities/values extracted from EDUs)
- **Retrieval**: Dense similarity → LLM filter → optional PageRank propagation

### Critical Numbers
- **738 tokens** average context passed to QA model (vs 23,653 for full-context) — 97% reduction
- **0.780 overall LLM score** on LoCoMo benchmark (SOTA at time of publication)
- EMem (no graph) nearly matches EMem-G (with graph): the graph adds marginal value
- **Ablation results**: the LLM relevance filter provides the single largest performance gain, especially for multi-hop reasoning

### What To Steal
- The EDU extraction prompt structure (system prompt + one-shot example + structured JSON output)
- Pydantic models for type-safe LLM output parsing
- The "maximally inclusive" filtering philosophy — recall matters more than precision at retrieval time
- Synonym edges between arguments based on embedding similarity (handles entity deduplication cheaply)

### What To Skip
- The graph construction + PageRank — adds complexity, marginal gains. Only worth it if retrieval quality proves insufficient
- DSPy prompt optimization — over-engineered for this use case
- The argument extraction second pass — EDUs alone are sufficient for keyword/semantic search

### Gotcha
- EDU extraction quality degrades on very long sessions. They cap at reasonable session lengths. For Claude Code sessions that run 500+ turns, consider chunking into logical segments first.

---

## Memoria — "Scalable Agentic Memory" (arXiv:2512.12686)

### Architecture
- **Four modules**: conversation logging (SQL), dynamic user KG (triplets), session summaries, recency-weighted retrieval
- **Storage**: SQLite for raw logs, ChromaDB for vector embeddings
- **KG extraction**: Triplets extracted from user messages ONLY (not assistant responses) — captures user intent accurately

### Critical Numbers
- **87.1% accuracy** on single-session-user tasks (vs 85.7% full context, 78.5-84.2% A-Mem)
- **80.8% accuracy** on knowledge-update tasks (vs 78.2% full context)
- **~400 tokens** average context (vs 115K full context) — 99.6% reduction
- **38.7% latency reduction** vs full-context approach
- **Decay rate α = 0.02** (per minute) for exponential weighting
- **Top-K = 20** for retrieval

### What To Steal
- **Exponential decay weighting**: `w = e^(-α * minutes_since_creation)`, then min-max normalize. Simple, effective, handles contradictions by favoring recent facts
- **Extract from user messages primarily** — they contain the ground truth about intent and preferences. Assistant messages contain decisions but also a lot of noise (tool outputs, explanations)
- **ChromaDB + SQLite** stack — proven at this scale, zero external dependencies
- Session summary as a coarse-grained index — enables "what did we talk about last week?" queries without hitting individual EDUs

### What To Skip
- Their KG triplet format is less expressive than full EDUs. Use EDUs instead — they contain more context and are more searchable
- OpenAI-specific embedding (text-embedding-ada-002) — use a local model instead

### Key Insight
> "Memoria's recency-aware weighting resolving contradictions and emphasizing updated information, whereas A-Mem's unweighted retrieval proves less precise."

When a user changes a config value or reverses a decision, the most recent fact should win. Exponential decay achieves this naturally without explicit conflict resolution logic.

---

## "From Human Memory to AI Memory" Survey (arXiv:2504.15965)

### Taxonomy (3D-8Q)
Three axes: Object (personal/system), Form (parametric/non-parametric), Time (short/long-term) → 8 quadrants.

**Our system lives in Quadrant II**: Personal, Non-parametric, Long-term memory.

### Implementation-Relevant Findings
- **Memory processing stages map to our pipeline**: Encoding (EDU extraction) → Storage (ChromaDB) → Retrieval (query pipeline) → Consolidation (incremental updates)
- **Four retrieval approaches for non-parametric memory**:
  1. SQL queries for structured data (metadata filters)
  2. Fuzzy search for triplets (embedding similarity)
  3. Knowledge graphs for comprehensive recall (we're skipping this)
  4. Vector formats for semantic meaning (our primary approach)
- **Forgetting curves are real**: MemoryBank implements Ebbinghaus forgetting curves. For our use case, exponential decay on retrieval scores achieves similar effect without deleting data

### Six Research Transitions (relevance to us)
1. **Unimodal → Multimodal** — future: could index screenshots, diagrams from sessions
2. **Static → Stream** — relevant: incremental ingestion of new sessions
3. **Specific → Comprehensive** — relevant: cross-project memory linking
4. **Rule-based → Automated Evolution** — future: self-curating memory that prunes stale facts

---

## "Memory in LLMs: Mechanisms, Evaluation" Survey (arXiv:2509.18868)

### Evaluation Framework We Should Adopt

**Metric families relevant to our system:**
1. **Accuracy**: Does the retrieved context help answer the question correctly?
2. **Recall@K**: Are the relevant facts in the top-K results?
3. **Groundedness**: Are returned facts actually from conversations (not hallucinated)?
4. **Efficiency**: Latency per query, tokens per query

**Testing approach:**
- Create a test set of 50-100 questions about past conversations with known answers
- Measure Recall@10 (are the relevant EDUs in top 10?)
- Measure end-to-end accuracy (does Claude Code answer correctly with the retrieved context?)

### Key Finding: "Visible ≠ Usable"
> Models systematically lose information positioned mid-sequence. Performance degrades for evidence buried in the middle of context, regardless of context window size.

**Implication**: Our approach of returning <1K tokens of targeted EDUs should outperform dumping large chunks of conversation history, even if the model could fit them in context. This validates the entire architecture.

### Write-Read-Inhibit Cycle
- **Write** = ingestion (EDU extraction + embedding)
- **Read** = retrieval (query pipeline)
- **Inhibit** = not deleting, but deprioritizing via recency decay

---

## "Memory in the Age of AI Agents" Survey (arXiv:2512.13564)

### Taxonomy Mapping
Our system in their framework:
- **Form**: Token-level (EDUs are explicit text, stored discretely)
- **Function**: Factual memory (declarative knowledge from past conversations)
- **Dynamics**: Formation via extraction, retrieval via embedding search, evolution via incremental ingestion

### Gap We're NOT Filling
The survey identifies **parametric experiential memory** (agents updating weights from experience) as severely under-researched. We're not doing this — we're firmly in the non-parametric, retrieval-augmented camp. And that's fine for this use case.

### Relevant Architecture Patterns from Cataloged Papers
- **Generative Agents** (Park et al.): Reflection + importance scoring. Memories get an importance score (1-10) at creation time. Could add this — "how likely is this fact to be useful in future conversations?"
- **HippoRAG**: Schemaless KG + PageRank. Over-engineered for our needs but the "synonym edges" idea is cheap and useful
- **MemGPT/Letta**: LLM manages its own memory (decides what to save/search). Interesting but burns tokens. Our batch extraction approach is more efficient

---

## A-Mem — "Agentic Memory for LLM Agents" (arXiv:2502.12110)

### What's Useful
- **Note construction prompt**: Extract keywords + context (one-sentence summary) + categorical tags per memory. We could add tags to EDUs cheaply:
  ```json
  {"text": "...", "keywords": ["pipewire", "audio", "filter-chain"], "tags": ["configuration", "audio"]}
  ```
- **Link generation**: Compare new memories against nearest neighbors to find connections. Could use this during incremental ingestion — when a new EDU is similar to an existing one, flag potential updates/contradictions
- **Evolution actions**: `strengthen` (reinforce a fact that keeps coming up) and `update_neighbor` (propagate changes to related facts). Nice-to-have, not essential for v1

### What's Not Useful
- Their agentic approach (LLM decides in real-time what to memorize) doesn't fit our batch ingestion model
- The memory evolution loop is complex and our recency weighting achieves similar effect more simply

---

## Synthesis: The Implementation Recipe

Based on convergent findings across all papers:

1. **EDU extraction is the highest-leverage component** — spend time on prompt quality here
2. **ChromaDB + local embeddings** is the proven stack for this scale (< 100K documents)
3. **Exponential decay weighting** (α ≈ 0.005-0.02) handles contradiction resolution and recency naturally
4. **LLM relevance filter at query time** provides the single biggest retrieval quality boost (EMem ablation)
5. **< 1K tokens of targeted context** outperforms 20-100K tokens of raw history (Lost in the Middle effect)
6. **Skip the knowledge graph for v1** — marginal gains, significant complexity. Add if retrieval quality plateaus
7. **Extract from user messages primarily** — they contain intent, preferences, ground truth. Include assistant messages for decisions and technical details, but filter out tool outputs
8. **Test with known-answer questions** about past conversations to measure quality
