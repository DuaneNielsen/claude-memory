# Benchmarks

Without measurement we are doing art, not engineering. This document defines
how we evaluate claude-memory design choices.

---

## Why Benchmark

Every design decision (index structure, ranking formula, retrieval method,
injection strategy) creates a measurable effect. Without benchmarks we are
choosing architectures based on paper claims and intuition — both are
unreliable in this space:

- **Paper claims are often unreplicable**. Memoria, EMem, A-Mem all cite
  different eval setups with non-comparable numbers.
- **Vendor leaderboards are gamed**. MemPalace patched failing questions then
  retested (100% "hybrid" score). Mem0 jumped from 49% → 93.4% in one year
  without independent replication.
- **Benchmarks disagree**. Mem0 scores 93% on LongMemEval but **5%** on
  MemoryAgentBench BANKING-77. Systems that win one often lose the other.

A local benchmark suite lets us compare our own design choices against each
other on the same data — which is the only comparison that matters for
shipping decisions.

---

## Benchmark Stack

Three tiers. Run Tier 1 for rapid iteration, Tier 2 for honest evaluation,
Tier 3 for the final ship/no-ship decision.

### Tier 1 — LongMemEval-S (primary, already scaffolded)

- **Status**: `benchmarks/longmemeval_lite.py` implements a 30-question
  subset runner. Data at `benchmarks/data/longmemeval_s_cleaned.json`
  (265 MB, full LongMemEval-S).
- **What it tests**: 7 question types × chat-memory setting. 500 questions,
  ~115K tokens of conversation history per test.
- **Metrics**: Recall@K, LLM-judge accuracy (GPT-4o judge has >97% human
  agreement).
- **Why**: fastest iteration, directly comparable to published Mem0 / Zep /
  Supermemory numbers.
- **Known weakness**: chat-assistant flavor, not agentic code work. Tier 3
  covers the gap.

### Tier 2 — MemoryAgentBench (secondary)

- **Dataset**: `huggingface.co/datasets/ai-hyz/MemoryAgentBench`
- **Code**: `github.com/HUST-AI-HYZ/MemoryAgentBench`
- **What it tests**: 4 competencies — accurate retrieval, test-time learning,
  long-range understanding, selective forgetting.
- **Why**: harder to game than LongMemEval. Uses incremental turn-by-turn
  feeding (no oracle retrieval). Adds selective-forgetting dimension that
  LongMemEval lacks.
- **Reality check**: every memory system currently fails multi-hop conflict
  resolution. Paper's explicit conclusion: "current methods fall short of
  mastering all four."

### Tier 3 — Local "Claude Code Recall" (definitive)

Public benchmarks test open-domain chat. Our use case is multi-day software
engineering with tool use. A local benchmark built from real session archives
is the only honest test.

**Raw material**: ~122 sessions at `~/.claude/projects/-home-duane/`.

**Construction methodology**:

1. **Mine questions**: for each session jsonl, LLM-generate 2-3 factoid
   questions whose answer requires that session's content. Bias toward:
   - Decisions made
   - Config values chosen
   - Rejected approaches and why
   - Tool commands that worked
2. **Categorize** along LongMemEval's 7 types + 2 code-specific additions:
   - `config-value-lookup` — what port/path/flag did we settle on?
   - `decision-reversal` — what approach did we abandon and why?
3. **Ground truth**: answer + source session UUID + turn range.
4. **Question timestamp**: fix to the session immediately *after* the source.
   Tests recency-weighted retrieval (EDU should be available but not dominant).
5. **Haystack sizes**: build 5 / 20 / 50 / 100-session test variants. Measures
   scaling behavior.
6. **Metrics**: Recall@10 on session UUIDs + LLM-judge on final answer.
   Reuse `longmemeval_lite.py` evaluator plumbing.
7. **Size target**: 100-150 questions across 9 categories. LongMemEval uses
   500; 100 is the practical floor for discriminating design choices.

---

## Baselines

Every experiment must compare against these four baselines. Without them,
scores are meaningless.

### 1. Raw full-context

Dump raw session transcripts into the model's context. No preprocessing,
no retrieval.

- **LongMemEval-S, GPT-4o**: 60.2% — this is the *floor*.
- Useful only where the corpus fits in the model's window.

### 2. EDU-dump (stable order)

Dump ALL extracted EDUs into the context in a stable order. No retrieval,
no ranking.

- **Prompt-cache perfect**: same prefix every turn → near-zero marginal cost
  after turn 1.
- Tests whether retrieval/ranking adds value *given* ingestion.
- At 500 EDUs × ~20 tokens = 10K tokens. Fits anywhere.
- **If our retrieval system doesn't beat EDU-dump at current corpus size,
  ship EDU-dump.** It's simpler, cache-friendly, no MCP orchestration.

### 3. Oracle retrieval

Retrieve exactly the correct EDUs (using ground-truth session labels).
This is the *ceiling* — no retrieval system can do better.

- Tells us the gap between ranking quality and what's achievable.
- If oracle - our system is small, ranking is solved. Focus elsewhere.

### 4. BM25 retrieval

Lexical baseline. Fast, deterministic, no embeddings.

- **MemoryAgentBench**: BM25 scores 45.3 / 74.6 vs NV-Embed-v2 at 55.0 /
  72.8. Often competitive with dense retrieval.
- If BM25 matches our dense pipeline, the embeddings aren't earning their
  cost.

---

## What Design Choices Each Benchmark Measures

| Design choice | Benchmark |
|---|---|
| Index structure (flat vs grouped) | Tier 1 + Tier 3: A/B on Recall@10 + accuracy |
| Ranking formula (recency-only vs hybrid) | All tiers: parameter sweep on α, halflife |
| Section sizes (200 / 500 / 800 tok) | Tier 1 + Tier 3: find the knee |
| Always-on injection vs tool-only | All tiers: compare both modes |
| Project scoping vs global | Tier 3 only (Tiers 1-2 are single-user) |
| EDU extraction quality | EDU-dump vs raw-full-context on Tier 3 |
| Retrieval quality given ingestion | Our system vs EDU-dump on Tier 3 |
| Dense vs lexical retrieval | All tiers: ours vs BM25 baseline |

---

## SOTA Snapshot (April 2026)

Numbers to orient against. Treat >85% on LongMemEval as marketing until
independently replicated.

### LongMemEval-S

| System | Score | Trust |
|---|---|---|
| MemPalace | 96.6% | gamed — patched failing Qs, retested |
| OMEGA | 95.4% | vendor self-report |
| Mem0 (2026 algorithm) | 93.4% | vendor blog, credible |
| Supermemory (Gemini-3-Pro) | 85.2% | credible |
| TiMem (GPT-4o-mini) | 76.88% | peer-reviewed |
| Zep / Graphiti | 71.2% | third-party verified |
| **Full-context baseline** | **60.2%** | **the bar to beat** |
| Mem0 (original 2024) | 49.0% | — |

**Credible target for claude-memory: 75-85%.** Higher needs replication to
believe.

### MemoryAgentBench (from paper)

Accurate Retrieval (LongMemEval-S / EventQA):
- GPT-4.1-mini long-context: **55.7 / 82.6**
- HippoRAG-v2: 50.7 / 67.6
- Mem0: 36.0 / 37.5

Test-Time Learning (BANKING-77 / TREC-Fine):
- Claude-3.7-Sonnet: **97 / 79**
- Mem0: **5 / 1** — broken
- Cognee: 34 / 18

Long-Range Understanding (∞Bench):
- Claude-3.7-Sonnet: **52.5**
- HippoRAG-v2: 14.6
- All memory systems: <21

Conflict Resolution (single-hop / multi-hop):
- GPT-4o: 60 / 5
- **Every system fails multi-hop.**

**Key finding**: long-context models often beat memory systems. Memory wins
on *specific factoid recall*, loses on *broad synthesis*.

---

## Techniques That Win

From published results and ablations:

### LongMemEval
- User-fact extraction as retrieval keys (+5pp)
- Temporal indexing (+7-11pp on temporal reasoning category)
- Treating assistant-generated facts as first-class (Mem0's +53pp jump on
  single-session-assistant)
- Chain-of-note reading at query time (+10pp)
- LLM relevance filter at query time — EMem ablation: single biggest win

### MemoryAgentBench
- Dense embeddings (NV-Embed-v2) for accurate retrieval
- HippoRAG-v2 graph retrieval competitive but not dominant
- For long-range / test-time-learning: no memory system wins. Use long
  context directly.

---

## Metrics

Standard families. Report multiple, not just one.

| Metric | Use for | Notes |
|---|---|---|
| **Recall@K** | Retrieval-only eval | K=10 standard. Sensitive to ranking. |
| **NDCG@K** | Ranking quality | Rewards putting the right answer near top. |
| **LLM-judge accuracy** | End-to-end answer correctness | Headline number. GPT-4o judge has >97% human agreement. |
| **Token efficiency** | Cost | Avg context size passed to answer model. |
| **Latency** | UX | p50 / p95 per query, including ingestion if applicable. |

---

## Gaming Protection

Our benchmarks must be resistant to overfitting. Rules:

1. **Never tune on Tier 3 test set**. Hold it out. Use Tier 1/2 for
   iteration.
2. **Report all baselines every time**. A system that beats full-context by
   3pp on LongMemEval is not a win if EDU-dump is +5pp at 1/10 the
   complexity.
3. **Report standard deviation** across seeds (LongMemEval-S subsets, Tier 3
   question samples). Single-point scores are unreliable.
4. **Version lock**: every benchmark run records claude-memory git sha,
   model version (Claude / GPT-4o judge), ChromaDB schema version. Results
   without a lock are noise.

---

## Key Files

- `benchmarks/longmemeval_lite.py` — 30-question LongMemEval-S runner,
  resumable, ChromaDB + Claude CLI judge.
- `benchmarks/data/longmemeval_s_cleaned.json` — full LongMemEval-S dataset.
- `benchmarks/data/longmemeval_lite.json` — 15 MB sampled subset.
- `~/.claude/projects/-home-duane/` — session archive, source for Tier 3.

---

## Sources

- [LongMemEval paper](https://arxiv.org/abs/2410.10813) · [LongMemEval GitHub](https://github.com/xiaowu0162/LongMemEval) · [HF dataset](https://huggingface.co/datasets/xiaowu0162/longmemeval)
- [MemoryAgentBench paper](https://arxiv.org/abs/2507.05257) · [MemoryAgentBench GitHub](https://github.com/HUST-AI-HYZ/MemoryAgentBench) · [HF dataset](https://huggingface.co/datasets/ai-hyz/MemoryAgentBench)
- [Mem0 2026 algorithm blog](https://mem0.ai/blog/mem0-the-token-efficient-memory-algorithm)
- [Supermemory research](https://supermemory.ai/research/)
- [Lost in the Middle](https://arxiv.org/abs/2307.03172) — context position effects, methodology reusable as ablation
- [Context Length Hurts Despite Retrieval](https://arxiv.org/abs/2510.05381) — 13-85% degradation evidence
- [LoCoMo](https://arxiv.org/abs/2402.17753) — EMem's primary benchmark, persona chat. Skip for our use case.
