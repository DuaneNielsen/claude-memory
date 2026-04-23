"""LongMemEvalLite — benchmark claude-memory against a 30-question LongMemEval subset.

Usage:
    python benchmarks/longmemeval_lite.py sample
    python benchmarks/longmemeval_lite.py extract --model sonnet
    python benchmarks/longmemeval_lite.py evaluate --run run_001 --extraction sonnet_42
    python benchmarks/longmemeval_lite.py report --run run_001
    python benchmarks/longmemeval_lite.py all --run run_001 --model sonnet
"""

import argparse
import asyncio
import hashlib
import json
import logging
import math
import re
import sys
import uuid
import yaml
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add project root to path so we can import claude_memory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from claude_memory.extractor import EDU, EDUTag, extract_edus_from_session, call_claude
from claude_memory.parser import Turn, Session
from claude_memory.config import (
    DEFAULT_MODEL,
    EMBEDDING_MODEL,
    RETRIEVAL_CANDIDATES,
    RECENCY_DECAY_ALPHA,
    DEFAULT_MAX_RESULTS,
)

log = logging.getLogger(__name__)

BENCHMARKS_DIR = Path(__file__).resolve().parent
DATA_DIR = BENCHMARKS_DIR / "data"
EXTRACTIONS_DIR = BENCHMARKS_DIR / "extractions"
RUNS_DIR = BENCHMARKS_DIR / "runs"

FULL_DATASET = DATA_DIR / "longmemeval_s_cleaned.json"
LITE_DATASET = DATA_DIR / "longmemeval_lite.json"

QUESTIONS_PER_TYPE = 5
SAMPLE_SEED = 42

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def content_hash(session_messages: list[dict]) -> str:
    """Stable hash of a session's message content for deduplication."""
    raw = json.dumps(session_messages, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def parse_longmemeval_date(date_str: str) -> datetime:
    """Parse LongMemEval date like '2023/05/20 (Sat) 02:21' -> datetime (UTC)."""
    m = re.match(r"(\d{4}/\d{2}/\d{2})\s+\(\w+\)\s+(\d{2}:\d{2})", date_str)
    if not m:
        raise ValueError(f"Cannot parse date: {date_str!r}")
    return datetime.strptime(f"{m.group(1)} {m.group(2)}", "%Y/%m/%d %H:%M").replace(
        tzinfo=timezone.utc
    )


def longmemeval_to_session(
    session_id: str,
    messages: list[dict],
    session_date: datetime,
) -> Session:
    """Convert a LongMemEval session to a claude-memory Session object."""
    turns = []
    for i, msg in enumerate(messages):
        turns.append(
            Turn(
                turn_id=i,
                session_id=session_id,
                project="longmemeval",
                timestamp=session_date + timedelta(minutes=i),
                speaker=msg["role"],
                text=msg["content"],
                git_branch=None,
            )
        )
    return Session(
        session_id=session_id,
        project="longmemeval",
        file_path=Path("synthetic"),
        turns=turns,
    )


def serialize_edus(edus: list[EDU]) -> list[dict]:
    """Serialize EDU objects to JSON-safe dicts."""
    out = []
    for e in edus:
        out.append(
            {
                "edu_id": e.edu_id,
                "text": e.text,
                "source_turn_ids": e.source_turn_ids,
                "timestamp": e.timestamp.isoformat(),
                "speakers": e.speakers,
                "tag": e.tag.value,
            }
        )
    return out


def deserialize_edus(raw: list[dict], session_id: str, project: str = "longmemeval") -> list[EDU]:
    """Reconstruct EDU objects from cached JSON."""
    edus = []
    for d in raw:
        edus.append(
            EDU(
                edu_id=d["edu_id"],
                text=d["text"],
                source_turn_ids=d["source_turn_ids"],
                session_id=session_id,
                project=project,
                timestamp=datetime.fromisoformat(d["timestamp"]),
                speakers=d.get("speakers", []),
                tag=EDUTag.coerce(d.get("tag")),
            )
        )
    return edus


# ---------------------------------------------------------------------------
# search_at_date — local copy of query.search() with reference_date param
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    text: str
    score: float
    session_id: str
    project: str
    timestamp: datetime
    similarity: float
    recency_weight: float


def search_at_date(
    collection,
    embedding_fn,
    query: str,
    reference_date: datetime,
    max_results: int = DEFAULT_MAX_RESULTS,
    candidates: int = RETRIEVAL_CANDIDATES,
    decay_alpha: float = RECENCY_DECAY_ALPHA,
) -> list[SearchResult]:
    """Search with a fixed reference date instead of now()."""
    query_embedding = embedding_fn([query])[0]
    raw = collection.query(
        query_embeddings=[query_embedding],
        n_results=candidates,
        include=["documents", "metadatas", "distances"],
    )

    if not raw["ids"] or not raw["ids"][0]:
        return []

    results = []
    for i, doc_id in enumerate(raw["ids"][0]):
        doc = raw["documents"][0][i]
        meta = raw["metadatas"][0][i]
        distance = raw["distances"][0][i]

        similarity = 1.0 - (distance / 2.0)
        ts = datetime.fromisoformat(meta["timestamp"])
        days_ago = max(0, (reference_date - ts).total_seconds() / 86400.0)
        recency_weight = math.exp(-decay_alpha * days_ago)
        score = similarity * recency_weight

        results.append(
            SearchResult(
                text=doc,
                score=score,
                session_id=meta["session_id"],
                project=meta["project"],
                timestamp=ts,
                similarity=similarity,
                recency_weight=recency_weight,
            )
        )

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:max_results]


# ---------------------------------------------------------------------------
# Phase 1: sample
# ---------------------------------------------------------------------------


def run_sample():
    """Create the 30-question sample from the full dataset."""
    import random

    if not FULL_DATASET.exists():
        print(f"ERROR: Full dataset not found at {FULL_DATASET}")
        print("Download it first (see README).")
        sys.exit(1)

    print(f"Loading {FULL_DATASET.name}...")
    with open(FULL_DATASET) as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} questions")

    # Group by type
    by_type: dict[str, list] = {}
    for q in questions:
        by_type.setdefault(q["question_type"], []).append(q)

    # Sample
    random.seed(SAMPLE_SEED)
    sample = []
    for qtype, qs in sorted(by_type.items()):
        picked = random.sample(qs, min(QUESTIONS_PER_TYPE, len(qs)))
        sample.extend(picked)
        print(f"  {qtype}: {len(picked)} questions")

    # Save
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(LITE_DATASET, "w") as f:
        json.dump(sample, f, indent=2)

    print(f"\nSaved {len(sample)} questions to {LITE_DATASET.name}")

    # Stats
    all_hashes = set()
    total_sessions = 0
    total_turns = 0
    for q in sample:
        for sess in q["haystack_sessions"]:
            h = content_hash(sess)
            if h not in all_hashes:
                all_hashes.add(h)
                total_turns += len(sess)
            total_sessions += 1

    print(f"Total session refs: {total_sessions}")
    print(f"Unique sessions: {len(all_hashes)}")
    print(f"Total turns (unique): {total_turns}")


# ---------------------------------------------------------------------------
# Phase 2: extract
# ---------------------------------------------------------------------------


async def run_extract(args):
    """Extract EDUs from all unique sessions in the sample."""
    from tqdm import tqdm

    if not LITE_DATASET.exists():
        print("ERROR: Run 'sample' first.")
        sys.exit(1)

    model = args.model or DEFAULT_MODEL
    extraction_name = f"{model}_{SAMPLE_SEED}"
    extraction_dir = EXTRACTIONS_DIR / extraction_name
    cache_dir = extraction_dir / "edu_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Write extraction config
    config = {
        "created": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "sample_seed": SAMPLE_SEED,
        "questions_per_type": QUESTIONS_PER_TYPE,
    }
    with open(extraction_dir / "extraction_config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Load sample
    with open(LITE_DATASET) as f:
        questions = json.load(f)

    # Collect unique sessions: hash -> (messages, date_str)
    unique_sessions: dict[str, tuple[list[dict], str]] = {}
    for q in questions:
        dates = q.get("haystack_dates", [])
        for i, sess in enumerate(q["haystack_sessions"]):
            h = content_hash(sess)
            if h not in unique_sessions:
                date_str = dates[i] if i < len(dates) else "2023/01/01 (Sun) 00:00"
                unique_sessions[h] = (sess, date_str)

    # Filter to uncached
    to_extract = {
        h: v for h, v in unique_sessions.items() if not (cache_dir / f"{h}.json").exists()
    }

    print(f"Extraction: {extraction_name}")
    print(f"Unique sessions: {len(unique_sessions)}")
    print(f"Already cached: {len(unique_sessions) - len(to_extract)}")
    print(f"To extract: {len(to_extract)}")

    if not to_extract:
        print("Nothing to extract.")
        return

    sem = asyncio.Semaphore(args.concurrency)
    failed = 0

    async def extract_one(h: str, messages: list[dict], date_str: str):
        nonlocal failed
        async with sem:
            try:
                session_date = parse_longmemeval_date(date_str)
                session = longmemeval_to_session(h, messages, session_date)
                edus = await extract_edus_from_session(session, model=model)
                cache_file = cache_dir / f"{h}.json"
                with open(cache_file, "w") as f:
                    json.dump(
                        {"session_hash": h, "edus": serialize_edus(edus)},
                        f,
                        indent=2,
                    )
            except Exception as e:
                log.error(f"Failed to extract {h}: {e}")
                failed += 1
            finally:
                pbar.update(1)

    pbar = tqdm(total=len(to_extract), desc="Extracting EDUs")
    tasks = [
        extract_one(h, msgs, date_str) for h, (msgs, date_str) in to_extract.items()
    ]
    await asyncio.gather(*tasks)
    pbar.close()

    cached_count = sum(1 for f in cache_dir.iterdir() if f.suffix == ".json")
    print(f"\nDone. Cached: {cached_count}/{len(unique_sessions)} sessions, Failed: {failed}")


# ---------------------------------------------------------------------------
# Phase 3: evaluate
# ---------------------------------------------------------------------------

ANSWER_SYSTEM_PROMPT = """\
You are answering a question based on your memory of past conversations with a user. \
Use the provided facts to answer. If the facts don't contain enough information, say "I don't know." \
Be concise and direct."""

JUDGE_SYSTEM_PROMPT = """\
You are evaluating whether a generated answer correctly responds to a question. \
The generated answer doesn't need to match the reference exactly — it should contain the key information. \
Partial correctness counts as incorrect. Semantic equivalence (different wording, same meaning) is correct. \
Extra correct details beyond the reference don't make it wrong."""

ANSWER_JSON_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
)

JUDGE_JSON_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {
            "correct": {"type": "boolean"},
            "reasoning": {"type": "string"},
        },
        "required": ["correct", "reasoning"],
    }
)


async def call_claude_raw(prompt: str, system_prompt: str, model: str, json_schema: str) -> dict:
    """Call claude CLI with a custom prompt and schema (no one-shot wrapper)."""
    proc = await asyncio.create_subprocess_exec(
        "claude",
        "-p",
        "--model", model,
        "--tools", "",
        "--system-prompt", system_prompt,
        "--json-schema", json_schema,
        "--output-format", "json",
        "--no-session-persistence",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate(prompt.encode())

    if proc.returncode != 0:
        raise RuntimeError(f"claude CLI failed (exit {proc.returncode}): {stderr.decode()}")

    content = stdout.decode().strip()
    if not content:
        raise ValueError("claude CLI returned empty output")

    events = json.loads(content)
    if isinstance(events, list):
        for event in reversed(events):
            if isinstance(event, dict) and "structured_output" in event:
                return event["structured_output"]

    raise ValueError("No structured_output found in claude CLI response")


async def generate_answer(results: list[SearchResult], question: str, model: str) -> str:
    """Generate an answer from retrieved EDUs + question."""
    if not results:
        return "I don't know."

    context_lines = []
    for i, r in enumerate(results, 1):
        date_str = r.timestamp.strftime("%Y-%m-%d")
        context_lines.append(f"{i}. {r.text} (from {date_str}, score={r.score:.3f})")

    prompt = f"""Here are relevant facts from past conversations:
{chr(10).join(context_lines)}

Question: {question}"""

    result = await call_claude_raw(prompt, ANSWER_SYSTEM_PROMPT, model, ANSWER_JSON_SCHEMA)
    return result.get("answer", "I don't know.")


async def judge_answer(question: str, reference: str, generated: str, model: str) -> dict:
    """Use LLM judge to evaluate correctness."""
    prompt = f"""Question: {question}
Reference answer: {reference}
Generated answer: {generated}

Is the generated answer correct?"""

    return await call_claude_raw(prompt, JUDGE_SYSTEM_PROMPT, model, JUDGE_JSON_SCHEMA)


async def run_evaluate(args):
    """Evaluate retrieval + answer quality for each question."""
    import chromadb
    from sentence_transformers import SentenceTransformer

    if not LITE_DATASET.exists():
        print("ERROR: Run 'sample' first.")
        sys.exit(1)

    extraction_name = args.extraction
    extraction_dir = EXTRACTIONS_DIR / extraction_name
    cache_dir = extraction_dir / "edu_cache"

    if not cache_dir.exists():
        print(f"ERROR: Extraction '{extraction_name}' not found at {extraction_dir}")
        sys.exit(1)

    run_name = args.run
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Params
    embedding_model_name = args.embedding_model or EMBEDDING_MODEL
    max_results = args.max_results
    retrieval_candidates = args.retrieval_candidates
    decay_alpha = args.decay_alpha
    judge_model = args.judge_model or args.model or DEFAULT_MODEL

    # Write run config
    config = {
        "created": datetime.now(timezone.utc).isoformat(),
        "extraction": extraction_name,
        "embedding_model": embedding_model_name,
        "max_results": max_results,
        "retrieval_candidates": retrieval_candidates,
        "recency_decay_alpha": decay_alpha,
        "judge_model": judge_model,
    }
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Load sample
    with open(LITE_DATASET) as f:
        questions = json.load(f)

    # Load existing results for resumability
    results_file = run_dir / "results.json"
    existing_results = []
    done_ids = set()
    if results_file.exists():
        with open(results_file) as f:
            existing_results = json.load(f)
        done_ids = {r["question_id"] for r in existing_results}
        print(f"Resuming: {len(done_ids)}/{len(questions)} already evaluated")

    remaining = [q for q in questions if q["question_id"] not in done_ids]
    if not remaining:
        print("All questions already evaluated.")
        return

    # Init embedding model (reused across questions)
    print(f"Loading embedding model: {embedding_model_name}")
    embedder = SentenceTransformer(embedding_model_name, device="cpu")

    def embed_fn(texts):
        return embedder.encode(texts, show_progress_bar=False).tolist()

    # Init ChromaDB
    chroma_dir = run_dir / "chromadb"
    chroma_dir.mkdir(exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))

    # Build hash -> session_id mapping for each question
    # (questions use index-based IDs, we use content hashes for cache)
    print(f"\nEvaluating {len(remaining)} questions...")
    results = list(existing_results)

    for qi, q in enumerate(remaining):
        qid = q["question_id"]
        qtype = q["question_type"]
        print(f"\n[{qi + 1 + len(done_ids)}/{len(questions)}] {qid} ({qtype})")

        # 1. Load EDUs for this question's sessions
        edus = []
        sessions_loaded = 0
        for sess in q["haystack_sessions"]:
            h = content_hash(sess)
            cache_file = cache_dir / f"{h}.json"
            if cache_file.exists():
                with open(cache_file) as f:
                    data = json.load(f)
                edus.extend(deserialize_edus(data["edus"], session_id=h))
                sessions_loaded += 1

        print(f"  Loaded {len(edus)} EDUs from {sessions_loaded} sessions")

        # 2. Create isolated collection
        coll_name = f"bench_{qid}".replace("/", "_").replace(" ", "_")[:63]
        try:
            client.delete_collection(coll_name)
        except Exception:
            pass
        collection = client.create_collection(coll_name, metadata={"hnsw:space": "cosine"})

        # 3. Store EDUs
        if edus:
            texts = [e.text for e in edus]
            embeddings = embed_fn(texts)
            collection.add(
                ids=[e.edu_id for e in edus],
                documents=texts,
                embeddings=embeddings,
                metadatas=[
                    {
                        "session_id": e.session_id,
                        "project": e.project,
                        "timestamp": e.timestamp.isoformat(),
                        "speakers": ",".join(e.speakers),
                        "source_turns": json.dumps(e.source_turn_ids),
                    }
                    for e in edus
                ],
            )

        # 4. Search
        ref_date = parse_longmemeval_date(q["question_date"])
        search_results = search_at_date(
            collection,
            embed_fn,
            q["question"],
            reference_date=ref_date,
            max_results=max_results,
            candidates=retrieval_candidates,
            decay_alpha=decay_alpha,
        )
        print(f"  Search returned {len(search_results)} results (top score: {search_results[0].score:.3f})" if search_results else "  Search returned 0 results")

        # 5. Generate answer
        try:
            answer = await generate_answer(search_results, q["question"], judge_model)
        except Exception as e:
            log.error(f"  Answer generation failed: {e}")
            answer = f"ERROR: {e}"

        # 6. Judge
        try:
            judgment = await judge_answer(q["question"], q["answer"], answer, judge_model)
            correct = judgment.get("correct", False)
            reasoning = judgment.get("reasoning", "")
        except Exception as e:
            log.error(f"  Judging failed: {e}")
            correct = False
            reasoning = f"ERROR: {e}"

        status = "CORRECT" if correct else "WRONG"
        print(f"  Answer: {answer[:80]}...")
        print(f"  Reference: {q['answer'][:80]}...")
        print(f"  Verdict: {status}")

        # 7. Save result
        result = {
            "question_id": qid,
            "question_type": qtype,
            "question": q["question"],
            "reference_answer": q["answer"],
            "generated_answer": answer,
            "correct": correct,
            "reasoning": reasoning,
            "num_edus_loaded": len(edus),
            "num_results": len(search_results),
        }
        results.append(result)

        # Write after each question for resumability
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        # 8. Cleanup collection
        try:
            client.delete_collection(coll_name)
        except Exception:
            pass

    print(f"\nDone. Results saved to {results_file}")


# ---------------------------------------------------------------------------
# Phase 4: report
# ---------------------------------------------------------------------------


def run_report(args):
    """Print results summary."""
    run_name = args.run
    run_dir = RUNS_DIR / run_name
    results_file = run_dir / "results.json"

    if not results_file.exists():
        print(f"ERROR: No results found at {results_file}")
        sys.exit(1)

    with open(results_file) as f:
        results = json.load(f)

    # Config
    config_file = run_dir / "config.yaml"
    if config_file.exists():
        with open(config_file) as f:
            config = yaml.safe_load(f)
        print(f"Run: {run_name}")
        print(f"Extraction: {config.get('extraction', '?')}")
        print(f"Embedding: {config.get('embedding_model', '?')}")
        print(f"Decay alpha: {config.get('recency_decay_alpha', '?')}")
        print(f"Max results: {config.get('max_results', '?')}")
        print(f"Candidates: {config.get('retrieval_candidates', '?')}")
        print()

    # Aggregate by type
    by_type: dict[str, list[bool]] = {}
    for r in results:
        by_type.setdefault(r["question_type"], []).append(r["correct"])

    # Print table
    header = f"{'Question Type':<30} | {'Correct':>7} | {'Total':>5} | {'Accuracy':>8}"
    sep = "-" * 30 + "-+-" + "-" * 7 + "-+-" + "-" * 5 + "-+-" + "-" * 8
    lines = [header, sep]

    total_correct = 0
    total_count = 0
    for qtype in sorted(by_type.keys()):
        vals = by_type[qtype]
        correct = sum(vals)
        count = len(vals)
        acc = 100.0 * correct / count if count else 0
        lines.append(f"{qtype:<30} | {correct:>7} | {count:>5} | {acc:>7.1f}%")
        total_correct += correct
        total_count += count

    lines.append(sep)
    overall = 100.0 * total_correct / total_count if total_count else 0
    lines.append(f"{'Overall':<30} | {total_correct:>7} | {total_count:>5} | {overall:>7.1f}%")

    report = "\n".join(lines)
    print(report)

    # Save
    report_file = run_dir / "report.txt"
    with open(report_file, "w") as f:
        f.write(report + "\n")
    print(f"\nSaved to {report_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="LongMemEvalLite — benchmark claude-memory against LongMemEval"
    )
    parser.add_argument(
        "phase",
        choices=["sample", "extract", "evaluate", "report", "all"],
        help="Which phase to run",
    )
    parser.add_argument("--run", help="Run name (for evaluate/report)")
    parser.add_argument("--extraction", help="Extraction name (for evaluate, e.g. sonnet_42)")
    parser.add_argument("--model", default=None, help="Model for extraction/answer generation")
    parser.add_argument("--judge-model", default=None, help="Model for LLM judge")
    parser.add_argument("--embedding-model", default=None, help="Embedding model name")
    parser.add_argument("--concurrency", type=int, default=2, help="Concurrent extraction calls")
    parser.add_argument("--max-results", type=int, default=DEFAULT_MAX_RESULTS, help="Top-K for search")
    parser.add_argument("--retrieval-candidates", type=int, default=RETRIEVAL_CANDIDATES, help="Initial search pool")
    parser.add_argument("--decay-alpha", type=float, default=RECENCY_DECAY_ALPHA, help="Recency decay per day")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    phase = args.phase

    if phase in ("sample", "all"):
        run_sample()

    if phase in ("extract", "all"):
        asyncio.run(run_extract(args))

    if phase in ("evaluate", "all"):
        if not args.run:
            parser.error("--run is required for evaluate")
        if not args.extraction:
            # Default extraction name from model
            model = args.model or DEFAULT_MODEL
            args.extraction = f"{model}_{SAMPLE_SEED}"
        asyncio.run(run_evaluate(args))

    if phase in ("report", "all"):
        if not args.run:
            parser.error("--run is required for report")
        run_report(args)


if __name__ == "__main__":
    main()
