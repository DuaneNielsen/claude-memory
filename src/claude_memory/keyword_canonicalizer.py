"""Canonicalize trajectory keywords against a project's existing keyword cloud.

The extractor tells the LLM to prefer existing keywords, but LLMs drift: you
can get "pipewire" in one session and "pipe-wire" in the next, or a brand-new
keyword that's a near-synonym of an existing one. We post-process each new
keyword by embedding it, comparing against existing keywords, and either:

  - auto-merging to the nearest existing keyword (similarity ≥ MERGE_THRESHOLD)
  - flagging it for periodic review (FLAG_THRESHOLD ≤ sim < MERGE_THRESHOLD)
  - accepting it as a new canonical keyword (similarity < FLAG_THRESHOLD)

Flagged cases are appended to a JSONL log so the user can review borderline
calls and tune thresholds or manually merge.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable, Iterable

from .config import (
    KEYWORD_FLAG_THRESHOLD,
    KEYWORD_FLAGS_LOG,
    KEYWORD_MERGE_THRESHOLD,
)

log = logging.getLogger(__name__)


@dataclass
class CanonicalizationResult:
    mapping: dict[str, str]  # raw_keyword -> canonical_keyword
    auto_merged: list[tuple[str, str, float]] = field(default_factory=list)
    flagged: list[tuple[str, str, float]] = field(default_factory=list)
    new_accepted: list[str] = field(default_factory=list)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def canonicalize_keywords(
    raw_keywords: Iterable[str],
    existing_keywords: Iterable[str],
    embed_fn: Callable[[list[str]], list[list[float]]],
    merge_threshold: float = KEYWORD_MERGE_THRESHOLD,
    flag_threshold: float = KEYWORD_FLAG_THRESHOLD,
) -> CanonicalizationResult:
    """Map each raw keyword to its canonical form.

    Args:
        raw_keywords: keywords emitted by the extractor (may duplicate or near-dup existing).
        existing_keywords: the project's current canonical keyword cloud.
        embed_fn: callable taking a list of strings → list of embeddings.
        merge_threshold: similarity at or above which a new keyword is auto-merged.
        flag_threshold: similarity below merge_threshold but at or above this is
            flagged for review (accepted as new, but logged).

    Returns a CanonicalizationResult whose `mapping` is a dict covering every
    input raw_keyword. Callers use it to rewrite keyword lists before persist.
    """
    raw_set = {k for k in raw_keywords if k}
    existing_set = {k for k in existing_keywords if k}

    # Keywords already in the cloud are trivially canonical
    mapping: dict[str, str] = {k: k for k in raw_set & existing_set}
    new_only = sorted(raw_set - existing_set)

    result = CanonicalizationResult(mapping=mapping)

    if not new_only:
        return result

    # If there are no existing keywords, everything new is accepted as-is
    if not existing_set:
        for kw in new_only:
            mapping[kw] = kw
            result.new_accepted.append(kw)
        return result

    existing_list = sorted(existing_set)
    # Embed existing + new in one call for efficiency
    all_embeds = embed_fn(existing_list + new_only)
    existing_embeds = all_embeds[: len(existing_list)]
    new_embeds = all_embeds[len(existing_list):]

    for kw, emb in zip(new_only, new_embeds):
        # Combined similarity: embedding catches semantic near-synonyms;
        # edit-distance catches punctuation/spelling variants that embeddings
        # can miss on short strings (e.g. "pipe-wire" vs "pipewire" embeds at
        # ~0.87 but edits at ~0.94). Take the max so either signal can fire.
        best_sim = -1.0
        best_kw = existing_list[0]
        for ex_kw, ex_emb in zip(existing_list, existing_embeds):
            embed_sim = _cosine(emb, ex_emb)
            edit_sim = SequenceMatcher(None, kw, ex_kw).ratio()
            sim = max(embed_sim, edit_sim)
            if sim > best_sim:
                best_sim = sim
                best_kw = ex_kw

        if best_sim >= merge_threshold:
            mapping[kw] = best_kw
            result.auto_merged.append((kw, best_kw, best_sim))
        elif best_sim >= flag_threshold:
            mapping[kw] = kw
            result.flagged.append((kw, best_kw, best_sim))
            result.new_accepted.append(kw)
        else:
            mapping[kw] = kw
            result.new_accepted.append(kw)

    return result


def append_flags(
    flagged: list[tuple[str, str, float]],
    project: str,
    flags_path: Path | None = None,
) -> None:
    """Append flagged keyword cases to the JSONL log for periodic review."""
    if not flagged:
        return
    path = Path(flags_path) if flags_path else KEYWORD_FLAGS_LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now().isoformat()
    with path.open("a") as f:
        for new_kw, nearest_kw, sim in flagged:
            rec = {
                "timestamp": now,
                "project": project,
                "new_keyword": new_kw,
                "nearest_existing": nearest_kw,
                "similarity": round(sim, 4),
            }
            f.write(json.dumps(rec) + "\n")


def apply_mapping(keywords: Iterable[str], mapping: dict[str, str]) -> list[str]:
    """Rewrite a keyword list using a canonicalization mapping, deduped, order-preserving."""
    out: list[str] = []
    seen: set[str] = set()
    for kw in keywords:
        canon = mapping.get(kw, kw)
        if canon not in seen:
            seen.add(canon)
            out.append(canon)
    return out
