"""
Negative example generation for protein–RNA binding benchmarks.

Strategies implemented
----------------------
shuffle_uniform
    Uniform random permutation preserving nucleotide composition (multiset shuffle).

shuffle_dinucleotide
    Random Eulerian trail over the directed dinucleotide multigraph (Altschul–Erickson
    style). Preserves all adjacent dinucleotide counts of the linear sequence.

cross_protein
    Keep the anchor RNA; pair with a non-cognate protein from another curated positive.
    Partner selection prefers low 3-mer Jaccard similarity between protein sequences.

cross_rna
    Keep the anchor protein; pair with a non-cognate RNA from another curated positive.
    Partner selection prefers low 3-mer Jaccard similarity between RNA sequences.

Each generator uses rejection sampling to enforce minimum Hamming distance from the
anchor (for shuffle strategies) and to avoid reproducing forbidden pair keys.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from src.data.rna_sequence import (
    composition_preserved,
    dinucleotide_profile_preserved,
    gc_within_tolerance,
    hamming_distance,
    kmer_jaccard,
    min_required_hamming,
    normalize_rna,
    pair_key,
    passes_hamming_threshold,
    protein_kmer_jaccard,
)


@dataclass(frozen=True)
class NegativeSamplingConfig:
    """Hyper-parameters for external benchmark negative generation."""

    seed: int = 42
    n_shuffle_uniform: int = 1
    n_shuffle_dinucleotide: int = 1
    n_cross_protein: int = 1
    n_cross_rna: int = 1
    min_hamming_absolute: int = 3
    min_hamming_fraction: float = 0.05
    gc_match_tolerance: float = 0.0  # 0 = composition-preserving shuffles already match GC
    max_attempts_per_draw: int = 500
    min_protein_jaccard_distance: float = 0.05  # cross_protein partners must differ by at least this
    min_rna_jaccard_distance: float = 0.05


@dataclass
class CuratedPair:
    pair_id: str
    protein_name: str
    protein_sequence: str
    rna_sequence: str
    binding_label: int
    source_row: int | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class GeneratedNegative:
    pair_id: str
    parent_pair_id: str
    protein_name: str
    protein_sequence: str
    rna_sequence: str
    neg_strategy: str
    generation_seed: int
    hamming_to_parent_rna: int | None
    partner_pair_id: str | None
    rejection_attempts: int
    metadata: dict = field(default_factory=dict)


def _derive_seed(base_seed: int, *parts: str) -> int:
    payload = "|".join([str(base_seed)] + list(parts))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def uniform_shuffle(seq: str, rng: np.random.Generator) -> str:
    """Random permutation of nucleotides (composition preserving)."""
    chars = list(normalize_rna(seq))
    rng.shuffle(chars)
    return "".join(chars)


def dinucleotide_shuffle(seq: str, rng: np.random.Generator) -> str | None:
    """
    Sample a sequence with identical linear dinucleotide counts.

    Builds the directed multigraph of adjacent pairs and draws a random Eulerian trail
    from first to last nucleotide. Returns None if no trail is found for this edge order.
    """
    seq = normalize_rna(seq)
    n = len(seq)
    if n < 2:
        return seq
    if n == 2:
        return seq[::-1] if seq[0] != seq[1] else None

    edge_counts: Counter[tuple[str, str]] = Counter()
    for i in range(n - 1):
        edge_counts[(seq[i], seq[i + 1])] += 1

    graph_lists: dict[str, list[str]] = defaultdict(list)
    for (u, v), count in edge_counts.items():
        graph_lists[u].extend([v] * count)
    for node in graph_lists:
        rng.shuffle(graph_lists[node])

    graph = {k: deque(v) for k, v in graph_lists.items()}
    start, end = seq[0], seq[-1]

    stack = [start]
    path: list[str] = []
    while stack:
        v = stack[-1]
        if graph.get(v):
            u = graph[v].popleft()
            stack.append(u)
        else:
            path.append(stack.pop())

    path.reverse()
    if len(path) != n:
        return None
    if path[0] != start or path[-1] != end:
        return None

    result = "".join(path)
    if result == seq:
        return None
    if not dinucleotide_profile_preserved(seq, result):
        return None
    return result


def _reject_shuffle(
    original_rna: str,
    sampler: Callable[[str, np.random.Generator], str | None],
    rng: np.random.Generator,
    forbidden_rnas: set[str],
    cfg: NegativeSamplingConfig,
) -> tuple[str | None, int]:
    """Rejection sampling wrapper for shuffle-based generators."""
    original_rna = normalize_rna(original_rna)
    for attempt in range(cfg.max_attempts_per_draw):
        candidate = sampler(original_rna, rng)
        if candidate is None:
            continue
        candidate = normalize_rna(candidate)
        if candidate in forbidden_rnas:
            continue
        if not passes_hamming_threshold(
            original_rna,
            candidate,
            min_absolute=cfg.min_hamming_absolute,
            min_fraction=cfg.min_hamming_fraction,
        ):
            continue
        if cfg.gc_match_tolerance > 0 and not gc_within_tolerance(
            original_rna, candidate, tolerance=cfg.gc_match_tolerance
        ):
            continue
        if not composition_preserved(original_rna, candidate):
            continue
        return candidate, attempt + 1
    return None, cfg.max_attempts_per_draw


def _rank_cross_protein_partners(anchor: CuratedPair, pool: list[CuratedPair]) -> list[CuratedPair]:
    candidates = [
        p
        for p in pool
        if p.binding_label == 1
        and p.pair_id != anchor.pair_id
        and p.protein_name != anchor.protein_name
        and p.protein_sequence != anchor.protein_sequence
    ]
    if not candidates:
        return []

    scored = []
    for p in candidates:
        sim = protein_kmer_jaccard(anchor.protein_sequence, p.protein_sequence, k=3)
        dist = 1.0 - sim
        scored.append((dist, p))
    scored.sort(key=lambda x: (-x[0], x[1].pair_id))
    return [p for _, p in scored]


def _rank_cross_rna_partners(anchor: CuratedPair, pool: list[CuratedPair]) -> list[CuratedPair]:
    candidates = [
        p
        for p in pool
        if p.binding_label == 1
        and p.pair_id != anchor.pair_id
        and normalize_rna(p.rna_sequence) != normalize_rna(anchor.rna_sequence)
    ]
    if not candidates:
        return []

    scored = []
    for p in candidates:
        sim = kmer_jaccard(anchor.rna_sequence, p.rna_sequence, k=3)
        dist = 1.0 - sim
        scored.append((dist, p))
    scored.sort(key=lambda x: (-x[0], x[1].pair_id))
    return [p for _, p in scored]


def generate_shuffle_uniform(
    anchor: CuratedPair,
    draw_index: int,
    forbidden_rnas: set[str],
    cfg: NegativeSamplingConfig,
) -> GeneratedNegative | None:
    seed = _derive_seed(cfg.seed, anchor.pair_id, "shuffle_uniform", str(draw_index))
    rng = np.random.default_rng(seed)

    def sampler(original: str, r: np.random.Generator) -> str | None:
        return uniform_shuffle(original, r)

    candidate, attempts = _reject_shuffle(anchor.rna_sequence, sampler, rng, forbidden_rnas, cfg)
    if candidate is None:
        return None

    return GeneratedNegative(
        pair_id=f"{anchor.pair_id}__neg_shuffle_uniform_{draw_index}",
        parent_pair_id=anchor.pair_id,
        protein_name=anchor.protein_name,
        protein_sequence=anchor.protein_sequence,
        rna_sequence=candidate,
        neg_strategy="shuffle_uniform",
        generation_seed=seed,
        hamming_to_parent_rna=hamming_distance(anchor.rna_sequence, candidate),
        partner_pair_id=None,
        rejection_attempts=attempts,
        metadata={
            "parent_rna_length": len(normalize_rna(anchor.rna_sequence)),
            "gc_content": float(
                sum(1 for c in candidate if c in "GC") / max(len(candidate), 1)
            ),
        },
    )


def generate_shuffle_dinucleotide(
    anchor: CuratedPair,
    draw_index: int,
    forbidden_rnas: set[str],
    cfg: NegativeSamplingConfig,
) -> GeneratedNegative | None:
    seed = _derive_seed(cfg.seed, anchor.pair_id, "shuffle_dinucleotide", str(draw_index))
    rng = np.random.default_rng(seed)
    original = normalize_rna(anchor.rna_sequence)

    candidate: str | None = None
    attempts = 0
    for attempt in range(cfg.max_attempts_per_draw):
        attempts = attempt + 1
        # Re-seed each attempt so retries are independent but reproducible
        attempt_rng = np.random.default_rng(_derive_seed(seed, str(attempt)))
        proposal = dinucleotide_shuffle(original, attempt_rng)
        if proposal is None:
            continue
        proposal = normalize_rna(proposal)
        if proposal in forbidden_rnas:
            continue
        if not passes_hamming_threshold(
            original,
            proposal,
            min_absolute=cfg.min_hamming_absolute,
            min_fraction=cfg.min_hamming_fraction,
        ):
            continue
        if not dinucleotide_profile_preserved(original, proposal):
            continue
        candidate = proposal
        break

    if candidate is None:
        return None

    return GeneratedNegative(
        pair_id=f"{anchor.pair_id}__neg_shuffle_dinucleotide_{draw_index}",
        parent_pair_id=anchor.pair_id,
        protein_name=anchor.protein_name,
        protein_sequence=anchor.protein_sequence,
        rna_sequence=candidate,
        neg_strategy="shuffle_dinucleotide",
        generation_seed=seed,
        hamming_to_parent_rna=hamming_distance(original, candidate),
        partner_pair_id=None,
        rejection_attempts=attempts,
        metadata={
            "dinucleotide_profile_preserved": True,
            "parent_rna_length": len(original),
        },
    )


def generate_cross_protein(
    anchor: CuratedPair,
    draw_index: int,
    positive_pool: list[CuratedPair],
    forbidden_pairs: set[tuple[str, str]],
    cfg: NegativeSamplingConfig,
) -> GeneratedNegative | None:
    ranked = _rank_cross_protein_partners(anchor, positive_pool)
    if not ranked:
        return None

    seed = _derive_seed(cfg.seed, anchor.pair_id, "cross_protein", str(draw_index))
    rng = np.random.default_rng(seed)

    best = ranked[0]
    best_dist = 1.0 - protein_kmer_jaccard(anchor.protein_sequence, best.protein_sequence, k=3)
    tier = [p for p in ranked if (1.0 - protein_kmer_jaccard(anchor.protein_sequence, p.protein_sequence, k=3))
            >= max(cfg.min_protein_jaccard_distance, best_dist - 1e-9)]
    if not tier:
        tier = ranked[:1]

    for offset in range(len(tier)):
        partner = tier[(draw_index + offset) % len(tier)]
        key = pair_key(partner.protein_sequence, anchor.rna_sequence)
        if key in forbidden_pairs:
            continue
        if normalize_rna(anchor.rna_sequence) == normalize_rna(partner.rna_sequence):
            continue

        return GeneratedNegative(
            pair_id=f"{anchor.pair_id}__neg_cross_protein_{draw_index}",
            parent_pair_id=anchor.pair_id,
            protein_name=partner.protein_name,
            protein_sequence=partner.protein_sequence,
            rna_sequence=anchor.rna_sequence,
            neg_strategy="cross_protein",
            generation_seed=seed,
            hamming_to_parent_rna=None,
            partner_pair_id=partner.pair_id,
            rejection_attempts=offset + 1,
            metadata={
                "partner_protein_name": partner.protein_name,
                "protein_jaccard_distance": 1.0 - protein_kmer_jaccard(
                    anchor.protein_sequence, partner.protein_sequence, k=3
                ),
                "partner_tier_size": len(tier),
                "rng_offset": int(rng.integers(0, 2**31 - 1)),
            },
        )
    return None


def generate_cross_rna(
    anchor: CuratedPair,
    draw_index: int,
    positive_pool: list[CuratedPair],
    forbidden_pairs: set[tuple[str, str]],
    cfg: NegativeSamplingConfig,
) -> GeneratedNegative | None:
    ranked = _rank_cross_rna_partners(anchor, positive_pool)
    if not ranked:
        return None

    seed = _derive_seed(cfg.seed, anchor.pair_id, "cross_rna", str(draw_index))
    rng = np.random.default_rng(seed)

    best = ranked[0]
    best_dist = 1.0 - kmer_jaccard(anchor.rna_sequence, best.rna_sequence, k=3)
    tier = [p for p in ranked if (1.0 - kmer_jaccard(anchor.rna_sequence, p.rna_sequence, k=3))
            >= max(cfg.min_rna_jaccard_distance, best_dist - 1e-9)]
    if not tier:
        tier = ranked[:1]

    for offset in range(len(tier)):
        partner = tier[(draw_index + offset) % len(tier)]
        key = pair_key(anchor.protein_sequence, partner.rna_sequence)
        if key in forbidden_pairs:
            continue
        if anchor.protein_name == partner.protein_name:
            continue

        return GeneratedNegative(
            pair_id=f"{anchor.pair_id}__neg_cross_rna_{draw_index}",
            parent_pair_id=anchor.pair_id,
            protein_name=anchor.protein_name,
            protein_sequence=anchor.protein_sequence,
            rna_sequence=partner.rna_sequence,
            neg_strategy="cross_rna",
            generation_seed=seed,
            hamming_to_parent_rna=hamming_distance(anchor.rna_sequence, partner.rna_sequence),
            partner_pair_id=partner.pair_id,
            rejection_attempts=offset + 1,
            metadata={
                "partner_protein_name": partner.protein_name,
                "partner_pair_id": partner.pair_id,
                "rna_jaccard_distance": 1.0 - kmer_jaccard(
                    anchor.rna_sequence, partner.rna_sequence, k=3
                ),
                "partner_tier_size": len(tier),
                "rng_offset": int(rng.integers(0, 2**31 - 1)),
            },
        )
    return None


def generate_all_negatives_for_anchor(
    anchor: CuratedPair,
    positive_pool: list[CuratedPair],
    forbidden_pairs: set[tuple[str, str]],
    forbidden_rnas_for_protein: set[str],
    cfg: NegativeSamplingConfig,
) -> tuple[list[GeneratedNegative], dict[str, int]]:
    """
    Generate all configured negative types for one curated positive anchor.

    Returns (negatives, failure_counts_by_strategy).
    """
    if anchor.binding_label != 1:
        return [], {}

    generated: list[GeneratedNegative] = []
    failures: Counter[str] = Counter()
    used_rnas = set(forbidden_rnas_for_protein)
    used_pairs = set(forbidden_pairs)

    def accept(neg: GeneratedNegative | None, strategy: str) -> None:
        if neg is None:
            failures[strategy] += 1
            return
        key = pair_key(neg.protein_sequence, neg.rna_sequence)
        if key in used_pairs:
            failures[f"{strategy}_duplicate"] += 1
            return
        generated.append(neg)
        used_pairs.add(key)
        used_rnas.add(normalize_rna(neg.rna_sequence))

    for i in range(cfg.n_shuffle_uniform):
        accept(
            generate_shuffle_uniform(anchor, i, used_rnas, cfg),
            "shuffle_uniform",
        )

    for i in range(cfg.n_shuffle_dinucleotide):
        accept(
            generate_shuffle_dinucleotide(anchor, i, used_rnas, cfg),
            "shuffle_dinucleotide",
        )

    for i in range(cfg.n_cross_protein):
        accept(
            generate_cross_protein(anchor, i, positive_pool, used_pairs, cfg),
            "cross_protein",
        )

    for i in range(cfg.n_cross_rna):
        accept(
            generate_cross_rna(anchor, i, positive_pool, used_pairs, cfg),
            "cross_rna",
        )

    return generated, dict(failures)
