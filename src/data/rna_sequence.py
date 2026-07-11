"""
RNA sequence utilities for validation, composition statistics, and similarity checks.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable

RNA_ALPHABET = "AUGC"
RNA_SET = frozenset(RNA_ALPHABET)


def normalize_rna(seq: str) -> str:
    """Uppercase, map T→U, strip whitespace."""
    return str(seq).strip().upper().replace("T", "U")


def validate_rna(seq: str) -> tuple[bool, str]:
    """Return (is_valid, cleaned_sequence). Invalid chars are removed."""
    cleaned = normalize_rna(seq)
    if len(cleaned) < 1:
        return False, cleaned
    invalid = {c for c in cleaned if c not in RNA_SET}
    if invalid:
        cleaned = "".join(c for c in cleaned if c in RNA_SET)
    if len(cleaned) < 1:
        return False, cleaned
    return True, cleaned


def gc_content(seq: str) -> float:
    seq = normalize_rna(seq)
    if not seq:
        return 0.0
    gc = sum(1 for c in seq if c in "GC")
    return gc / len(seq)


def nucleotide_counts(seq: str) -> Counter[str]:
    return Counter(normalize_rna(seq))


def dinucleotide_counts(seq: str, *, circular: bool = False) -> Counter[str]:
    """
    Count dinucleotides in a linear RNA sequence.

    When ``circular`` is True, also counts the wrap-around pair (last, first).
    """
    seq = normalize_rna(seq)
    counts: Counter[str] = Counter()
    if len(seq) < 2:
        return counts
    for i in range(len(seq) - 1):
        counts[seq[i : i + 2]] += 1
    if circular and len(seq) >= 2:
        counts[seq[-1] + seq[0]] += 1
    return counts


def hamming_distance(a: str, b: str) -> int:
    """Hamming distance for equal-length sequences; length mismatch → max possible edits."""
    a, b = normalize_rna(a), normalize_rna(b)
    if len(a) != len(b):
        return max(len(a), len(b))
    return sum(x != y for x, y in zip(a, b))


def min_required_hamming(length: int, *, min_absolute: int, min_fraction: float) -> int:
    if length <= 0:
        return 0
    return max(min_absolute, int(math.ceil(length * min_fraction)))


def passes_hamming_threshold(
    original: str,
    candidate: str,
    *,
    min_absolute: int,
    min_fraction: float,
) -> bool:
    original, candidate = normalize_rna(original), normalize_rna(candidate)
    if original == candidate:
        return False
    required = min_required_hamming(len(original), min_absolute=min_absolute, min_fraction=min_fraction)
    return hamming_distance(original, candidate) >= required


def gc_within_tolerance(original: str, candidate: str, *, tolerance: float) -> bool:
    return abs(gc_content(original) - gc_content(candidate)) <= tolerance


def kmer_jaccard(a: str, b: str, k: int = 3) -> float:
    """Jaccard similarity of k-mer sets (used for cross-pair partner ranking)."""
    a, b = normalize_rna(a), normalize_rna(b)

    def kmers(s: str) -> set[str]:
        if len(s) < k:
            return {s} if s else set()
        return {s[i : i + k] for i in range(len(s) - k + 1)}

    sa, sb = kmers(a), kmers(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def protein_kmer_jaccard(a: str, b: str, k: int = 3) -> float:
    """Jaccard on amino-acid k-mers for cross-protein partner ranking."""
    a, b = str(a).upper(), str(b).upper()
    if len(a) < k or len(b) < k:
        return 1.0 if a == b else 0.0

    sa = {a[i : i + k] for i in range(len(a) - k + 1)}
    sb = {b[i : i + k] for i in range(len(b) - k + 1)}
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def composition_preserved(original: str, candidate: str) -> bool:
    return nucleotide_counts(original) == nucleotide_counts(candidate)


def dinucleotide_profile_preserved(original: str, candidate: str, *, circular: bool = False) -> bool:
    return dinucleotide_counts(original, circular=circular) == dinucleotide_counts(
        candidate, circular=circular
    )


def pair_key(protein_sequence: str, rna_sequence: str) -> tuple[str, str]:
    return (str(protein_sequence).upper(), normalize_rna(rna_sequence))
