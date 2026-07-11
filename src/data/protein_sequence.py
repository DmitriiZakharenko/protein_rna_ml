"""
Protein sequence cleaning and validation for export deliverables.
"""

from __future__ import annotations

import re

STANDARD_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
# Common extended codes kept when strict=False
EXTENDED_AA = frozenset("XBZJUO")


def sanitize_protein_sequence(
    seq: str,
    *,
    strict: bool = True,
    source: str = "",
) -> tuple[str, list[str]]:
    """
    Clean a protein sequence for FASTA export.

    - Uppercase
    - Strip trailing stop codon markers (*)
    - Remove digits and other non-amino-acid characters
    - In strict mode keep only 20 standard amino acids

    Returns (cleaned_sequence, warning_messages).
    """
    warnings: list[str] = []
    if not seq or not str(seq).strip():
        return "", ["empty_sequence"] if source else ["empty_sequence"]

    raw = str(seq).strip().upper()
    prefix = f"{source}: " if source else ""

    if raw.endswith("*"):
        raw = raw.rstrip("*")
        warnings.append(f"{prefix}stripped_trailing_stop_codon")

    allowed = STANDARD_AA if strict else (STANDARD_AA | EXTENDED_AA)
    removed_digits = bool(re.search(r"\d", raw))
    removed_other = False
    cleaned_chars: list[str] = []

    for ch in raw:
        if ch in allowed:
            cleaned_chars.append(ch)
        elif ch == "*":
            removed_other = True
        elif ch.isdigit():
            continue
        elif ch.isspace():
            continue
        else:
            removed_other = True

    if removed_digits:
        warnings.append(f"{prefix}removed_embedded_digits")
    if removed_other:
        warnings.append(f"{prefix}removed_non_amino_acid_characters")

    cleaned = "".join(cleaned_chars)
    if not cleaned:
        warnings.append(f"{prefix}empty_after_sanitization")
    elif len(cleaned) != len(raw):
        warnings.append(f"{prefix}length_changed_{len(raw)}_to_{len(cleaned)}")

    return cleaned, warnings


def validate_protein_sequence(seq: str, *, strict: bool = True) -> tuple[bool, str]:
    """Return (is_valid, reason)."""
    if not seq:
        return False, "empty"
    allowed = STANDARD_AA if strict else (STANDARD_AA | EXTENDED_AA)
    bad = sorted({c for c in seq.upper() if c not in allowed})
    if bad:
        return False, f"invalid_characters:{''.join(bad)}"
    return True, "ok"
