"""Protein name normalization for cross-protocol matching.

Design rules (strict — avoid silent wrong merges):
1. Within a protocol, never merge distinct native names via aliases.
2. Construct suffixes are stripped only for *candidate* gene matching;
   classifiers must still filter on the chosen native name string.
3. Synonyms (A2BP1↔RBFOX1, HuR↔ELAVL1) are applied only when they do not
   create a within-protocol collision (two natives → same key).
"""

from __future__ import annotations

import re

# Applied only via resolve_match_key(..., apply_synonyms=True) after collision checks.
SYNONYMS: dict[str, str] = {
    "A2BP1": "RBFOX1",
    "TDP43": "TARDBP",
    "HUR": "ELAVL1",  # HuR
    "HU": "ELAVL1",
}

_CONSTRUCT_RE = re.compile(
    r"(?:[-_]construct\d+|[-_]isoform\d+|[-_]trunc(?:ation)?\d*)$",
    re.IGNORECASE,
)
_NON_ALNUM_RE = re.compile(r"[^A-Z0-9]+")


def strip_construct_suffix(name: str) -> str:
    """Remove SELEX-style construct / isoform suffixes."""
    prev = name
    while True:
        nxt = _CONSTRUCT_RE.sub("", prev)
        if nxt == prev:
            return prev
        prev = nxt


def base_gene_key(name: str) -> str:
    """
    Canonical gene token without synonyms.

    Uppercase, strip construct/isoform suffixes, keep A–Z/0–9 only.
    """
    if name is None or (isinstance(name, float) and str(name) == "nan"):
        return ""
    s = str(name).strip()
    if not s or s.lower() == "nan":
        return ""
    s = strip_construct_suffix(s)
    s = s.upper()
    return _NON_ALNUM_RE.sub("", s)


def normalize_protein_name(name: str, aliases: dict[str, str] | None = None) -> str:
    """
    Back-compat wrapper.

    Default: base_gene_key only (NO synonyms). Pass aliases=SYNONYMS explicitly
    if you intentionally want synonym folding — prefer resolve_match_key().
    """
    key = base_gene_key(name)
    if not key:
        return ""
    if aliases:
        return aliases.get(key, key)
    return key


def resolve_match_key(
    name: str,
    *,
    keys_already_in_protocol: set[str] | None = None,
    apply_synonyms: bool = True,
) -> tuple[str, str]:
    """
    Return (match_key, match_rule).

    match_rule ∈ {exact_base, synonym, synonym_blocked_collision, empty}.

    If applying a synonym would collide with another native already mapped to
    the synonym target in the same protocol, keep the base key instead.
    """
    base = base_gene_key(name)
    if not base:
        return "", "empty"
    if not apply_synonyms or base not in SYNONYMS:
        return base, "exact_base"
    target = SYNONYMS[base]
    occupied = keys_already_in_protocol or set()
    # Collision: synonym target already present as a distinct entry's base key
    if target in occupied and target != base:
        return base, "synonym_blocked_collision"
    return target, "synonym"


def choose_representative_native(natives: list[str]) -> tuple[str, str]:
    """
    Pick one native name when several constructs map to the same gene in one protocol.

    Preference:
      1. Name without construct/isoform suffix
      2. Shorter name (fewer annotations)
      3. Lexicographic stability

    Returns (chosen_native, selection_rule).
    """
    if not natives:
        return "", "empty"
    uniq = sorted(set(str(n).strip() for n in natives if str(n).strip()))
    if len(uniq) == 1:
        return uniq[0], "unique"

    def is_plain(n: str) -> bool:
        return strip_construct_suffix(n) == n

    plain = [n for n in uniq if is_plain(n)]
    if len(plain) == 1:
        return plain[0], "prefer_no_construct_suffix"
    if len(plain) > 1:
        # Prefer human-typical casing length then alpha
        chosen = sorted(plain, key=lambda n: (len(n), n.upper()))[0]
        return chosen, "prefer_plain_shortest_stable"
    # All are constructs — prefer construct2-like middle? use shortest stable
    chosen = sorted(uniq, key=lambda n: (len(n), n.upper()))[0]
    return chosen, "prefer_construct_shortest_stable"


def coarse_domain_class(architecture: str) -> str:
    """
    Map Table S1 'Domains in construct' string to a coarse family label.

    Examples:
      'RRM;RRM;RRM' → 'RRM'
      'KH;KH' → 'KH'
      'RRM;KH' → 'multi'
      '' → 'unknown'
    """
    if architecture is None or (isinstance(architecture, float) and str(architecture) == "nan"):
        return "unknown"
    raw = str(architecture).strip()
    if not raw or raw.lower() == "nan":
        return "unknown"
    parts = [p.strip().upper() for p in raw.replace(",", ";").split(";") if p.strip()]
    if not parts:
        return "unknown"
    canon = []
    for p in parts:
        if p.startswith("RRM"):
            canon.append("RRM")
        elif p.startswith("KH"):
            canon.append("KH")
        elif "PUM" in p or p in {"PUMILIO", "PUF"}:
            canon.append("PUM")
        elif p in {"CCCH", "C3H", "ZNF_CCCH"} or "CCCH" in p:
            canon.append("CCCH")
        elif "ZINC" in p or p.startswith("ZF") or p in {"CCHC", "C2H2"}:
            canon.append("ZF")
        else:
            canon.append(p)
    uniq = sorted(set(canon))
    if len(uniq) == 1:
        return uniq[0]
    return "multi"
