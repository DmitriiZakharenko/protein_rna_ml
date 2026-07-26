"""
Table S1 construct sequences + domain intervals for domain-aware V2.

Coordinate-frame rules (do not violate):
1. Table S1 Domain Boundaries are 1-based indices into Construct AA seq.
2. UniProt FT intervals are full-length UniProt coordinates — never mix with (1).
3. Training `protein_sequence` may equal, contain, be contained in, or be
   unrelated to the Table S1 construct. Blind interval transfer onto train
   sequences is only allowed after an explicit alignment relation.

Join aliases (roster key → Table S1 protein_key) match script 37.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.data.protein_names import base_gene_key, coarse_domain_class
from src.data.protein_sequence import sanitize_protein_sequence

# Roster gene key → Table S1 gene key (same row, different spelling).
TABLE_S1_JOIN_ALIASES: dict[str, str] = {
    "HNRPLL": "HNRNPLL",
    "RBFOX1": "A2BP1",
    "PUM1": "PUM",
}

# Relations where construct ↔ train seq share a clear substring relationship.
ALIGNED_RELATIONS = frozenset(
    {"exact", "construct_in_train", "train_in_construct"}
)

# Safe to map construct-indexed intervals onto the *training* sequence.
SAFE_TRAIN_MASK_RELATIONS = frozenset({"exact", "construct_in_train"})


@dataclass(frozen=True)
class ConstructRecord:
    protein_key: str
    protein_name_native: str
    species: str
    domain_architecture: str
    domain_class: str
    domain_intervals: tuple[tuple[int, int], ...]  # 1-based inclusive on construct
    construct_seq: str
    construct_seq_column: str
    mean_top10_z: float
    rnacompete_ids: str


@dataclass
class AlignmentHit:
    protein_name: str
    protein_key: str
    dataset_source: str
    train_seq: str
    train_len: int
    relation: str  # exact|construct_in_train|train_in_construct|disjoint|no_s1
    construct_seq: str
    construct_len: int
    construct_offset_in_train: int  # 0-based; -1 if not construct_in_train/exact
    intervals: tuple[tuple[int, int], ...]
    intervals_fit_construct: bool
    safe_train_mask: bool
    s1_native: str
    species: str
    domain_architecture: str
    domain_class: str
    mean_top10_z: float
    n_s1_candidates: int
    notes: list[str] = field(default_factory=list)
    variant_id: str = ""  # stable id for (name, seq) collisions


def variant_key(protein_name: str, protein_sequence: str) -> str:
    """
    Unique id for a concrete protein sequence instance.

    CRITICAL: generalized_v3a reuses the same protein_name with different
    sequences across (and even within) splits — e.g. HNRNPA1 val has both
    SELEX 320aa and RNAcompete 226aa. Never key remaps by name alone.
    """
    seq, _ = sanitize_protein_sequence(protein_sequence, strict=True)
    return f"{protein_name}||{seq}"



def parse_domain_boundaries(raw) -> list[tuple[int, int]]:
    """Parse Table S1 'Domain Boundaries' like '46;125;126;213' → 1-based pairs."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []
    s = str(raw).strip()
    if not s or s.lower() == "nan":
        return []
    parts: list[int] = []
    for tok in s.replace(",", ";").split(";"):
        tok = tok.strip()
        if not tok:
            continue
        try:
            parts.append(int(float(tok)))
        except ValueError:
            continue
    pairs: list[tuple[int, int]] = []
    for i in range(0, len(parts) - 1, 2):
        a, b = parts[i], parts[i + 1]
        if b >= a > 0:
            pairs.append((a, b))
    return pairs


def parse_mean_z(raw) -> float:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return float("-inf")
    s = str(raw).strip()
    if not s or s.lower() == "nan":
        return float("-inf")
    tok = s.split(",")[0].strip()
    try:
        return float(tok)
    except ValueError:
        return float("-inf")


def _extract_construct_seq(row: pd.Series) -> tuple[str, str]:
    """Prefer Construct AA seq; fall back to RBD/RBR. Sanitize strictly."""
    for col in ("Construct AA seq", "RBD or RBR AA Sequence"):
        if col not in row.index or pd.isna(row[col]):
            continue
        cleaned, _ = sanitize_protein_sequence(str(row[col]), strict=True, source=col)
        if cleaned:
            return cleaned, col
    return "", ""


def load_table_s1_constructs(path: str | Path) -> list[ConstructRecord]:
    """Load every Table S1 row that has a usable construct AA sequence."""
    ts1 = pd.read_excel(path)
    out: list[ConstructRecord] = []
    for _, r in ts1.iterrows():
        pname = str(r["Protein name"]).strip() if pd.notna(r.get("Protein name")) else ""
        if not pname:
            continue
        seq, scol = _extract_construct_seq(r)
        if not seq:
            continue
        pairs = tuple(parse_domain_boundaries(r.get("Domain Boundaries")))
        arch = (
            str(r["Domains in construct"]).strip()
            if pd.notna(r.get("Domains in construct"))
            else ""
        )
        species = str(r["Species"]).strip() if pd.notna(r.get("Species")) else ""
        rnids = (
            str(r["RNAcompete ID(s)"]).strip()
            if pd.notna(r.get("RNAcompete ID(s)"))
            else ""
        )
        out.append(
            ConstructRecord(
                protein_key=base_gene_key(pname),
                protein_name_native=pname,
                species=species,
                domain_architecture=arch,
                domain_class=coarse_domain_class(arch),
                domain_intervals=pairs,
                construct_seq=seq,
                construct_seq_column=scol,
                mean_top10_z=parse_mean_z(r.get("Average Z-score of top 10 7-mers")),
                rnacompete_ids=rnids,
            )
        )
    return out


def s1_lookup_keys(protein_key: str) -> list[str]:
    """Keys to try in Table S1 for a training/roster protein_key."""
    keys = [protein_key]
    if protein_key in TABLE_S1_JOIN_ALIASES:
        keys.append(TABLE_S1_JOIN_ALIASES[protein_key])
    # Reverse: training may already use the S1 spelling (A2BP1, HNRNPLL, PUM)
    for roster_key, s1_key in TABLE_S1_JOIN_ALIASES.items():
        if protein_key == s1_key:
            keys.append(roster_key)
    # Stable unique
    seen: set[str] = set()
    out: list[str] = []
    for k in keys:
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


def index_constructs_by_key(
    records: Iterable[ConstructRecord],
) -> dict[str, list[ConstructRecord]]:
    by_key: dict[str, list[ConstructRecord]] = {}
    for rec in records:
        by_key.setdefault(rec.protein_key, []).append(rec)
    return by_key


def score_construct_vs_train(construct: str, train: str) -> tuple[str, int]:
    """
    Return (relation, construct_offset_in_train).

    offset is 0-based index of construct inside train for exact/construct_in_train;
    -1 otherwise.
    """
    if not construct or not train:
        return "disjoint", -1
    if train == construct:
        return "exact", 0
    if construct in train:
        return "construct_in_train", train.index(construct)
    if train in construct:
        return "train_in_construct", -1
    return "disjoint", -1


def intervals_fit_sequence(
    intervals: tuple[tuple[int, int], ...] | list[tuple[int, int]],
    seq_len: int,
) -> bool:
    if not intervals:
        return False
    return all(1 <= a <= b <= seq_len for a, b in intervals)


def pick_best_construct(
    candidates: list[ConstructRecord],
    train_seq: str,
) -> tuple[ConstructRecord, str, int, list[str]]:
    """
    Prefer constructs that align to train_seq; then Homo sapiens; then highest Z.

    Ranking:
      1. relation tier: exact > construct_in_train > train_in_construct > disjoint
      2. human species
      3. intervals fit construct
      4. mean_top10_z
    """
    notes: list[str] = []
    if not candidates:
        raise ValueError("pick_best_construct called with empty candidates")

    tier = {
        "exact": 0,
        "construct_in_train": 1,
        "train_in_construct": 2,
        "disjoint": 3,
    }

    scored = []
    for rec in candidates:
        rel, off = score_construct_vs_train(rec.construct_seq, train_seq)
        human = int(rec.species.lower() == "homo sapiens")
        fit = int(intervals_fit_sequence(rec.domain_intervals, len(rec.construct_seq)))
        scored.append((tier[rel], -human, -fit, -rec.mean_top10_z, rec, rel, off))

    scored.sort(key=lambda x: x[:4])
    best = scored[0]
    # Ambiguity note if multiple natives with different architectures among top relation
    top_rel = best[5]
    same_rel = [s for s in scored if s[5] == top_rel]
    archs = {s[4].domain_architecture for s in same_rel}
    if len(archs) > 1:
        notes.append(f"ambiguous_architectures_in_relation:{top_rel}")
    natives = {s[4].protein_name_native for s in same_rel}
    if len(natives) > 1:
        notes.append(f"n_s1_natives_same_relation={len(natives)}")

    return best[4], best[5], best[6], notes


def align_protein_to_s1(
    protein_name: str,
    protein_sequence: str,
    by_key: dict[str, list[ConstructRecord]],
    dataset_source: str = "",
) -> AlignmentHit:
    key = base_gene_key(protein_name)
    train_seq, _ = sanitize_protein_sequence(protein_sequence, strict=True)
    vid = variant_key(protein_name, train_seq)
    candidates: list[ConstructRecord] = []
    for k in s1_lookup_keys(key):
        candidates.extend(by_key.get(k, []))

    if not candidates:
        return AlignmentHit(
            protein_name=protein_name,
            protein_key=key,
            dataset_source=dataset_source,
            train_seq=train_seq,
            train_len=len(train_seq),
            relation="no_s1",
            construct_seq="",
            construct_len=0,
            construct_offset_in_train=-1,
            intervals=tuple(),
            intervals_fit_construct=False,
            safe_train_mask=False,
            s1_native="",
            species="",
            domain_architecture="",
            domain_class="unknown",
            mean_top10_z=float("nan"),
            n_s1_candidates=0,
            notes=["no_table_s1_match"],
            variant_id=vid,
        )

    rec, rel, off, notes = pick_best_construct(candidates, train_seq)
    fit = intervals_fit_sequence(rec.domain_intervals, len(rec.construct_seq))
    if not fit:
        notes.append("intervals_do_not_fit_construct")
    if rel == "construct_in_train" and train_seq.count(rec.construct_seq) > 1:
        notes.append("construct_substring_not_unique")
        safe = False
    else:
        safe = rel in SAFE_TRAIN_MASK_RELATIONS and fit and (
            rel != "construct_in_train" or train_seq.count(rec.construct_seq) == 1
        )

    return AlignmentHit(
        protein_name=protein_name,
        protein_key=key,
        dataset_source=dataset_source,
        train_seq=train_seq,
        train_len=len(train_seq),
        relation=rel,
        construct_seq=rec.construct_seq,
        construct_len=len(rec.construct_seq),
        construct_offset_in_train=off,
        intervals=rec.domain_intervals,
        intervals_fit_construct=fit,
        safe_train_mask=safe,
        s1_native=rec.protein_name_native,
        species=rec.species,
        domain_architecture=rec.domain_architecture,
        domain_class=rec.domain_class,
        mean_top10_z=rec.mean_top10_z if rec.mean_top10_z != float("-inf") else float("nan"),
        n_s1_candidates=len(candidates),
        notes=notes,
        variant_id=vid,
    )


def mask_outside_intervals(
    seq: str,
    intervals_1based: tuple[tuple[int, int], ...] | list[tuple[int, int]],
    *,
    mask_char: str = "X",
) -> str:
    """
    Keep residues inside 1-based inclusive intervals; set others to mask_char.

    mask_char must NOT be in the 20-AA alphabet so one-hot encodes to all zeros
    (same as padding). Default 'X' is stripped by strict sanitization elsewhere
    but is intentional here for the CNN input path.
    """
    if not seq:
        return seq
    if not intervals_1based:
        return seq
    keep = [False] * len(seq)
    for a, b in intervals_1based:
        lo = max(1, a)
        hi = min(len(seq), b)
        for i in range(lo, hi + 1):
            keep[i - 1] = True
    return "".join(ch if keep[i] else mask_char for i, ch in enumerate(seq))


def shift_intervals(
    intervals: tuple[tuple[int, int], ...] | list[tuple[int, int]],
    offset_0based: int,
) -> tuple[tuple[int, int], ...]:
    """Shift construct-indexed 1-based intervals by a 0-based offset into train."""
    return tuple((a + offset_0based, b + offset_0based) for a, b in intervals)


def is_length_reducing_unique_substring(hit: AlignmentHit) -> bool:
    """
    True iff Table S1 construct is a unique contiguous substring of train
    AND construct is strictly shorter than train.

    This is the only case where construct_replace implements
    full-length → shorter construct (the intended ablation).
    """
    if hit.relation != "construct_in_train":
        return False
    if not hit.construct_seq or hit.train_len <= hit.construct_len:
        return False
    if hit.train_seq.count(hit.construct_seq) != 1:
        return False
    if hit.construct_offset_in_train < 0:
        return False
    return True


def resolve_protein_input(
    hit: AlignmentHit,
    mode: str,
    *,
    allow_unaligned_replace: bool = False,
    replace_policy: str = "length_reducing_substring",
) -> tuple[str, str]:
    """
    Return (protein_sequence_for_model, apply_status).

    replace_policy (construct_replace / construct_domain_mask only)
    ---------------------------------------------------------------
    length_reducing_substring  (DEFAULT — scientifically intended ablation)
        Replace ONLY when construct is a unique strict substring of train
        and shorter than train. exact / train_in_construct / disjoint →
        keep train (status noop_*). RNAcompete rows that already equal
        construct are unchanged in BOTH arms — they do not drive the contrast.
    aligned_any
        Legacy: replace for exact | construct_in_train | train_in_construct.
        WARNING: on v3a most eligible rows are exact (RNAcompete already
        construct), so full_length vs construct_replace is nearly a no-op.

    Modes
    -----
    full_length
        Unchanged training sequence.
    construct_replace
        See replace_policy.
    construct_domain_mask
        Same gate as construct_replace, then zero outside domain intervals
        on the construct coordinate frame.
    train_domain_mask
        Zero outside domains on the training sequence only when
        safe_train_mask. Independent of replace_policy.
    """
    if replace_policy not in {"length_reducing_substring", "aligned_any"}:
        raise ValueError(f"Unknown replace_policy={replace_policy!r}")

    if mode == "full_length":
        return hit.train_seq, "full_length"

    if mode in {"construct_replace", "construct_domain_mask"}:
        if hit.relation == "no_s1" or not hit.construct_seq:
            return hit.train_seq, "noop_no_s1"

        if replace_policy == "length_reducing_substring":
            if not is_length_reducing_unique_substring(hit):
                if hit.relation == "exact":
                    return hit.train_seq, "noop_exact_already_construct"
                if hit.relation == "train_in_construct":
                    return hit.train_seq, "noop_would_lengthen"
                if hit.relation == "construct_in_train":
                    return hit.train_seq, "noop_substring_not_unique_or_not_shorter"
                if hit.relation == "disjoint":
                    if allow_unaligned_replace:
                        # still refused under length_reducing policy
                        return hit.train_seq, "noop_disjoint_blocked_by_policy"
                    return hit.train_seq, "noop_disjoint"
                return hit.train_seq, f"noop_{hit.relation}"

            if mode == "construct_replace":
                return hit.construct_seq, "construct_replace:length_reducing_substring"

            # construct_domain_mask
            if not hit.intervals_fit_construct or not hit.intervals:
                return hit.construct_seq, "construct_replace_no_intervals"
            masked = mask_outside_intervals(hit.construct_seq, hit.intervals)
            return masked, "construct_domain_mask:length_reducing_substring"

        # aligned_any (legacy)
        if hit.relation in ALIGNED_RELATIONS:
            if mode == "construct_replace":
                return hit.construct_seq, f"construct_replace:{hit.relation}"
            if not hit.intervals_fit_construct or not hit.intervals:
                return hit.construct_seq, "construct_replace_no_intervals"
            masked = mask_outside_intervals(hit.construct_seq, hit.intervals)
            return masked, f"construct_domain_mask:{hit.relation}"
        if allow_unaligned_replace and hit.relation == "disjoint":
            if mode == "construct_replace":
                return hit.construct_seq, "construct_replace:disjoint_forced"
            if not hit.intervals_fit_construct or not hit.intervals:
                return hit.construct_seq, "construct_replace_no_intervals"
            masked = mask_outside_intervals(hit.construct_seq, hit.intervals)
            return masked, "construct_domain_mask:disjoint_forced"
        return hit.train_seq, "fallback_unaligned"

    if mode == "train_domain_mask":
        if not hit.safe_train_mask or not hit.intervals:
            if hit.relation == "no_s1":
                return hit.train_seq, "fallback_no_s1"
            return hit.train_seq, "fallback_unsafe_train_mask"
        intervals = hit.intervals
        if hit.relation == "construct_in_train":
            intervals = shift_intervals(hit.intervals, hit.construct_offset_in_train)
        if not intervals_fit_sequence(intervals, len(hit.train_seq)):
            return hit.train_seq, "fallback_intervals_out_of_range"
        masked = mask_outside_intervals(hit.train_seq, intervals)
        return masked, f"train_domain_mask:{hit.relation}"

    raise ValueError(
        f"Unknown mode={mode!r}; expected full_length|construct_replace|"
        f"construct_domain_mask|train_domain_mask"
    )


def build_alignment_table(
    unique_variants: pd.DataFrame,
    by_key: dict[str, list[ConstructRecord]],
    *,
    name_col: str = "protein_name",
    seq_col: str = "protein_sequence",
    source_col: str = "dataset_source",
) -> pd.DataFrame:
    """Align each unique (protein_name, protein_sequence) variant to Table S1."""
    rows = []
    for _, r in unique_variants.iterrows():
        src = str(r[source_col]) if source_col in r.index and pd.notna(r.get(source_col)) else ""
        hit = align_protein_to_s1(
            str(r[name_col]),
            str(r[seq_col]),
            by_key,
            dataset_source=src,
        )
        rows.append(
            {
                "variant_id": hit.variant_id,
                "protein_name": hit.protein_name,
                "protein_key": hit.protein_key,
                "dataset_source": hit.dataset_source,
                "relation": hit.relation,
                "train_len": hit.train_len,
                "construct_len": hit.construct_len,
                "construct_offset_in_train": hit.construct_offset_in_train,
                "intervals_fit_construct": int(hit.intervals_fit_construct),
                "safe_train_mask": int(hit.safe_train_mask),
                "n_intervals": len(hit.intervals),
                "domain_intervals": ";".join(f"{a}-{b}" for a, b in hit.intervals),
                "s1_native": hit.s1_native,
                "species": hit.species,
                "domain_architecture": hit.domain_architecture,
                "domain_class": hit.domain_class,
                "mean_top10_z": hit.mean_top10_z,
                "n_s1_candidates": hit.n_s1_candidates,
                "notes": ";".join(hit.notes),
                "train_seq": hit.train_seq,
                "construct_seq": hit.construct_seq,
            }
        )
    return pd.DataFrame(rows)


def alignment_summary(qc: pd.DataFrame) -> dict:
    name_nseq = qc.groupby("protein_name")["train_seq"].nunique()
    multi = name_nseq[name_nseq > 1]
    return {
        "n_variants": int(len(qc)),
        "n_protein_names": int(qc["protein_name"].nunique()),
        "n_names_with_multiple_sequences": int(len(multi)),
        "multi_sequence_names": sorted(multi.index.tolist()),
        "relation_counts": qc["relation"].value_counts().to_dict(),
        "safe_train_mask_n": int(qc["safe_train_mask"].sum()),
        "by_dataset_source": {
            str(src): sub["relation"].value_counts().to_dict()
            for src, sub in qc.groupby("dataset_source")
        },
        "domain_class_counts": qc["domain_class"].value_counts().to_dict(),
    }


# ── Domain-class labels for conditioning ─────────────────────────────────────

# Stable vocab order (unknown always index 0 for clarity in ablations).
DOMAIN_CLASS_VOCAB: tuple[str, ...] = (
    "unknown",
    "RRM",
    "KH",
    "multi",
    "CCCH",
    "CSD",
    "PUM",
    "ZF",
    "THUMP",
    "SAM",
    "S1",
    "other",
)


def domain_class_to_id(domain_class: str, vocab: tuple[str, ...] = DOMAIN_CLASS_VOCAB) -> int:
    c = str(domain_class).strip() if domain_class is not None else "unknown"
    if not c or c.lower() == "nan":
        c = "unknown"
    if c in vocab:
        return vocab.index(c)
    return vocab.index("other") if "other" in vocab else vocab.index("unknown")


def build_domain_label_maps(
    qc: pd.DataFrame,
    domains_tsv: Path | str | None = None,
    *,
    vocab: tuple[str, ...] = DOMAIN_CLASS_VOCAB,
) -> tuple[dict[str, str], dict[str, int], dict]:
    """
    Map variant_id → domain_class string and → integer id.

    Priority per variant:
      1. QC Table S1 / UniProt architecture class if not unknown
      2. Roster domains TSV (script 37) by protein_key (+ join aliases)
      3. unknown
    """
    key_to_class: dict[str, str] = {}
    if domains_tsv is not None:
        path = Path(domains_tsv)
        if path.exists():
            dom = pd.read_csv(path, sep="\t")
            for _, r in dom.iterrows():
                k = str(r["protein_key"])
                c = str(r["domain_class"]) if pd.notna(r.get("domain_class")) else "unknown"
                if c and c != "unknown":
                    key_to_class[k] = c
            for roster_key, s1_key in TABLE_S1_JOIN_ALIASES.items():
                if s1_key in key_to_class and roster_key not in key_to_class:
                    key_to_class[roster_key] = key_to_class[s1_key]
                if roster_key in key_to_class and s1_key not in key_to_class:
                    key_to_class[s1_key] = key_to_class[roster_key]

    class_by_vid: dict[str, str] = {}
    id_by_vid: dict[str, int] = {}
    source_tag: dict[str, str] = {}

    for _, r in qc.iterrows():
        vid = str(r["variant_id"]) if "variant_id" in r and pd.notna(r["variant_id"]) else variant_key(
            str(r["protein_name"]), str(r["train_seq"])
        )
        qc_class = str(r["domain_class"]) if pd.notna(r.get("domain_class")) else "unknown"
        key = str(r["protein_key"])
        if qc_class and qc_class != "unknown":
            c, tag = qc_class, "alignment_qc"
        elif key in key_to_class:
            c, tag = key_to_class[key], "domains_tsv"
        else:
            # try aliases
            found = None
            for k in s1_lookup_keys(key):
                if k in key_to_class:
                    found = key_to_class[k]
                    break
            if found:
                c, tag = found, "domains_tsv_alias"
            else:
                c, tag = "unknown", "unknown"
        class_by_vid[vid] = c
        id_by_vid[vid] = domain_class_to_id(c, vocab)
        source_tag[vid] = tag

    summary = {
        "n_variants": len(class_by_vid),
        "domain_class_counts": pd.Series(list(class_by_vid.values())).value_counts().to_dict(),
        "label_source_counts": pd.Series(list(source_tag.values())).value_counts().to_dict(),
        "n_unknown": int(sum(1 for c in class_by_vid.values() if c == "unknown")),
        "vocab": list(vocab),
    }
    return class_by_vid, id_by_vid, summary
