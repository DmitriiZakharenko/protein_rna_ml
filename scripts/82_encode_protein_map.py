#!/usr/bin/env python3
"""Script 82: map of ENCODE TF ChIP-seq and eCLIP proteins in ESM-2 space.

Stages
  proteins  table of TF (ChIP-seq) and RBP (eCLIP) proteins with class flags, v3b split and family.
  embed     ESM-2 650M mean embedding of every protein for all 34 layers (GPU VM, sliding window above 1,022 residues).
  kmer      5-mer enrichment profile (log2 positives against GC-matched negatives, 1,024 values) per protein and assay from
            the TF ChIP-seq window table and the Skipper eCLIP window table (run where these tables are, the GPU VM).
  stats     per layer: nearest-neighbour agreement with the class label, cohesion of the proteins that have both
            assays, silhouette, each against a label-permutation null.
  kmerstats per layer: Spearman correlation between the centred cosine of two proteins and the similarity of their 5-mer
            profiles (Pearson r and Jaccard of the 50 most enriched 5-mers), by pair group, with a permutation p-value.
  map       2D PNG per layer and one self-contained interactive 3D page: rotation, zoom, search, click on a protein
            lists its nearest and farthest proteins with centred cosine, 5-mer profile r and Jaccard.
  selftest  planted synthetic embeddings and profiles through all analysis stages and the page (no ESM, no data).

Class (from the chosen ChIP definition): tf = ChIP-seq only, rbp = eCLIP only, both = ChIP-seq and eCLIP.
ChIP definition: tf = ENCODE TF-annotated targets (list of the TF FASTA), broad = any target of the ENCODE TF ChIP-seq assay (GRCh38, released).
eCLIP = ENCODE eCLIP targets (GRCh38, released). Both lists are the tables results/encode_map/encode_{chip,eclip}_grch38.tsv (ENCODE API, 2026-11-26).
Centred cosine: embeddings are centred over proteins before the cosine, as in scripts/77 and 80.

Usage (project root)
  python scripts/82_encode_protein_map.py proteins
  python scripts/82_encode_protein_map.py embed          # GPU VM
  python scripts/82_encode_protein_map.py stats
  python scripts/82_encode_protein_map.py map --layers 0 18 22 26 33
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DEFAULT = "results/encode_map"
NL = 34
KM = 5
NK = 4 ** KM
NGC = 20
TOPK = 50


def resolve(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def read_fasta(path: Path) -> dict[str, tuple[str, str]]:
    out, name, acc, buf = {}, None, "", []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if name:
                out[name] = (acc, "".join(buf))
            head = line[1:].split("|")
            name, acc, buf = head[0].upper(), (head[1] if len(head) > 1 else ""), []
        else:
            buf.append(line.strip())
    if name:
        out[name] = (acc, "".join(buf))
    return out


def stage_proteins(args) -> None:
    out = resolve(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tf = read_fasta(resolve(args.tf_fasta))
    rbp = pd.read_csv(resolve(args.rbp_tsv), sep="\t")
    rbp["symbol"] = rbp.symbol.str.upper()
    rbp_seq = dict(zip(rbp.symbol, rbp.protein_sequence))
    rbp_acc = dict(zip(rbp.symbol, rbp.best_uniprot_accession))
    eclip = set(pd.read_csv(resolve(args.eclip_tsv), sep="\t").target.str.upper())
    chip_broad = set(pd.read_csv(resolve(args.chip_tsv), sep="\t").target.str.upper())
    split = pd.read_csv(resolve(args.split_map), sep="\t")
    split["key"] = split.protein_name.str.upper()
    sp = split.set_index("key")
    rows = []
    for s in sorted(set(tf) | set(rbp_seq)):
        seq = rbp_seq.get(s) or tf[s][1]
        acc = rbp_acc.get(s) or tf[s][0]
        differs = bool(s in tf and s in rbp_seq and tf[s][1] != rbp_seq[s])
        rows.append(dict(symbol=s, accession=acc, length=len(seq), sequence=seq, chip_tf=s in tf, chip_broad=s in chip_broad,
                         eclip=s in eclip, seq_differs=differs,
                         v3_split=sp.split[s] if s in sp.index else "", family=sp.family[s] if s in sp.index else ""))
    df = pd.DataFrame(rows)
    for d, col in (("tf", "chip_tf"), ("broad", "chip_broad")):
        df["class_" + d] = np.where(df[col] & df.eclip, "both", np.where(df[col], "tf", "rbp"))
    df.to_csv(out / "proteins.tsv", sep="\t", index=False)
    print(f"{len(df)} proteins: TF-annotated ChIP {int(df.chip_tf.sum())}, eCLIP {int(df.eclip.sum())}, "
          f"both (TF-annotated) {int((df.class_tf == 'both').sum())}, both (broad ChIP) {int((df.class_broad == 'both').sum())}, "
          f"sequence differs between sources {int(df.seq_differs.sum())}")


def load_embed_one():
    spec = importlib.util.spec_from_file_location("s80", ROOT / "scripts" / "80_esm_layer_profile_similarity.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.embed_one


def stage_embed(args) -> None:
    import torch
    import esm
    out = resolve(args.out_dir)
    df = pd.read_csv(out / "proteins.tsv", sep="\t")
    embed_one = load_embed_one()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.eval().to(device)
    conv = alphabet.get_batch_converter()
    layers = list(range(NL))
    emb = np.full((len(df), NL, 1280), np.nan, dtype=np.float32)
    skipped, t0 = [], time.time()
    for i, r in enumerate(df.itertuples()):
        if r.length > args.max_len:
            skipped.append(r.symbol)
            continue
        emb[i] = embed_one(model, conv, r.sequence, device, layers, None)[0]
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(df)}  {time.time() - t0:.0f}s", flush=True)
    np.savez(out / "esm2_layers.npz", emb=emb.astype(np.float16), symbols=df.symbol.to_numpy(dtype=object))
    print(f"saved {len(df) - len(skipped)} proteins x {NL} layers, skipped above {args.max_len} residues: {skipped}")


def centred_cosine(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(0, keepdims=True)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)
    return x @ x.T


def nn_agreement(c: np.ndarray, lab: np.ndarray) -> float:
    c = c.copy()
    np.fill_diagonal(c, -np.inf)
    nn = c.argmax(1)
    return float((lab[nn] == lab).mean())


def cohesion(c: np.ndarray, idx: np.ndarray) -> float:
    sub = c[np.ix_(idx, idx)]
    return float(sub[np.triu_indices(len(idx), k=1)].mean())


def layer_stats(emb: np.ndarray, lab: np.ndarray, known: np.ndarray, n_perm: int, rng, fam: np.ndarray | None) -> dict:
    from sklearn.metrics import silhouette_score
    c = centred_cosine(emb)
    both = np.where(lab == "both")[0]
    res = {}
    obs = nn_agreement(c, lab)
    null = [nn_agreement(c, rng.permutation(lab)) for _ in range(n_perm)]
    res.update(nn_class=obs, nn_class_null=float(np.mean(null)), nn_class_p=(1 + sum(x >= obs for x in null)) / (n_perm + 1))
    if len(set(lab)) > 1:
        d = np.clip(1 - c, 0, None)
        np.fill_diagonal(d, 0)
        res["silhouette_class"] = float(silhouette_score(d, lab, metric="precomputed"))
    if len(both) >= 3:
        o = cohesion(c, both)
        nl = [cohesion(c, rng.choice(len(lab), len(both), replace=False)) for _ in range(n_perm)]
        res.update(both_cohesion=o, both_cohesion_null=float(np.mean(nl)), both_cohesion_z=(o - np.mean(nl)) / max(np.std(nl), 1e-9),
                   both_cohesion_p=(1 + sum(x >= o for x in nl)) / (n_perm + 1))
    if fam is not None and known.sum() > 20:
        k = np.where(known)[0]
        ck = c[np.ix_(k, k)]
        f = fam[k]
        o = nn_agreement(ck, f)
        nl = [nn_agreement(ck, rng.permutation(f)) for _ in range(n_perm)]
        res.update(nn_family=o, nn_family_null=float(np.mean(nl)), nn_family_p=(1 + sum(x >= o for x in nl)) / (n_perm + 1))
    return res


def load_for_stats(out: Path, chip_def: str):
    df = pd.read_csv(out / "proteins.tsv", sep="\t").fillna({"family": "", "v3_split": ""})
    z = np.load(out / "esm2_layers.npz", allow_pickle=True)
    emb = z["emb"].astype(np.float32)
    assert list(z["symbols"]) == list(df.symbol), "proteins.tsv and esm2_layers.npz differ, rerun the embed stage"
    ok = ~np.isnan(emb[:, 0, 0])
    return df[ok].reset_index(drop=True), emb[ok]


def run_stats(df: pd.DataFrame, emb: np.ndarray, chip_def: str, n_perm: int, seed: int, layers: list[int]) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    lab = df["class_" + chip_def].to_numpy()
    fam = df.family.to_numpy()
    known = (fam != "") & (fam != "other")
    rows = []
    for L in layers:
        r = layer_stats(emb[:, L], lab, known, n_perm, rng, fam)
        r["layer"] = L
        rows.append(r)
        print(f"  layer {L:2d}: nn_class {r['nn_class']:.3f} (null {r['nn_class_null']:.3f}, p {r['nn_class_p']:.3f})"
              + (f", both_cohesion z {r['both_cohesion_z']:.2f}" if "both_cohesion_z" in r else ""), flush=True)
    return pd.DataFrame(rows).set_index("layer").reset_index()


def stage_stats(args) -> None:
    out = resolve(args.out_dir)
    df, emb = load_for_stats(out, args.chip_def)
    print(f"{len(df)} proteins, class counts {df['class_' + args.chip_def].value_counts().to_dict()}")
    res = run_stats(df, emb, args.chip_def, args.n_perm, args.seed, list(range(NL)))
    res.to_csv(out / f"layer_stats_{args.chip_def}.tsv", sep="\t", index=False)
    plot_stats(res, out / f"layer_stats_{args.chip_def}.png")
    print("saved", out / f"layer_stats_{args.chip_def}.tsv")


def plot_stats(res: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(res.layer, res.nn_class, label="observed")
    ax[0].plot(res.layer, res.nn_class_null, "--", label="label permutation")
    ax[0].set_xlabel("ESM-2 layer")
    ax[0].set_ylabel("nearest neighbour has the same class")
    ax[0].legend()
    if "both_cohesion_z" in res:
        ax[1].plot(res.layer, res.both_cohesion_z)
        ax[1].axhline(0, color="grey", lw=0.8)
        ax[1].set_ylabel("cohesion of proteins with ChIP-seq and eCLIP (z)")
        ax[1].set_xlabel("ESM-2 layer")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


LUT = np.full(256, -1, dtype=np.int16)
for _i, _ch in enumerate("ACGU"):
    LUT[ord(_ch)] = _i
LUT[ord("T")] = 3


def kmer_counts(seq: str):
    a = LUT[np.frombuffer(seq.upper().encode(), dtype=np.uint8)].astype(np.int32)
    n = len(a)
    if n < KM:
        return None, 0.0
    m = n - KM + 1
    idx = np.zeros(m, dtype=np.int32)
    bad = np.zeros(m, dtype=bool)
    for j in range(KM):
        seg = a[j:j + m]
        idx = idx * 4 + seg
        bad |= seg < 0
    gc = float(((a == 1) | (a == 2)).sum()) / n
    return np.bincount(idx[~bad], minlength=NK).astype(np.float32), gc


def profile_from_counts(cnt, nseq, min_pos: int):
    pos_b, neg_b = nseq[1], nseq[0]
    keep = (pos_b > 0) & (neg_b > 0)
    npos = float(pos_b.sum())
    if npos < min_pos or pos_b[keep].sum() < 0.9 * npos:
        return None
    pos = cnt[1][keep].sum(0)
    w = pos_b[keep] / neg_b[keep]
    neg = (cnt[0][keep] * w[:, None]).sum(0)
    p = (pos + 0.5) / (pos.sum() + 0.5 * NK)
    q = (neg + 0.5) / (neg.sum() + 0.5 * NK)
    return np.log2(p / q).astype(np.float32), int(npos), float(pos_b[keep].sum() / npos)


def accumulate(acc: dict, df: pd.DataFrame, assay: str) -> None:
    for name, seq, lab in zip(df.protein_name.to_numpy(), df.rna_sequence.to_numpy(), df.binding_label.to_numpy()):
        c, gc = kmer_counts(seq)
        if c is None:
            continue
        key = (str(name).upper(), assay)
        if key not in acc:
            acc[key] = (np.zeros((2, NGC, NK), np.float32), np.zeros((2, NGC), np.float64))
        b = min(int(gc * NGC), NGC - 1)
        y = 1 if lab else 0
        acc[key][0][y, b] += c
        acc[key][1][y, b] += 1


def stage_kmer(args) -> None:
    out = resolve(args.out_dir)
    acc: dict = {}
    t0 = time.time()
    for chunk in pd.read_csv(resolve(args.tf_pairs), sep="\t", usecols=["protein_name", "rna_sequence", "binding_label", "neg_type"], chunksize=200000):
        chunk = chunk[(chunk.binding_label == 1) | (chunk.neg_type == "gc")]
        accumulate(acc, chunk, "chip")
        print(f"  chip rows so far, {len(acc)} keys, {time.time() - t0:.0f}s", flush=True)
    for chunk in pd.read_csv(resolve(args.eclip_pairs), sep="\t", usecols=["protein_name", "rna_sequence", "binding_label"], chunksize=100000):
        accumulate(acc, chunk, "eclip")
        print(f"  eclip rows so far, {len(acc)} keys, {time.time() - t0:.0f}s", flush=True)
    keys, prof, npos, cover = [], [], [], []
    for (name, assay), (cnt, ns) in sorted(acc.items()):
        r = profile_from_counts(cnt, ns, args.min_pos)
        if r is None:
            continue
        keys.append(f"{name}|{assay}")
        prof.append(r[0])
        npos.append(r[1])
        cover.append(r[2])
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "kmer_profiles.npz", keys=np.array(keys, dtype=object), profiles=np.stack(prof), npos=np.array(npos), coverage=np.array(cover))
    n_chip = sum(k.endswith("|chip") for k in keys)
    print(f"saved {len(keys)} profiles: chip {n_chip}, eclip {len(keys) - n_chip}, min positives {args.min_pos}")


def kmer_similarity(prof: np.ndarray):
    z = prof - prof.mean(1, keepdims=True)
    z = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-9)
    r = z @ z.T
    b = np.zeros(prof.shape, dtype=np.float32)
    np.put_along_axis(b, np.argsort(-prof, axis=1)[:, :TOPK], 1.0, axis=1)
    inter = b @ b.T
    return r, inter / (2 * TOPK - inter)


def kmer_str(i: int) -> str:
    return "".join("ACGU"[(int(i) >> (2 * (KM - 1 - p))) & 3] for p in range(KM))


def load_kmer(out: Path):
    f = out / "kmer_profiles.npz"
    if not f.exists():
        return None
    z = np.load(f, allow_pickle=True)
    return [str(k) for k in z["keys"]], z["profiles"].astype(np.float32)


def rank_rho(a: np.ndarray, rb: np.ndarray) -> float:
    from scipy.stats import rankdata
    return float(np.corrcoef(rankdata(a), rb)[0, 1])


def mantel(A: np.ndarray, B: np.ndarray, mask: np.ndarray, square: bool, n_perm: int, rng):
    from scipy.stats import rankdata
    rb = rankdata(B[mask])
    rho = rank_rho(A[mask], rb)
    ge = 0
    for _ in range(n_perm):
        p = rng.permutation(A.shape[0])
        Ap = A[np.ix_(p, p)] if square else A[p, :]
        ge += abs(rank_rho(Ap[mask], rb)) >= abs(rho)
    return rho, (ge + 1) / (n_perm + 1)


def kmer_stats(df: pd.DataFrame, emb: np.ndarray, kmer, layers: list[int], n_perm: int, seed: int) -> pd.DataFrame:
    keys, prof = kmer
    R, J = kmer_similarity(prof)
    sym = {s: i for i, s in enumerate(df.symbol)}
    item = [(k.split("|")[0], k.split("|")[1], t) for t, k in enumerate(keys) if k.split("|")[0] in sym]
    ts = np.array([x[2] for x in item])
    pi = np.array([sym[x[0]] for x in item])
    assay = np.array([x[1] for x in item])
    rng = np.random.default_rng(seed)
    rows = []
    for L in layers:
        C = centred_cosine(emb[:, L])[np.ix_(pi, pi)]
        for g, (a1, a2) in {"chip-chip": ("chip", "chip"), "eclip-eclip": ("eclip", "eclip"), "chip-eclip": ("chip", "eclip")}.items():
            i1, i2 = np.where(assay == a1)[0], np.where(assay == a2)[0]
            if len(i1) < 8 or len(i2) < 8:
                continue
            square = a1 == a2
            A = C[np.ix_(i1, i2)]
            same = pi[i1][:, None] == pi[i2][None, :]
            mask = (np.triu(np.ones(A.shape, bool), 1) if square else np.ones(A.shape, bool)) & ~same
            for tname, S in (("pearson_5mer", R), ("jaccard_top50", J)):
                B = S[np.ix_(ts[i1], ts[i2])]
                rho, p = mantel(A, B, mask, square, n_perm, rng)
                rows.append(dict(layer=L, group=g, target=tname, rho=rho, p=p, n_pairs=int(mask.sum())))
        print(f"  layer {L:2d} done", flush=True)
    return pd.DataFrame(rows)


def dual_test(keys, kmer_r, n_perm: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    chip = {k.split("|")[0]: t for t, k in enumerate(keys) if k.endswith("|chip")}
    ecl = {k.split("|")[0]: t for t, k in enumerate(keys) if k.endswith("|eclip")}
    both = sorted(set(chip) & set(ecl))
    if len(both) < 3:
        return {}
    obs = float(np.mean([kmer_r[chip[s], ecl[s]] for s in both]))
    cs, es = list(chip), list(ecl)
    null = []
    for _ in range(n_perm):
        a = rng.choice(len(cs), len(both), replace=False)
        b = rng.choice(len(es), len(both), replace=False)
        null.append(np.mean([kmer_r[chip[cs[x]], ecl[es[y]]] for x, y in zip(a, b) if cs[x] != es[y]]))
    return dict(n_both=len(both), mean_r_same_protein=obs, null_mean=float(np.mean(null)), z=(obs - np.mean(null)) / max(np.std(null), 1e-9),
                p=(1 + sum(x >= obs for x in null)) / (n_perm + 1))


def stage_kmerstats(args) -> None:
    out = resolve(args.out_dir)
    df, emb = load_for_stats(out, args.chip_def)
    kmer = load_kmer(out)
    if kmer is None:
        raise SystemExit("kmer_profiles.npz not found, run the kmer stage first")
    res = kmer_stats(df, emb, kmer, list(range(NL)), args.n_perm, args.seed)
    res.to_csv(out / "kmer_layer_stats.tsv", sep="\t", index=False)
    R, _ = kmer_similarity(kmer[1])
    d = dual_test(kmer[0], R, 1000, args.seed)
    (out / "kmer_dual_test.json").write_text(json.dumps(d, indent=1))
    print("dual test (same protein, ChIP profile against eCLIP profile):", d)
    plot_kmer_stats(res, out / "kmer_layer_stats.png")
    print("saved", out / "kmer_layer_stats.tsv")


def plot_kmer_stats(res: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tg = sorted(res.target.unique())
    fig, ax = plt.subplots(1, len(tg), figsize=(5 * len(tg), 3.6), squeeze=False)
    for k, tname in enumerate(tg):
        for g, d in res[res.target == tname].groupby("group"):
            ax[0][k].plot(d.layer, d.rho, label=g)
        ax[0][k].axhline(0, color="grey", lw=0.8)
        ax[0][k].set_title(tname)
        ax[0][k].set_xlabel("ESM-2 layer")
        ax[0][k].set_ylabel("Spearman: centred cosine vs profile similarity")
        ax[0][k].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def coords(emb_l: np.ndarray, seed: int, dim: int = 3) -> np.ndarray:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    x = emb_l - emb_l.mean(0, keepdims=True)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)
    x = PCA(n_components=min(50, x.shape[0] - 1, x.shape[1]), random_state=seed).fit_transform(x)
    perp = max(5, min(30, (len(x) - 1) // 3))
    return TSNE(n_components=dim, perplexity=perp, init="pca", random_state=seed).fit_transform(x)


def build_neighbors(df: pd.DataFrame, emb: np.ndarray, layers: list[int], kmer, k_near: int = 20, k_far: int = 8):
    sym = list(df.symbol)
    pmap, R, J, prof = {}, None, None, None
    if kmer is not None:
        keys, prof = kmer
        R, J = kmer_similarity(prof)
        for t, k in enumerate(keys):
            s, a = k.split("|")
            pmap.setdefault(s, {})[a] = t

    def pk(a: int, b: int):
        pa, pb = pmap.get(sym[a]), pmap.get(sym[b])
        if not pa or not pb:
            return None, None, None
        for assay in ("chip", "eclip"):
            if assay in pa and assay in pb:
                return round(float(R[pa[assay], pb[assay]]), 3), round(float(J[pa[assay], pb[assay]]), 3), f"{assay}-{assay}"
        x, y = next(iter(pa)), next(iter(pb))
        return round(float(R[pa[x], pb[y]]), 3), round(float(J[pa[x], pb[y]]), 3), f"{x}-{y}"

    nbr = {}
    for L in layers:
        C = centred_cosine(emb[:, L])
        near, far = [], []
        for i in range(len(sym)):
            o = np.argsort(-C[i])
            nn = [int(j) for j in o if j != i][:k_near]
            ff = [int(j) for j in o[::-1] if j != i][:k_far]
            near.append([[j, round(float(C[i, j]), 3), *pk(i, j)] for j in nn])
            far.append([[j, round(float(C[i, j]), 3), *pk(i, j)] for j in ff])
        nbr[str(L)] = dict(near=near, far=far)
    top = {}
    if prof is not None:
        for s, d in pmap.items():
            top[s] = {a: [kmer_str(x) for x in np.argsort(-prof[t])[:8]] for a, t in d.items()}
    return nbr, top


HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>ENCODE protein map</title><style>
:root{--bg:#f7f7f4;--card:#fff;--ink:#1c2230;--mute:#5d6678;--line:#dcdfe6}
@media (prefers-color-scheme:dark){:root{--bg:#12151c;--card:#1a1f2a;--ink:#e8ebf2;--mute:#9aa3b5;--line:#2c3342}}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,sans-serif}
header{padding:10px 16px;display:flex;flex-wrap:wrap;gap:12px;align-items:center;border-bottom:1px solid var(--line)}
h1{font-size:17px;margin:0}select,input,button{font:inherit;padding:3px 8px;background:var(--card);color:var(--ink);border:1px solid var(--line);border-radius:6px}
.banner{background:#b4531c;color:#fff;padding:6px 16px;font-weight:600}
main{display:grid;grid-template-columns:1fr 430px}
canvas{width:100%;height:calc(100vh - 92px);background:var(--card);cursor:grab;display:block}
aside{padding:10px 14px;border-left:1px solid var(--line);font-size:13.5px;overflow:auto;height:calc(100vh - 92px)}
.k{display:flex;align-items:center;gap:8px;margin:5px 0}.sh{width:13px;height:13px;display:inline-block}
.c{border-radius:50%}.d{transform:rotate(45deg) scale(.85)}
table{border-collapse:collapse;width:100%;font-size:12.5px}th,td{padding:3px 5px;text-align:left;border-bottom:1px solid var(--line)}th{color:var(--mute);font-weight:600}
tr.r{cursor:pointer}tr.r:hover{background:var(--bg)}
#tip{position:fixed;pointer-events:none;background:var(--card);border:1px solid var(--line);padding:4px 8px;border-radius:6px;display:none;font-size:13px}
.m{color:var(--mute);font-size:12.5px}h3{margin:10px 0 4px;font-size:14px}code{background:var(--bg);padding:0 4px;border-radius:4px}
@media (max-width:900px){main{grid-template-columns:1fr}aside{height:auto;border-left:0}}
</style></head><body>
__BANNER__
<header><h1>ENCODE proteins in ESM-2 space</h1>
<label>Layer <select id="layer"></select></label>
<label>Colour by <select id="by"><option value="cls">class (TF / RBP / both)</option><option value="split">v3b split</option><option value="family">family</option></select></label>
<label><input type="checkbox" id="auto" checked> auto-rotate</label><button id="reset">reset view</button><button id="clear">clear selection</button>
<input id="q" placeholder="symbol, e.g. SFPQ"><span id="info" class="m"></span></header>
<main><canvas id="cv"></canvas><aside id="side"></aside></main><div id="tip"></div>
<script>
var D=__DATA__,P=D.proteins,IDX={};P.forEach(function(p,i){IDX[p.symbol]=i});
var CLS={tf:{c:"#2458a6",n:"TF only (ChIP-seq)",s:"c",g:"●"},rbp:{c:"#b4531c",n:"RBP only (eCLIP)",s:"q",g:"■"},both:{c:"#2d7a4f",n:"Both assays",s:"d",g:"◆"}};
var pal=["#2458a6","#b4531c","#2d7a4f","#8a4fa3","#c9a227","#2a9d9f","#a33a54","#6b7a8f","#7d5a3c","#4c8f3a"];
var cv=document.getElementById("cv"),ctx=cv.getContext("2d"),tip=document.getElementById("tip"),ls=document.getElementById("layer"),side=document.getElementById("side");
D.layers.forEach(function(L){var o=document.createElement("option");o.value=L;o.textContent=L;ls.appendChild(o)});
var yaw=0.6,pitch=0.35,zoom=1,drag=false,lx=0,ly=0,proj=[],dpr=window.devicePixelRatio||1,moved=false,sel=-1,N=null,curL=null,pmap={};
function val(p,by){return by=="cls"?p.cls:(by=="split"?(p.split||"none"):(p.family||"unknown"))}
function norm(L){var C=D.coords[L],m=[0,0,0];C.forEach(function(c){for(var k=0;k<3;k++)m[k]+=c[k]/C.length});var r=0;C.forEach(function(c){var d=Math.hypot(c[0]-m[0],c[1]-m[1],c[2]-m[2]);if(d>r)r=d});return C.map(function(c){return[(c[0]-m[0])/r,(c[1]-m[1])/r,(c[2]-m[2])/r]})}
function shape(x,y,r,s){ctx.beginPath();if(s=="c")ctx.arc(x,y,r,0,6.2832);else if(s=="q")ctx.rect(x-r,y-r,2*r,2*r);else{ctx.moveTo(x,y-r*1.35);ctx.lineTo(x+r*1.35,y);ctx.lineTo(x,y+r*1.35);ctx.lineTo(x-r*1.35,y);ctx.closePath()}}
function nb(){return D.nbr[ls.value]}
function draw(){var L=ls.value;if(L!==curL){N=norm(L);curL=L;panel()}
 var by=document.getElementById("by").value,q=document.getElementById("q").value.trim().toUpperCase();
 var W=cv.clientWidth,H=cv.clientHeight;if(cv.width!==W*dpr||cv.height!==H*dpr){cv.width=W*dpr;cv.height=H*dpr}
 ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,W,H);
 var vals=[];P.forEach(function(p){var v=val(p,by);if(vals.indexOf(v)<0)vals.push(v)});vals.sort();
 var cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch),S=Math.min(W,H)*0.36*zoom;proj=[];var byI={};
 N.forEach(function(c,i){var x1=c[0]*cy+c[2]*sy,z1=-c[0]*sy+c[2]*cy,y2=c[1]*cp-z1*sp,z2=c[1]*sp+z1*cp,f=3/(3-z2),o={i:i,x:W/2+x1*f*S,y:H/2+y2*f*S,z:z2,f:f};proj.push(o);byI[i]=o});
 var nset={};if(sel>=0){nb().near[sel].slice(0,10).forEach(function(e){nset[e[0]]=e[1]});ctx.lineWidth=1.2;nb().near[sel].slice(0,10).forEach(function(e){var a=byI[sel],b=byI[e[0]];ctx.globalAlpha=0.35+0.65*Math.max(0,e[1]);ctx.strokeStyle="#c9a227";ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke()})}
 proj.sort(function(a,b){return a.z-b.z});
 proj.forEach(function(o){var p=P[o.i],hit=q&&p.symbol.indexOf(q)>=0,isSel=o.i==sel,isN=nset.hasOwnProperty(o.i),dim=(q&&!hit)||(sel>=0&&!isSel&&!isN),base=p.cls=="both"?6.5:(p.cls=="rbp"?4.2:3.6);
  var r=((hit||isSel)?10:(isN?7:base))*o.f*Math.min(1.4,zoom+0.3);ctx.globalAlpha=dim?0.12:Math.max(0.35,Math.min(1,0.55+o.z*0.45));
  ctx.fillStyle=by=="cls"?CLS[p.cls].c:pal[vals.indexOf(val(p,by))%pal.length];shape(o.x,o.y,r,CLS[p.cls].s);ctx.fill();
  if(p.cls=="both"||hit||isSel||isN){ctx.globalAlpha=dim?0.12:1;ctx.lineWidth=isSel?2.2:1.3;ctx.strokeStyle=isSel?"#c9a227":"#000";ctx.stroke()}
  if(hit||isSel||isN){ctx.globalAlpha=1;ctx.fillStyle=getComputedStyle(document.body).color;ctx.font=(isSel?"bold ":"")+"12px system-ui";ctx.fillText(p.symbol,o.x+10,o.y-7)}});
 ctx.globalAlpha=1;document.getElementById("info").textContent=P.length+" proteins, layer "+L}
function f3(x){return x==null?"-":x.toFixed(2)}
function row(e){var p=P[e[0]];return '<tr class="r" data-i="'+e[0]+'"><td>'+CLS[p.cls].g+" "+p.symbol+"</td><td>"+f3(e[1])+"</td><td>"+f3(e[2])+"</td><td>"+f3(e[3])+"</td><td class='m'>"+(e[4]||"")+"</td></tr>"}
function legend(){var by=document.getElementById("by").value,cnt={tf:0,rbp:0,both:0};P.forEach(function(p){cnt[p.cls]++});
 var h="<b>Class (shape)</b>"+Object.keys(CLS).map(function(k){var d=CLS[k];return '<div class="k"><span class="sh '+d.s+'" style="background:'+d.c+';'+(k=="both"?"outline:1px solid #000":"")+'"></span>'+d.n+" ("+cnt[k]+")</div>"}).join("");
 if(by!="cls"){var vals=[];P.forEach(function(p){var v=val(p,by);if(vals.indexOf(v)<0)vals.push(v)});vals.sort();h+="<b>Colour: "+by+"</b>"+vals.map(function(v){return '<div class="k"><span class="sh c" style="background:'+pal[vals.indexOf(v)%pal.length]+'"></span>'+v+" ("+P.filter(function(p){return val(p,by)==v}).length+")</div>"}).join("")}return h}
function panel(){var h=legend();
 if(sel<0){h+='<p class="m">Click a protein to list its nearest and farthest proteins. <b>cos</b>: centred cosine of ESM-2 mean embeddings at the chosen layer. <b>r5</b>: Pearson correlation of the 5-mer enrichment profiles (log2 positive windows over GC-matched negatives). <b>J</b>: Jaccard index of the 50 most enriched 5-mers. r5 and J exist only where both proteins have a ChIP-seq or eCLIP window table.</p>'}
 else{var p=P[sel],b=nb();h+="<h3>"+p.symbol+" <span class='m'>"+CLS[p.cls].n+"</span></h3><div class='m'>family: "+(p.family||"unknown")+", v3b split: "+(p.split||"none")+"</div>";
  var tp=D.top[p.symbol];if(tp)h+="<div class='m'>top 5-mers: "+Object.keys(tp).map(function(a){return a+" <code>"+tp[a].slice(0,5).join(" ")+"</code>"}).join(" ; ")+"</div>";else h+="<div class='m'>no 5-mer profile for this protein</div>";
  h+="<h3>Nearest (layer "+ls.value+")</h3><table><tr><th>protein</th><th>cos</th><th>r5</th><th>J</th><th>data</th></tr>"+b.near[sel].map(row).join("")+"</table>";
  h+="<h3>Farthest</h3><table><tr><th>protein</th><th>cos</th><th>r5</th><th>J</th><th>data</th></tr>"+b.far[sel].map(row).join("")+"</table>"}
 side.innerHTML=h;[].forEach.call(side.querySelectorAll("tr.r"),function(tr){tr.addEventListener("click",function(){sel=+tr.getAttribute("data-i");panel()})})}
function pick(mx,my){var best=null,bd=14;for(var k=proj.length-1;k>=0;k--){var d=Math.hypot(proj[k].x-mx,proj[k].y-my);if(d<bd){bd=d;best=proj[k]}}return best}
function loop(){if(document.getElementById("auto").checked&&!drag)yaw+=0.004;draw();requestAnimationFrame(loop)}
cv.addEventListener("mousedown",function(e){drag=true;moved=false;lx=e.clientX;ly=e.clientY;cv.style.cursor="grabbing"});
window.addEventListener("mouseup",function(e){if(drag&&!moved&&e.target===cv){var r=cv.getBoundingClientRect(),b=pick(e.clientX-r.left,e.clientY-r.top);if(b){sel=b.i;panel()}}drag=false;cv.style.cursor="grab"});
window.addEventListener("mousemove",function(e){if(drag){var dx=e.clientX-lx,dy=e.clientY-ly;if(Math.abs(dx)+Math.abs(dy)>2)moved=true;yaw+=dx*0.008;pitch=Math.max(-1.5,Math.min(1.5,pitch+dy*0.008));lx=e.clientX;ly=e.clientY;tip.style.display="none";return}
 if(e.target!==cv){tip.style.display="none";return}var r=cv.getBoundingClientRect(),best=pick(e.clientX-r.left,e.clientY-r.top);
 if(best){var p=P[best.i];tip.style.display="block";tip.style.left=e.clientX+12+"px";tip.style.top=e.clientY+12+"px";tip.textContent=p.symbol+" | "+CLS[p.cls].n+" | "+(p.family||"family unknown")+" | click for neighbours"}else tip.style.display="none"});
cv.addEventListener("wheel",function(e){e.preventDefault();zoom=Math.max(0.4,Math.min(4,zoom*(e.deltaY<0?1.1:0.9)))},{passive:false});
document.getElementById("reset").addEventListener("click",function(){yaw=0.6;pitch=0.35;zoom=1});
document.getElementById("clear").addEventListener("click",function(){sel=-1;panel()});
document.getElementById("by").addEventListener("change",panel);
document.getElementById("q").addEventListener("keydown",function(e){if(e.key=="Enter"){var s=this.value.trim().toUpperCase();if(IDX.hasOwnProperty(s)){sel=IDX[s];panel()}}});
ls.addEventListener("change",function(){curL=null});
ls.value=D.layers[D.layers.length-1];loop();
</script></body></html>"""


def build_html(df: pd.DataFrame, emb: np.ndarray, layers: list[int], chip_def: str, seed: int, path: Path, demo: bool, kmer=None) -> None:
    cl = {str(L): np.round(coords(emb[:, L], seed), 3).tolist() for L in layers}
    nbr, top = build_neighbors(df, emb, layers, kmer)
    prot = [dict(symbol=r.symbol, cls=getattr(r, "class_" + chip_def), split=r.v3_split or "", family=r.family or "") for r in df.itertuples()]
    banner = '<div class="banner">SYNTHETIC DEMO DATA: planted structure, not ESM-2 embeddings or 5-mer profiles of ENCODE proteins</div>' if demo else ""
    html = HTML.replace("__BANNER__", banner).replace("__DATA__", json.dumps(dict(layers=layers, coords=cl, proteins=prot, nbr=nbr, top=top), separators=(",", ":")))
    path.write_text(html)


def stage_map(args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = resolve(args.out_dir)
    df, emb = load_for_stats(out, args.chip_def)
    lab = df["class_" + args.chip_def].to_numpy()
    colors = {"tf": "#2458a6", "rbp": "#b4531c", "both": "#2d7a4f"}
    for L in args.layers:
        xy = coords(emb[:, L], args.seed, dim=2)
        fig, ax = plt.subplots(figsize=(6, 5.4))
        for k in ("tf", "rbp", "both"):
            m = lab == k
            ax.scatter(xy[m, 0], xy[m, 1], s=40 if k == "both" else 8, c=colors[k], label=f"{k} ({m.sum()})",
                       edgecolors="k" if k == "both" else "none", linewidths=0.6, alpha=0.85)
        ax.set_title(f"ESM-2 layer {L}, centred cosine geometry")
        ax.legend(markerscale=1.4, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(out / f"map_layer{L}.png", dpi=160)
        plt.close(fig)
    kmer = load_kmer(out)
    if kmer is None:
        print("kmer_profiles.npz not found: the page has cosine neighbours only (run the kmer stage for r5 and Jaccard)")
    build_html(df, emb, args.layers, args.chip_def, args.seed, out / "encode_protein_map.html", demo=False, kmer=kmer)
    print("saved maps and", out / "encode_protein_map.html")


def stage_selftest(args) -> None:
    rng = np.random.default_rng(0)
    n = 260
    cls = np.array(["tf"] * 120 + ["rbp"] * 120 + ["both"] * 20)
    grp = {"tf": rng.normal(size=1280), "rbp": rng.normal(size=1280), "both": rng.normal(size=1280)}
    emb = np.zeros((n, NL, 1280), dtype=np.float32)
    for L in range(NL):
        sig = L / (NL - 1) * 1.2
        for i, c in enumerate(cls):
            emb[i, L] = rng.normal(size=1280) + sig * grp[c]
    fam = np.array(rng.choice(["RRM", "KH", "C2H2", ""], size=n))
    df = pd.DataFrame(dict(symbol=[f"P{i}" for i in range(n)], class_tf=cls, class_broad=cls, v3_split="", family=fam))
    res = run_stats(df, emb, "tf", 100, 0, [0, 17, 33])
    assert res.nn_class.iloc[-1] > res.nn_class.iloc[0] + 0.2 and res.nn_class_p.iloc[-1] < 0.05, "planted class structure not recovered"
    assert res.both_cohesion_z.iloc[-1] > 3, "planted cohesion not recovered"
    def rand_seq(k, motif=None):
        s = list(rng.choice(list("ACGU"), size=k))
        if motif:
            pos = int(rng.integers(0, k - len(motif)))
            s[pos:pos + len(motif)] = list(motif)
        return "".join(s)

    rows = [dict(protein_name="PX", rna_sequence=rand_seq(151, "GCAUG"), binding_label=1) for _ in range(400)]
    rows += [dict(protein_name="PX", rna_sequence=rand_seq(151), binding_label=0) for _ in range(1200)]
    acc: dict = {}
    accumulate(acc, pd.DataFrame(rows), "eclip")
    cnt, ns = acc[("PX", "eclip")]
    pr = profile_from_counts(cnt, ns, 100)
    assert pr is not None and kmer_str(int(np.argmax(pr[0]))) == "GCAUG", "planted 5-mer motif not recovered"
    assert profile_from_counts(cnt, ns, 5000) is None, "minimum positives not enforced"
    base = {c: rng.normal(size=NK) for c in grp}
    keys, prof = [], []
    for i in list(range(0, 90)) + list(range(240, 260)):
        keys.append(f"P{i}|chip")
        prof.append(base[cls[i]] + rng.normal(size=NK) * 1.5)
    for i in list(range(120, 200)) + list(range(240, 260)):
        keys.append(f"P{i}|eclip")
        prof.append(base[cls[i]] + rng.normal(size=NK) * 1.5)
    kmer = (keys, np.stack(prof).astype(np.float32))
    ks = kmer_stats(df, emb, kmer, [0, 33], 100, 0)
    r0 = ks[(ks.layer == 0) & (ks.group == "chip-chip") & (ks.target == "pearson_5mer")].rho.iloc[0]
    r33 = ks[(ks.layer == 33) & (ks.group == "chip-chip") & (ks.target == "pearson_5mer")].rho.iloc[0]
    assert r33 > r0 + 0.05 and ks[(ks.layer == 33)].p.min() < 0.05, "planted profile similarity not recovered"
    d = dual_test(keys, kmer_similarity(kmer[1])[0], 200, 0)
    assert d["n_both"] == 20 and d["z"] > 3, "planted dual-protein similarity not recovered"
    with tempfile.TemporaryDirectory() as td:
        build_html(df, emb, [0, 33], "tf", 0, Path(td) / "m.html", demo=True, kmer=kmer)
        assert (Path(td) / "m.html").stat().st_size > 20000
        plot_stats(res, Path(td) / "s.png")
        plot_kmer_stats(ks, Path(td) / "k.png")
        assert (Path(td) / "s.png").exists() and (Path(td) / "k.png").exists()
    if args.demo_html:
        build_html(df, emb, [0, 18, 33], "tf", 0, resolve(args.demo_html), demo=True, kmer=kmer)
        print("demo page written to", resolve(args.demo_html))
    print("selftest passed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["proteins", "embed", "kmer", "stats", "kmerstats", "map", "selftest"])
    ap.add_argument("--out_dir", default=OUT_DEFAULT)
    ap.add_argument("--tf_fasta", default="share_tf/final/encode_tf_chipseq_proteins.fasta")
    ap.add_argument("--rbp_tsv", default="data/raw/eclip_skipper/manifests/encode_eclip_rbp_id_best_acc_seq.tsv")
    ap.add_argument("--eclip_tsv", default="results/encode_map/encode_eclip_grch38.tsv")
    ap.add_argument("--chip_tsv", default="results/encode_map/encode_chip_grch38.tsv")
    ap.add_argument("--split_map", default="results/v3b/protein_split_map.tsv")
    ap.add_argument("--tf_pairs", default="data/benchmarks/tf_chipseq/pairs.tsv")
    ap.add_argument("--eclip_pairs", default="data/benchmarks/skipper_eclip/fixlen_151_all.tsv")
    ap.add_argument("--min_pos", type=int, default=100)
    ap.add_argument("--chip_def", choices=["tf", "broad"], default="broad")
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 18, 22, 26, 33])
    ap.add_argument("--max_len", type=int, default=6000)
    ap.add_argument("--n_perm", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--demo_html", default="")
    args = ap.parse_args()
    {"proteins": stage_proteins, "embed": stage_embed, "kmer": stage_kmer, "stats": stage_stats, "kmerstats": stage_kmerstats,
     "map": stage_map, "selftest": stage_selftest}[args.stage](args)


if __name__ == "__main__":
    main()
