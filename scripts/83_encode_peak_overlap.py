#!/usr/bin/env python3
"""Script 83: genomic overlap of ENCODE TF ChIP-seq peaks and eCLIP peaks for proteins that have both assays.

Stages
  files     list the ENCODE files of the experiments of the chosen targets (ENCODE API), write results/encode_peaks/files.tsv.
  download  download the chosen bed files into data/external/encode_chip_eclip/ (not tracked in git, about 100 MB).
  overlap   Jaccard index of merged intervals for every ChIP-seq set against every eCLIP set of the same cell line, rank of the
            own-protein pair among all pairs, and Spearman correlation of the Jaccard matrix with the centred cosine of the
            ESM-2 embeddings (permutation p-value).
  selftest  synthetic intervals through the overlap statistics.

Files: ChIP-seq 'IDR thresholded peaks' (bed narrowPeak, GRCh38), eCLIP 'peaks' (bed narrowPeak, GRCh38, CLIPper). When an
experiment has several files, the file marked preferred_default is taken, else the largest. Peaks are merged without strand.
Null for the own-protein pair: the Jaccard of the same ChIP-seq set with the eCLIP sets of the other proteins in the same cell line.
This keeps the genomic distribution of eCLIP peaks (expressed genes) and removes the shared-gene bias of a random shift.

Usage (project root)
  python scripts/83_encode_peak_overlap.py files
  python scripts/83_encode_peak_overlap.py download
  python scripts/83_encode_peak_overlap.py overlap
"""
from __future__ import annotations

import argparse
import gzip
import json
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://www.encodeproject.org"
CHROMS = [f"chr{i}" for i in list(range(1, 23)) + ["X"]]


def resolve(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def targets_with_both(chip: pd.DataFrame, eclip: pd.DataFrame) -> list[str]:
    return sorted(set(chip.target.str.upper()) & set(eclip.target.str.upper()))


def get_json(url: str) -> dict:
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "research-client/1.0"})
            return json.load(urllib.request.urlopen(req, timeout=90))
        except Exception as e:
            time.sleep(5 * (attempt + 1))
            err = e
    raise SystemExit(f"ENCODE request failed: {url} ({err})")


def pick_file(files: list[dict], output_type: str):
    c = [f for f in files if f.get("output_type") == output_type and f.get("file_type") == "bed narrowPeak"
         and f.get("assembly") == "GRCh38" and f.get("status") == "released"]
    if not c:
        return None
    pref = [f for f in c if f.get("preferred_default")]
    return pref[0] if pref else max(c, key=lambda f: f.get("file_size", 0))


def stage_files(args) -> None:
    out = resolve(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    chip = pd.read_csv(resolve(args.chip_tsv), sep="\t")
    eclip = pd.read_csv(resolve(args.eclip_tsv), sep="\t")
    chip["target"], eclip["target"] = chip.target.str.upper(), eclip.target.str.upper()
    tg = targets_with_both(chip, eclip)
    rows = []
    for assay, df, ot in (("chip", chip, "IDR thresholded peaks"), ("eclip", eclip, "peaks")):
        sub = df[df.target.isin(tg)]
        for r in sub.itertuples():
            e = get_json(f"{BASE}/experiments/{r.accession}/?format=json&frame=embedded")
            f = pick_file(e.get("files", []), ot)
            rows.append(dict(target=r.target, assay=assay, cell=r.cell, experiment=r.accession, file=f["accession"] if f else "",
                             href=f["href"] if f else "", file_size=f.get("file_size", 0) if f else 0))
            time.sleep(0.2)
        print(f"  {assay}: {len(sub)} experiments", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(out / "files.tsv", sep="\t", index=False)
    print(f"{len(t)} experiments, {int((t.file != '').sum())} with a file, targets {len(tg)}, total size {t.file_size.sum() / 1e6:.0f} MB")


def fetch(url: str, dest: Path, size: int, tries: int = 4) -> None:
    import shutil
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "research-client/1.0"}), timeout=60) as r, open(tmp, "wb") as h:
                shutil.copyfileobj(r, h)
            if size and tmp.stat().st_size != size:
                raise IOError(f"size {tmp.stat().st_size} differs from the expected {size}")
            tmp.rename(dest)
            return
        except Exception as e:
            print(f"  retry {attempt + 1} for {dest.name}: {e}", flush=True)
            time.sleep(5 * (attempt + 1))
    raise SystemExit(f"download failed: {url}")


def list_files(assay_title: str, output_type: str) -> pd.DataFrame:
    url = (f"{BASE}/search/?type=File&file_format=bed&file_format_type=narrowPeak&assembly=GRCh38&status=released&output_type={output_type.replace(' ', '+')}"
           f"&assay_title={assay_title.replace(' ', '+')}&field=dataset&field=accession&field=href&field=file_size&field=preferred_default&limit=all&format=json")
    g = get_json(url)["@graph"]
    return pd.DataFrame([dict(experiment=x["dataset"].strip("/").split("/")[-1], file=x["accession"], href=x["href"], file_size=x.get("file_size", 0),
                              preferred=bool(x.get("preferred_default"))) for x in g])


def stage_files_all(args) -> None:
    """All TF-annotated ChIP-seq sets and all eCLIP sets of the chosen cell lines, one file per experiment."""
    out = resolve(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    chip = pd.read_csv(resolve(args.chip_tsv), sep="\t")
    eclip = pd.read_csv(resolve(args.eclip_tsv), sep="\t")
    chip = chip[chip.investigated_as.fillna("").str.contains("transcription factor")]
    parts = []
    for assay, meta, at, ot in (("chip", chip, "TF ChIP-seq", "IDR thresholded peaks"), ("eclip", eclip, "eCLIP", "peaks")):
        f = list_files(at, ot)
        f = f.sort_values(["preferred", "file_size"], ascending=False).drop_duplicates("experiment")
        m = meta[meta.cell.isin(args.cells)].merge(f, left_on="accession", right_on="experiment", how="left")
        m["assay"] = assay
        m["target"] = m.target.str.upper()
        parts.append(m[["target", "assay", "cell", "accession", "file", "href", "file_size"]].rename(columns={"accession": "experiment"}))
        print(f"  {assay}: {len(f)} experiments with a file, {len(m)} experiments in {args.cells}, without a file {int(m.file.isna().sum())}", flush=True)
    t = pd.concat(parts).dropna(subset=["file"])
    t.to_csv(out / "files_all.tsv", sep="\t", index=False)
    print(f"{len(t)} files, {t.file_size.sum() / 1e6:.0f} MB, targets {t.target.nunique()}, by assay and cell:\n{t.groupby(['assay', 'cell']).target.nunique().to_string()}")


def stage_download(args) -> None:
    from concurrent.futures import ThreadPoolExecutor
    out = resolve(args.out_dir)
    dest = resolve(args.bed_dir)
    dest.mkdir(parents=True, exist_ok=True)
    t = pd.read_csv(out / args.files, sep="\t").fillna("")
    t = t[t.file != ""]
    todo = []
    for r in t.itertuples():
        f = dest / f"{r.target}_{r.assay}_{r.cell.replace(' ', '')}_{r.experiment}_{r.file}.bed.gz"
        if not (f.exists() and f.stat().st_size == int(r.file_size)):
            todo.append((BASE + r.href, f, int(r.file_size)))
    print(f"{len(t) - len(todo)} files present, {len(todo)} to download with {args.workers} workers", flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for _ in ex.map(lambda a: fetch(*a), todo):
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(todo)}", flush=True)
    print(f"all {len(t)} files in {dest}")


def read_bed(path: Path) -> dict[str, np.ndarray]:
    df = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2], names=["c", "s", "e"], compression="gzip" if str(path).endswith(".gz") else None)
    return {c: merge(g[["s", "e"]].to_numpy()) for c, g in df[df.c.isin(CHROMS)].groupby("c")}


def merge(iv: np.ndarray) -> np.ndarray:
    iv = iv[np.argsort(iv[:, 0], kind="stable")]
    out = []
    cs, ce = iv[0]
    for s, e in iv[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            out.append((cs, ce))
            cs, ce = s, e
    out.append((cs, ce))
    return np.array(out, dtype=np.int64)


def inter_len(a: np.ndarray, b: np.ndarray) -> int:
    i = j = tot = 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i, 0], b[j, 0]), min(a[i, 1], b[j, 1])
        if hi > lo:
            tot += hi - lo
        if a[i, 1] < b[j, 1]:
            i += 1
        else:
            j += 1
    return int(tot)


def total(d: dict) -> int:
    return int(sum((v[:, 1] - v[:, 0]).sum() for v in d.values()))


def jaccard(A: dict, B: dict) -> float:
    inter = sum(inter_len(A[c], B[c]) for c in A if c in B)
    union = total(A) + total(B) - inter
    return inter / union if union else 0.0


def union_sets(sets: list[dict]) -> dict:
    out = {}
    for c in set().union(*[set(s) for s in sets]):
        out[c] = merge(np.vstack([s[c] for s in sets if c in s]))
    return out


def residual(J: np.ndarray) -> np.ndarray:
    """log Jaccard without the effect of the ChIP-seq set (row) and of the eCLIP set (column), mostly the number of peaks and their width."""
    L = np.log(J + 1e-6)
    return L - L.mean(1, keepdims=True) - L.mean(0, keepdims=True) + L.mean()


def rank_stats(J: np.ndarray, rows: list[str], cols: list[str]) -> pd.DataFrame:
    res = []
    for i, r in enumerate(rows):
        if r not in cols:
            continue
        j = cols.index(r)
        others = np.delete(J[i], j)
        res.append(dict(protein=r, own=J[i, j], others_mean=float(others.mean()), others_sd=float(others.std()),
                        z=float((J[i, j] - others.mean()) / max(others.std(), 1e-12)), rank=int(1 + (others > J[i, j]).sum()), n_others=len(others)))
    return pd.DataFrame(res)


def mantel_spearman(J: np.ndarray, rows: list[str], cols: list[str], emb: np.ndarray, sym: list[str], layer: int, n_perm: int, rng):
    from scipy.stats import rankdata
    idx = {s: i for i, s in enumerate(sym)}
    x = emb[:, layer] - emb[:, layer].mean(0, keepdims=True)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)
    C = x @ x.T
    ri = np.array([idx[r] for r in rows])
    ci = np.array([idx[c] for c in cols])
    M = C[np.ix_(ri, ci)]
    off = np.array([[a != b for b in cols] for a in rows])
    rj = rankdata(J[off])
    rho = float(np.corrcoef(rankdata(M[off]), rj)[0, 1])
    ge = 0
    for _ in range(n_perm):
        p = rng.permutation(len(sym))
        Mp = C[np.ix_(p[ri], p[ci])]
        ge += abs(float(np.corrcoef(rankdata(Mp[off]), rj)[0, 1])) >= abs(rho)
    return rho, (ge + 1) / (n_perm + 1)


def stage_overlap(args) -> None:
    out = resolve(args.out_dir)
    dest = resolve(args.bed_dir)
    t = pd.read_csv(out / "files.tsv", sep="\t").fillna("")
    t = t[t.file != ""]
    sets = {}
    for (tg, assay, cell), g in t.groupby(["target", "assay", "cell"]):
        beds = [read_bed(dest / f"{r.target}_{r.assay}_{r.cell.replace(' ', '')}_{r.experiment}_{r.file}.bed.gz") for r in g.itertuples()]
        sets[(tg, assay, cell)] = union_sets(beds)
    sizes = pd.DataFrame([dict(target=k[0], assay=k[1], cell=k[2], bp=total(v)) for k, v in sets.items()])
    z = np.load(resolve(args.embeddings), allow_pickle=True)
    emb, sym = z["emb"].astype(np.float32), [str(s) for s in z["symbols"]]
    rng = np.random.default_rng(args.seed)
    rows_out, mats = [], {}
    for cell in sorted({k[2] for k in sets}):
        ch = sorted({k[0] for k in sets if k[1] == "chip" and k[2] == cell})
        ec = sorted({k[0] for k in sets if k[1] == "eclip" and k[2] == cell})
        if len(ch) < 5 or len(ec) < 5:
            continue
        J = np.array([[jaccard(sets[(a, "chip", cell)], sets[(b, "eclip", cell)]) for b in ec] for a in ch])
        pd.DataFrame(J, index=ch, columns=ec).to_csv(out / f"jaccard_{cell.replace(' ', '')}.tsv", sep="\t")
        rs = rank_stats(J, ch, ec)
        rs.insert(0, "cell", cell)
        rows_out.append(rs)
        own_first = int((rs["rank"] == 1).sum())
        print(f"{cell}: {len(ch)} ChIP sets x {len(ec)} eCLIP sets, own protein pairs {len(rs)}, rank 1 in {own_first}, median z {rs.z.median():.2f}, "
              f"median own Jaccard {rs.own.median():.4f} against {rs.others_mean.median():.4f} for other proteins")
        R = residual(J)
        rr = rank_stats(R, ch, ec)
        rr.insert(0, "cell", cell)
        rr.to_csv(out / f"own_protein_overlap_residual_{cell.replace(' ', '')}.tsv", sep="\t", index=False)
        print(f"   after removing set-size effects: rank 1 in {int((rr['rank'] == 1).sum())} of {len(rr)}, median rank {int(rr['rank'].median())} of {int(rr.n_others.iloc[0]) + 1}, median z {rr.z.median():.2f}")
        for layer in (0, 18, 21, 33):
            rho, p = mantel_spearman(J, ch, ec, emb, sym, layer, args.n_perm, rng)
            rho2, p2 = mantel_spearman(R, ch, ec, emb, sym, layer, args.n_perm, rng)
            print(f"   Spearman vs centred cosine, layer {layer}: raw Jaccard {rho:+.3f} (p {p:.3f}), size-adjusted {rho2:+.3f} (p {p2:.3f})")
            mats[f"{cell}|{layer}"] = dict(raw=[rho, p], size_adjusted=[rho2, p2])
    res = pd.concat(rows_out) if rows_out else pd.DataFrame()
    res.to_csv(out / "own_protein_overlap.tsv", sep="\t", index=False)
    sizes.to_csv(out / "peak_set_sizes.tsv", sep="\t", index=False)
    (out / "mantel.json").write_text(json.dumps(mats, indent=1))
    print("saved", out / "own_protein_overlap.tsv")



CH_OFF = {c: i * 300_000_000 for i, c in enumerate(CHROMS)}


class PeakSet:
    """Merged intervals of one protein, cell line and assay on a single axis (chromosome offset 3e8)."""

    def __init__(self, iv: np.ndarray):
        iv = merge(iv)
        self.starts, self.ends = iv[:, 0].astype(np.int64), iv[:, 1].astype(np.int64)
        self.lens = self.ends - self.starts
        self.prefix = np.concatenate([[0], np.cumsum(self.lens)[:-1]])
        self.bp = int(self.lens.sum())

    def cum(self, x: np.ndarray) -> np.ndarray:
        k = np.searchsorted(self.starts, x, side="right") - 1
        out = np.zeros(len(x), dtype=np.float64)
        m = k >= 0
        kk = k[m]
        out[m] = self.prefix[kk] + np.minimum(x[m] - self.starts[kk], self.lens[kk])
        return out


def read_axis(path: Path) -> np.ndarray:
    df = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2], names=["c", "s", "e"], compression="gzip" if str(path).endswith(".gz") else None)
    df = df[df.c.isin(CHROMS)]
    off = df.c.map(CH_OFF).to_numpy()
    return np.stack([df.s.to_numpy() + off, df.e.to_numpy() + off], 1)


def jaccard_fast(a: PeakSet, b: PeakSet) -> float:
    if a.bp > b.bp:
        a, b = b, a
    inter = float((b.cum(a.ends) - b.cum(a.starts)).sum())
    union = a.bp + b.bp - inter
    return inter / union if union else 0.0


def stage_overlap_all(args) -> None:
    from scipy.stats import rankdata
    out = resolve(args.out_dir)
    dest = resolve(args.bed_dir)
    t = pd.read_csv(out / "files_all.tsv", sep="\t").fillna("")
    t = t[t.file != ""]
    ann = pd.read_csv(resolve(args.annot), sep="\t").fillna("")
    z = np.load(resolve(args.embeddings), allow_pickle=True)
    emb, sym = z["emb"].astype(np.float32), [str(x) for x in z["symbols"]]
    assert sym == list(ann.symbol), "annotation and embedding tables differ"
    pidx = {x: i for i, x in enumerate(sym)}
    rng = np.random.default_rng(args.seed)
    summary = {}
    for cell in args.cells:
        sets = {}
        for (tg, assay), g in t[t.cell == cell].groupby(["target", "assay"]):
            arrs = [read_axis(dest / f"{r.target}_{r.assay}_{r.cell.replace(' ', '')}_{r.experiment}_{r.file}.bed.gz") for r in g.itertuples()]
            sets[(tg, assay)] = PeakSet(np.vstack(arrs))
        tfs = sorted(k[0] for k in sets if k[1] == "chip" and k[0] in pidx)
        rbs = sorted(k[0] for k in sets if k[1] == "eclip" and k[0] in pidx)
        print(f"{cell}: {len(tfs)} ChIP-seq sets x {len(rbs)} eCLIP sets = {len(tfs) * len(rbs)} pairs", flush=True)
        J = np.array([[jaccard_fast(sets[(a, "chip")], sets[(b, "eclip")]) for b in rbs] for a in tfs])
        L = np.log(J + 1e-6)
        R = L - L.mean(1, keepdims=True) - L.mean(0, keepdims=True) + L.mean()
        tag = cell.replace(" ", "")
        pd.DataFrame(J, index=tfs, columns=rbs).to_csv(out / f"jaccard_all_{tag}.tsv.gz", sep="\t", compression="gzip")
        pd.DataFrame(R, index=tfs, columns=rbs).to_csv(out / f"jaccard_all_residual_{tag}.tsv.gz", sep="\t", compression="gzip")
        off = np.array([[a != b for b in rbs] for a in tfs])
        res = {"n_tf": len(tfs), "n_rbp": len(rbs), "n_pairs": int(off.sum())}
        ri, ci = np.array([pidx[a] for a in tfs]), np.array([pidx[b] for b in rbs])
        # own protein (proteins that are in both lists)
        both = [a for a in tfs if a in rbs]
        if both:
            rr = rank_stats(R, tfs, rbs)
            res["own_pairs"] = len(rr)
            res["own_rank1"] = int((rr["rank"] == 1).sum())
            res["own_median_rank"] = float(rr["rank"].median())
            res["own_median_z_size_adjusted"] = float(rr.z.median())
        # Spearman with centred cosine of the embeddings
        for layer in args.layers:
            x = emb[:, layer] - emb[:, layer].mean(0, keepdims=True)
            x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)
            C = x @ x.T
            M = C[np.ix_(ri, ci)]
            for name, A in (("raw", J), ("size_adjusted", R)):
                rb = rankdata(A[off])
                rho = float(np.corrcoef(rankdata(M[off]), rb)[0, 1])
                ge = 0
                for _ in range(args.n_perm):
                    pr, pc = rng.permutation(len(sym)), rng.permutation(len(sym))
                    Mp = C[np.ix_(pr[ri], pc[ci])]
                    ge += abs(float(np.corrcoef(rankdata(Mp[off]), rb)[0, 1])) >= abs(rho)
                res[f"spearman_cos_layer{layer}_{name}"] = [rho, (ge + 1) / (args.n_perm + 1)]
            print(f"   layer {layer}: Spearman with centred cosine, raw {res[f'spearman_cos_layer{layer}_raw'][0]:+.3f}, size-adjusted {res[f'spearman_cos_layer{layer}_size_adjusted'][0]:+.3f} (p {res[f'spearman_cos_layer{layer}_size_adjusted'][1]:.3f})", flush=True)
        # domain group, RNA-binding keyword and RBP-likeness of the TF against its mean overlap with eCLIP sets
        grp = ann.domain_group.to_numpy()
        gt, gr = grp[ri], grp[ci]
        known = ~np.isin(gt, ["no Pfam entry", "other Pfam"])
        knownr = ~np.isin(gr, ["no Pfam entry", "other Pfam"])
        same = (gt[:, None] == gr[None, :]) & known[:, None] & knownr[None, :] & off
        diff = (gt[:, None] != gr[None, :]) & known[:, None] & knownr[None, :] & off
        if same.sum() > 20:
            res["same_domain_group_pairs"] = int(same.sum())
            res["mean_residual_same_group"] = float(R[same].mean())
            res["mean_residual_different_group"] = float(R[diff].mean())
        summary[cell] = res
        print("  ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items() if not k.startswith("spearman_cos")}, flush=True)
    (out / "overlap_all_summary.json").write_text(json.dumps(summary, indent=1))
    print("saved", out / "overlap_all_summary.json")



def stage_cross_cell(args) -> None:
    """Does a protein keep its peaks between two cell lines? Jaccard of the K562 set against the HepG2 sets, own protein against the others."""
    out = resolve(args.out_dir)
    dest = resolve(args.bed_dir)
    t = pd.read_csv(out / "files_all.tsv", sep="\t").fillna("")
    t = t[t.file != ""]
    rows = []
    c1, c2 = args.cells[0], args.cells[1]
    for assay in ("eclip", "chip"):
        sets = {}
        for (tg, cell), g in t[t.assay == assay].groupby(["target", "cell"]):
            arrs = [read_axis(dest / f"{r.target}_{r.assay}_{r.cell.replace(' ', '')}_{r.experiment}_{r.file}.bed.gz") for r in g.itertuples()]
            sets[(tg, cell)] = PeakSet(np.vstack(arrs))
        a = sorted(k[0] for k in sets if k[1] == c1)
        b = sorted(k[0] for k in sets if k[1] == c2)
        both = sorted(set(a) & set(b))
        if len(both) < 5:
            continue
        J = np.array([[jaccard_fast(sets[(x, c1)], sets[(y, c2)]) for y in b] for x in a])
        L = np.log(J + 1e-6)
        R = L - L.mean(1, keepdims=True) - L.mean(0, keepdims=True) + L.mean()
        for name, M in (("raw", J), ("size_adjusted", R)):
            rs = rank_stats(M, a, b)
            rows.append(dict(assay=assay, statistic=name, n_proteins_in_both=len(both), n_columns=len(b), own_rank1=int((rs["rank"] == 1).sum()),
                             top5_fraction=float((rs["rank"] <= 5).mean()), median_rank=float(rs["rank"].median()), median_z=float(rs.z.median()),
                             median_own_jaccard=float(rs.own.median()), median_other_jaccard=float(rs.others_mean.median())))
            rs.assign(assay=assay, statistic=name).to_csv(out / f"cross_cell_{assay}_{name}.tsv", sep="\t", index=False)
    res = pd.DataFrame(rows)
    res.to_csv(out / "cross_cell_summary.tsv", sep="\t", index=False)
    print(res.round(4).to_string(index=False))


def stage_selftest(args) -> None:
    rng = np.random.default_rng(0)
    L = 200_000_000

    def make(n, centres=None):
        s = np.sort(rng.integers(0, L, n)) if centres is None else np.sort(centres)
        return {"chr1": merge(np.stack([s, s + 300], 1))}

    prot = ["P%d" % i for i in range(8)]
    shared = [rng.integers(0, L, 400) for _ in prot]
    chip = {p: make(0, np.concatenate([shared[i], rng.integers(0, L, 400)])) for i, p in enumerate(prot)}
    ecl = {p: make(0, np.concatenate([shared[i], rng.integers(0, L, 400)])) for i, p in enumerate(prot)}
    J = np.array([[jaccard(chip[a], ecl[b]) for b in prot] for a in prot])
    rs = rank_stats(J, prot, prot)
    assert (rs["rank"] == 1).all() and rs.z.min() > 3, "planted own-protein overlap not recovered"
    for _ in range(5):
        ia = np.sort(rng.integers(0, 10_000_000, (50, 2)), axis=1)
        ib = np.sort(rng.integers(0, 10_000_000, (80, 2)), axis=1)
        ia, ib = ia[ia[:, 1] > ia[:, 0]], ib[ib[:, 1] > ib[:, 0]]
        A, B = PeakSet(ia), PeakSet(ib)
        ref = inter_len(merge(ia), merge(ib))
        got = float((B.cum(A.ends) - B.cum(A.starts)).sum())
        assert abs(ref - got) < 1e-6, "fast intersection differs from the reference"
        assert abs(jaccard_fast(A, B) - jaccard({"c": merge(ia)}, {"c": merge(ib)})) < 1e-9, "fast Jaccard differs"
    big = J * np.array([[1.0], [2.0], [4.0], [1.0], [3.0], [1.0], [1.0], [2.0]]) * np.array([[1, 5, 1, 1, 2, 1, 1, 1]])
    assert abs(residual(big).mean()) < 1e-6 and (rank_stats(residual(big), prot, prot)["rank"] == 1).all(), "size adjustment wrong"
    a = np.array([[0, 10], [20, 30]])
    b = np.array([[5, 25]])
    assert inter_len(a, b) == 10, "interval intersection wrong"
    assert abs(jaccard({"c": a}, {"c": b}) - 10 / 30) < 1e-9, "Jaccard wrong"
    assert merge(np.array([[0, 5], [3, 8], [10, 12]])).tolist() == [[0, 8], [10, 12]], "merge wrong"
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "x.bed.gz"
        with gzip.open(f, "wt") as h:
            h.write("chr1\t0\t10\tn\t0\t.\nchrUn\t0\t10\tn\t0\t.\n")
        assert list(read_bed(f)) == ["chr1"], "bed reader keeps non-canonical contigs"
    print("selftest passed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["files", "files_all", "download", "overlap", "overlap_all", "cross_cell", "selftest"])
    ap.add_argument("--out_dir", default="results/encode_peaks")
    ap.add_argument("--bed_dir", default="data/external/encode_chip_eclip")
    ap.add_argument("--chip_tsv", default="results/encode_map/encode_chip_grch38.tsv")
    ap.add_argument("--eclip_tsv", default="results/encode_map/encode_eclip_grch38.tsv")
    ap.add_argument("--embeddings", default="results/encode_map/esm2_layers.npz")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--files", default="files.tsv", help="download: file table inside out_dir")
    ap.add_argument("--cells", nargs="+", default=["K562", "HepG2"])
    ap.add_argument("--annot", default="results/encode_map/proteins_annot.tsv")
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 18, 21, 33])
    ap.add_argument("--n_perm", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    {"files": stage_files, "files_all": stage_files_all, "download": stage_download, "overlap": stage_overlap, "overlap_all": stage_overlap_all, "cross_cell": stage_cross_cell, "selftest": stage_selftest}[args.stage](args)


if __name__ == "__main__":
    main()
