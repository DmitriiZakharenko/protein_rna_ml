#!/usr/bin/env python3
"""
build_esm_layer_report.py
-------------------------
Builds reports/esm_layer_similarity.html (method, result table, figure) for script 80.

Values: results/esm_layer_profile/layer_correlation.tsv when present (columns layer, spearman_raw, spearman_centred); otherwise the
values typed in below, copied from the VM log results/esm_layer_analyze.log (175 proteins, 500 permutations).
If results/esm_layer_profile/layer_bootstrap.tsv exists (script 80, stage bootstrap) its 95% intervals are drawn.

Usage:  python scripts/build_esm_layer_report.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "results" / "esm_layer_profile"
RAW = [0.1118, 0.0757, 0.0718, 0.0716, 0.0744, 0.0815, 0.0838, 0.0805, 0.0755, 0.066, 0.0679, 0.0713, 0.0857, 0.0894, 0.082, 0.0818,
       0.0797, 0.0851, 0.0905, 0.0916, 0.0939, 0.0974, 0.1041, 0.1029, 0.1049, 0.1074, 0.1092, 0.1084, 0.1087, 0.1135, 0.1248, 0.1308,
       0.1322, 0.0873]
CEN = [0.1194, 0.1174, 0.1122, 0.1127, 0.1183, 0.1197, 0.1193, 0.1165, 0.1172, 0.0821, 0.0933, 0.0956, 0.1161, 0.1287, 0.1124, 0.1086,
       0.1117, 0.1226, 0.135, 0.1368, 0.1378, 0.1455, 0.1584, 0.1548, 0.1554, 0.1566, 0.1565, 0.152, 0.1466, 0.1479, 0.1481, 0.1436,
       0.1236, 0.0997]


def load() -> pd.DataFrame:
    f = D / "layer_correlation.tsv"
    if f.exists():
        df = pd.read_csv(f, sep="\t")[["layer", "spearman_raw", "spearman_centred"]]
    else:
        df = pd.DataFrame({"layer": range(34), "spearman_raw": RAW, "spearman_centred": CEN})
    b = D / "layer_bootstrap.tsv"
    return df.merge(pd.read_csv(b, sep="\t"), on="layer", how="left") if b.exists() else df


def svg(df: pd.DataFrame) -> str:
    W, H, L, R, T, B = 760, 380, 62, 20, 20, 52
    ymax = 0.2
    x = lambda v: L + (W - L - R) * v / 33  # noqa: E731
    y = lambda v: H - B - (H - T - B) * v / ymax  # noqa: E731
    o = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Spearman correlation by ESM-2 layer">']
    for t in (0, 0.05, 0.10, 0.15, 0.20):
        o.append(f'<line x1="{L}" x2="{W - R}" y1="{y(t):.1f}" y2="{y(t):.1f}" stroke="var(--line)"/>'
                 f'<text x="{L - 8}" y="{y(t) + 4:.1f}" text-anchor="end" font-size="13">{t:.2f}</text>')
    for t in range(0, 34, 3):
        o.append(f'<text x="{x(t):.1f}" y="{H - B + 20}" text-anchor="middle" font-size="13">{t}</text>')
    o.append(f'<text x="{(L + W - R) / 2:.0f}" y="{H - 10}" text-anchor="middle" font-size="14">ESM-2 650M layer (0 = token embedding, 33 = last)</text>')
    o.append(f'<text transform="rotate(-90)" x="{-(T + H - B) / 2:.0f}" y="16" text-anchor="middle" font-size="14">Spearman with 7-mer Z-score similarity</text>')
    for col, name, color, dash in (("spearman_raw", "cosine, raw", "#0b57d0", ""), ("spearman_centred", "cosine, centred over proteins", "#b3541e", "")):
        if f"{col}_lo" in df.columns and df[f"{col}_lo"].notna().all():
            up = [f"{x(a):.1f},{y(b):.1f}" for a, b in zip(df.layer, df[f"{col}_hi"])]
            dn = [f"{x(a):.1f},{y(b):.1f}" for a, b in zip(df.layer[::-1], df[f"{col}_lo"][::-1])]
            o.append(f'<polygon points="{" ".join(up + dn)}" fill="{color}" opacity="0.13"/>')
        pts = " ".join(f"{x(a):.1f},{y(b):.1f}" for a, b in zip(df.layer, df[col]))
        o.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.2"{dash}/>')
        o += [f'<circle cx="{x(a):.1f}" cy="{y(b):.1f}" r="2.8" fill="{color}"/>' for a, b in zip(df.layer, df[col])]
    o.append(f'<line x1="{x(33):.1f}" x2="{x(33):.1f}" y1="{T}" y2="{H - B}" stroke="var(--mut)" stroke-dasharray="4 4"/>'
             f'<text x="{x(33) - 6:.1f}" y="{T + 12}" text-anchor="end" font-size="13">layer 33 (used by the frozen heads)</text>')
    o.append(f'<rect x="{L + 8}" y="{T + 4}" width="12" height="3" fill="#0b57d0"/><text x="{L + 26}" y="{T + 10}" font-size="13">cosine, raw</text>'
             f'<rect x="{L + 8}" y="{T + 22}" width="12" height="3" fill="#b3541e"/><text x="{L + 26}" y="{T + 28}" font-size="13">cosine, centred over proteins</text>')
    return "".join(o) + "</svg>"


def main() -> None:
    df = load()
    ci = "Shaded bands are 95% bootstrap intervals over proteins." if "spearman_raw_lo" in df.columns else "No confidence intervals (not yet computed)."
    pick = [0, 5, 10, 15, 20, 22, 26, 30, 32, 33]
    rows = "".join(f"<tr><td>{int(r.layer)}</td><td>{r.spearman_raw:.3f}</td><td>{r.spearman_centred:.3f}</td></tr>"
                   for r in df[df.layer.isin(pick)].itertuples())
    best_r, best_c = df.loc[df.spearman_raw.idxmax()], df.loc[df.spearman_centred.idxmax()]
    html_ = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ESM-2 layers and RNAcompete similarity</title><style>
:root{{color-scheme:light;--bg:#fff;--fg:#17191c;--mut:#555d68;--line:#d6dbe1;--card:#f4f6f8}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font:17px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}}
main{{max-width:900px;margin:0 auto;padding:28px 20px 70px}}h1{{font-size:28px;margin:0 0 6px;line-height:1.2}}h2{{font-size:21px;margin:30px 0 8px}}
.meta{{color:var(--mut);font-size:15px;margin:0 0 18px}}table{{border-collapse:collapse;width:100%;max-width:520px;font-size:15.5px}}
th,td{{border-bottom:1px solid var(--line);padding:6px 10px;text-align:left}}th{{background:var(--card)}}svg text{{fill:var(--fg)}}
.note{{color:var(--mut);font-size:15px}}li{{margin:6px 0}}@media (max-width:760px){{body{{font-size:16px}}}}
</style></head><body><main>
<h1>Which ESM-2 layer reflects RNA binding similarity of RBPs?</h1>
<p class="meta">Dmitrii Zakharenko, 19.11.26. Test on RNAcompete data; protein embeddings are frozen ESM-2 650M.</p>
<h2>Method</h2><ol>
<li>Binding vector per protein: published RNAcompete 7-mer Z-scores (16,382 7-mers; mean over the experiments of the same protein sequence); 175 proteins.</li>
<li>Binding-similarity matrix: Pearson correlation between the 175 vectors.</li>
<li>Embedding-similarity matrices: mean-pooled residue embeddings of every ESM-2 layer (0-33), cosine similarity, raw and centred over proteins. Proteins longer than 1,022 residues would use windows of 1,022 with overlap 511 (none of the 175 is that long).</li>
<li>Per layer: Spearman correlation between the upper triangles of the two matrices (15,225 protein pairs); Mantel permutation test (500 permutations).</li></ol>
<h2>Result</h2>
<figure style="margin:8px 0">{svg(df)}<figcaption class="note">{ci}</figcaption></figure>
<table><tr><th>Layer</th><th>raw</th><th>centred</th></tr>{rows}</table>
<ul>
<li>All layers correlate positively with binding similarity (rho {df.spearman_raw.min():.2f}-{df.spearman_centred.max():.2f}); Mantel p = 0.002 for every layer (smallest value possible with 500 permutations).</li>
<li>Maximum: layer {int(best_r.layer)} for raw cosine ({best_r.spearman_raw:.3f}), layer {int(best_c.layer)} for centred cosine ({best_c.spearman_centred:.3f}).</li>
<li>The last layer (33), used by the frozen heads, is lower: {df.spearman_raw.iloc[-1]:.3f} raw, {df.spearman_centred.iloc[-1]:.3f} centred.</li>
<li>Layer 0 (token embedding, i.e. amino-acid composition) reaches {df.spearman_raw.iloc[0]:.3f} / {df.spearman_centred.iloc[0]:.3f}, similar to most deep layers.</li></ul>
<h2>Limits</h2><ul>
<li>The permutation test compares each layer with zero; differences between layers are not tested unless intervals are drawn above.</li>
<li>Protein pairs are not independent (each protein is in 174 pairs). Several proteins belong to the same families.</li>
<li>The whole protein sequence is pooled, not the RNA-binding domain only.</li></ul>
<p class="note">Source: scripts/80_esm_layer_profile_similarity.py; log in EXPERIMENT_LOG.md.</p>
</main></body></html>"""
    out = ROOT / "reports" / "esm_layer_similarity.html"
    out.write_text(html_)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
