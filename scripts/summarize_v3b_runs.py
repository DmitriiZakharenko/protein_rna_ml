#!/usr/bin/env python3
"""
summarize_v3b_runs.py
---------------------
One table of every finished run under results/v3b_frozen, results/v3b_diag, results/v3b_hybrid (results.json of scripts 67 / 71):
probe AUROC, standard-test AUROC, shuffled-protein probe AUROC, seen-protein positive-control AUROC, best epoch.

Usage:  python scripts/summarize_v3b_runs.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
rows = []
for base in ("v3b_frozen", "v3b_diag", "v3b_hybrid"):
    for f in sorted((ROOT / "results" / base).glob("*/results.json")):
        d = json.loads(f.read_text())
        g = lambda *k: next((d[x]["auroc"] for x in k if x in d and isinstance(d[x], dict) and "auroc" in d[x]), None)  # noqa: E731
        rows.append((f"{base}/{f.parent.name}", g("probe", "probe_stack"), g("test_standard", "test_standard_stack"),
                     (d.get("probe_shuffled_protein") or {}).get("auroc"), d.get("seen_probe_auroc"), d.get("best_epoch")))
fmt = lambda x: "   -  " if x is None else f"{x:6.3f}"  # noqa: E731
print(f"{'run':<42}{'probe':>7}{'standard':>9}{'shuf-prot':>10}{'seen-probe':>11}{'epoch':>6}")
for r in rows:
    print(f"{r[0]:<42}{fmt(r[1])}{fmt(r[2])}{fmt(r[3])}{fmt(r[4])}{(r[5] if r[5] is not None else '-'):>6}")
print(f"\n{len(rows)} finished runs")
