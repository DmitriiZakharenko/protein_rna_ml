#!/usr/bin/env python3
"""
summarize_v3b_runs.py
---------------------
Tables of every finished run under results/v3b_frozen, v3b_diag, v3b_hybrid, v3b_frozen_layers, v3b_frozen_tf (results.json of scripts 67 / 71):
probe AUROC, standard-test AUROC, shuffled-protein probe AUROC, seen-protein positive-control AUROC, best epoch.
A second table gives mean +- sd (sample sd, ddof=1) over seeds for runs whose directory name ends in `_s<seed>`, one row per configuration.

Usage:  python scripts/summarize_v3b_runs.py
"""
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASES = ("v3b_frozen", "v3b_diag", "v3b_hybrid", "v3b_frozen_layers", "v3b_frozen_tf")
rows, groups = [], {}
for base in BASES:
    for f in sorted((ROOT / "results" / base).glob("*/results.json")):
        d = json.loads(f.read_text())
        g = lambda *k: next((d[x]["auroc"] for x in k if x in d and isinstance(d[x], dict) and "auroc" in d[x]), None)  # noqa: E731
        probe, std = g("probe", "probe_stack"), g("test_standard", "test_standard_stack")
        rows.append((f"{base}/{f.parent.name}", probe, std,
                     (d.get("probe_shuffled_protein") or {}).get("auroc"), d.get("seen_probe_auroc"), d.get("best_epoch")))
        m = re.match(r"(.+)_s(\d+)$", f.parent.name)
        if m:
            groups.setdefault(f"{base}/{m.group(1)}", []).append((int(m.group(2)), probe, std))
fmt = lambda x: "   -  " if x is None else f"{x:6.3f}"  # noqa: E731
print(f"{'run':<46}{'probe':>7}{'standard':>9}{'shuf-prot':>10}{'seen-probe':>11}{'epoch':>6}")
for r in rows:
    print(f"{r[0]:<46}{fmt(r[1])}{fmt(r[2])}{fmt(r[3])}{fmt(r[4])}{(r[5] if r[5] is not None else '-'):>6}")
print(f"\n{len(rows)} finished runs")


def ms(v):
    v = [x for x in v if x is not None]
    if not v:
        return "      -      "
    return f"{np.mean(v):.3f} +- {np.std(v, ddof=1):.3f}" if len(v) > 1 else f"{v[0]:.3f} (1 seed)"


print(f"\n{'configuration (mean +- sd over seeds)':<46}{'n':>3}  {'probe':<18}{'standard':<18}")
for k, v in sorted(groups.items()):
    print(f"{k:<46}{len(v):>3}  {ms([x[1] for x in v]):<18}{ms([x[2] for x in v]):<18}")
