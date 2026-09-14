#!/usr/bin/env python3
"""WP2 — is the serotonin effect condition-dependent?

The mechanism question. A fixed displacement (M2) predicts the same 5-HT vector whatever
the animal is doing; context gating (M3) predicts it differs by condition. The four-way
splits stored in the manifold data cross stimulation with ONE context -- the animal's
choice -- so that is the context this can test. Stimulus side and block prior are not
stored separately and would need the PSTHs regenerating from spikes.

Two questions, both direct vector comparisons on data already in hand:

  A. Is the 5-HT displacement the same for left and right choices?
        dL = L_opto - L_no_opto      dR = R_opto - R_no_opto
     Parallel (small 1-|cos|) = additive, a fixed displacement.
     Divergent (large 1-|cos|) = context-gated.

  B. Does the choice axis itself change under stimulation?
        c_stim = L_opto - R_opto     c_nostim = L_no_opto - R_no_opto
     This is WP3's rotation test, which the review asked to have specified. An angle
     between the two says the readout ROTATED; a magnitude ratio says it was SCALED
     (selective gain, M5). Decoding accuracy is blind to a pure rotation, which is why
     it needs measuring separately.

Both are tested against the same 500 stored shuffles, TWO-TAILED -- the earlier pass
tested only the "more orthogonal than chance" tail and would have missed the fact that
the displacements are more PARALLEL than chance, which is the informative direction here.

Usage:  python3 wp2_condition_dependence.py
"""

from __future__ import annotations

import argparse
import json
import sys
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.decomposition import PCA

from wp1_orthogonality import (ORTH_WIN, N_DIM, MIN_NEURONS, SPLITS,
                               find_data, load_sessions, _collapse, _one_minus_cos)


def two_tailed_p(obs: float, null: np.ndarray) -> float:
    """Fraction of the null at least as extreme as the observation, either side."""
    centre = null.mean()
    return float((np.abs(null - centre) >= abs(obs - centre)).mean())


def condition_dependence(dicts, n_dim=N_DIM, win=ORTH_WIN):
    time_ax = dicts[0]["time"]
    T = len(time_ax)
    sel = (time_ax >= win[0]) & (time_ax <= win[1])
    real = {s: np.vstack([d[s] for d in dicts]) for s in SPLITS}
    if real["L_opto"].shape[0] < MIN_NEURONS:
        return None
    n_shuf = min(d["n_shuffles"] for d in dicts)

    def stat(m):
        stacked = np.vstack([m[s].T for s in SPLITS])
        pcs = PCA(n_components=min(n_dim, stacked.shape[1])).fit_transform(stacked)
        _, _, _, _, L_o, R_o, L_n, R_n = _collapse(pcs, T)
        dL, dR = L_o - L_n, R_o - R_n              # A: 5-HT effect per choice
        c_s, c_n = L_o - R_o, L_n - R_n            # B: choice axis per condition
        return dict(
            a_angle=float(np.mean(_one_minus_cos(dL, dR)[sel])),
            a_logmag=float(np.mean(np.log2(np.maximum(
                np.linalg.norm(dL, axis=1) / np.maximum(np.linalg.norm(dR, axis=1), 1e-12),
                1e-12))[sel])),
            b_angle=float(np.mean(_one_minus_cos(c_s, c_n)[sel])),
            b_logmag=float(np.mean(np.log2(np.maximum(
                np.linalg.norm(c_s, axis=1) / np.maximum(np.linalg.norm(c_n, axis=1), 1e-12),
                1e-12))[sel])))

    obs = stat(real)
    nulls = {k: [] for k in obs}
    for i in range(n_shuf):
        try:
            sh = {s: np.vstack([d["shuffle"][s][:, :, i] for d in dicts]) for s in SPLITS}
        except (KeyError, IndexError):
            continue
        if np.isnan(sh["L_opto"]).any():
            continue
        v = stat(sh)
        for k in nulls:
            nulls[k].append(v[k])
    if len(nulls["a_angle"]) < 20:
        return None

    out = dict(n_neurons=int(real["L_opto"].shape[0]), n_sessions=len(dicts),
               n_shuffles=len(nulls["a_angle"]))
    for k, v in obs.items():
        arr = np.array(nulls[k])
        out[k] = v
        out[f"{k}_null"] = float(arr.mean())
        out[f"{k}_p"] = two_tailed_p(v, arr)
        out[f"{k}_z"] = float((v - arr.mean()) / (arr.std() + 1e-12))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(Path.home() /
                    "Projects/mainenlab/serotonin-analysis/Data"))
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    data_dir, align = find_data(Path(args.data_root))
    sessions = load_sessions(data_dir)
    sert = [d for d in sessions if d["sert-cre"] == 1]
    wt = [d for d in sessions if d["sert-cre"] == 0]
    print(f"Data: {data_dir.name} (aligned to {align}) | SERT {len(sert)} files, "
          f"{len({d['subject'] for d in sert})} animals")

    print("\n=== POOLED ===")
    rows = []
    for label, grp in (("SERT-cre", sert), ("wild-type", wt)):
        r = condition_dependence(grp)
        if not r:
            continue
        rows.append(dict(group=label, **r))
        print(f"\n{label} ({r['n_neurons']} neurons)")
        print(f"  A · 5-HT displacement across choices   1-|cos| = {r['a_angle']:.3f} "
              f"(null {r['a_angle_null']:.3f}, z = {r['a_angle_z']:+.2f}, p = {r['a_angle_p']:.3f})")
        print(f"      magnitude |dL|/|dR| = {2**r['a_logmag']:.2f}x "
              f"(null {2**r['a_logmag_null']:.2f}x, p = {r['a_logmag_p']:.3f})")
        print(f"  B · choice axis across conditions      1-|cos| = {r['b_angle']:.3f} "
              f"(null {r['b_angle_null']:.3f}, z = {r['b_angle_z']:+.2f}, p = {r['b_angle_p']:.3f})")
        print(f"      magnitude |stim|/|no stim| = {2**r['b_logmag']:.2f}x "
              f"(null {2**r['b_logmag_null']:.2f}x, p = {r['b_logmag_p']:.3f})")
    pd.DataFrame(rows).to_csv(out / "wp2_pooled.csv", index=False)

    print("\n=== PER ANIMAL (SERT-cre) ===")
    per = []
    for subj in sorted({d["subject"] for d in sert}):
        r = condition_dependence([d for d in sert if d["subject"] == subj])
        if r:
            per.append(dict(subject=subj, **r))
    pa = pd.DataFrame(per)
    pa.to_csv(out / "wp2_per_animal.csv", index=False)
    if len(pa):
        print(pa[["subject", "n_neurons", "a_angle", "a_angle_null", "a_angle_p",
                  "b_angle", "b_angle_null", "b_angle_p"]].round(3).to_string(index=False))
        n = len(pa)
        kA = int((pa.a_angle < pa.a_angle_null).sum())
        kB = int((pa.b_angle > pa.b_angle_null).sum())
        pA = binomtest(kA, n, 0.5).pvalue
        pB = binomtest(kB, n, 0.5).pvalue
        print(f"\n  A · more PARALLEL than chance in {kA}/{n} animals  "
              f"(sign test p = {pA:.3f}) → {'additive' if kA > n/2 else 'context-gated'}")
        print(f"  B · choice axis MORE rotated than chance in {kB}/{n} animals "
              f"(sign test p = {pB:.3f})")
        print(f"      median choice-axis magnitude ratio "
              f"{2**pa.b_logmag.median():.2f}x under stimulation")
        json.dump(dict(n=n, A_parallel=kA, A_p=pA, B_rotated=kB, B_p=pB,
                       median_choice_mag_ratio=float(2**pa.b_logmag.median())),
                  open(out / "wp2_summary.json", "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
