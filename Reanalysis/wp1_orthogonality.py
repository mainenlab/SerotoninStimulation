#!/usr/bin/env python3
"""WP1 — does the orthogonality result survive its own caveats?

Reproduces the published statistic exactly, then varies the things the published
analysis holds fixed. Three analyses, all on the stored per-session manifold data
(condition-averaged PSTHs plus 500 shuffles per session); none needs ONE or brainbox.

  1. WILD-TYPE CONTROL. The published analysis skips opsin-negative animals outright
     (`if this_dict['sert-cre'] == 0: continue`), but the stored files carry them. They
     received identical light. If they show comparable orthogonality, the result is a
     light or heating artifact. This control does not exist in the paper.

  2. UNPOOLING (R1/R2). The published statistic pools every session into one
     pseudo-population. Here it is recomputed per session and per animal, each against
     its own shuffle distribution, then reduced to a sign test across animals -- the
     unit discipline the proposal commits to.

  3. ADDITIVITY. Is the serotonin displacement the same in the two choice contexts?
     delta_L = L_opto - L_no_opto against delta_R = R_opto - R_no_opto. The talk asserts
     superposition; the paper computed an angle between two other vectors and never
     tested this. A vector identity on data already in hand.

Plus a dimensionality sweep, because the published N_DIM = 3 is the single most
questioned choice and the honest response is to show the curve rather than assert it.

THE STATISTIC, as published (Figures/figure6/manifold_analysis_per_region.py):
  splits are stacked over time, PCA to N_DIM components is **refit for every shuffle**,
  the choice and 5-HT vectors are formed by collapsing the four-way splits, and the
  reported quantity is 1 - |cos(theta)| averaged over ORTH_WIN, the 50 ms before
  movement onset. Note the alignment: the headline figure uses `firstMovement_times`,
  so this is the window immediately preceding movement, not the whole trial.

Usage:
    python3 wp1_orthogonality.py                       # auto-finds the data
    python3 wp1_orthogonality.py --ndims 3 5 10 --out results/
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

ORTH_WIN = (-0.05, 0.0)     # as published: 50 ms before the alignment event
N_DIM = 3                    # as published
MIN_NEURONS = 20             # below this a per-session projection is not meaningful
SPLITS = ("L_opto", "R_opto", "L_no_opto", "R_no_opto")


# ------------------------------------------------------------------ loading

def find_data(root: Path) -> tuple[Path, str]:
    """Locate the per-session .npy directory; prefer the main-figure alignment."""
    for name in ("firstMovement_times", "stimOn_times"):
        for cand in root.rglob(name):
            if cand.is_dir() and list(cand.glob("*.npy")):
                return cand, name
    raise SystemExit(f"no manifold data under {root}")


def load_sessions(data_dir: Path) -> list[dict]:
    out = []
    for p in sorted(glob(str(data_dir / "*.npy"))):
        d = np.load(p, allow_pickle=True).flat[0]
        d["_file"] = Path(p).name
        out.append(d)
    return out


# ------------------------------------------------------------------ statistic

def _collapse(pcs: np.ndarray, T: int) -> tuple[np.ndarray, ...]:
    """Split the stacked PCA scores back into the four conditions and collapse."""
    L_opto, R_opto, L_no, R_no = (pcs[i * T:(i + 1) * T] for i in range(4))
    return ((L_opto + L_no) / 2, (R_opto + R_no) / 2,      # choice: left, right
            (L_opto + R_opto) / 2, (L_no + R_no) / 2,      # 5-HT: stim, no-stim
            L_opto, R_opto, L_no, R_no)


def _one_minus_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """1 - |cos| per timepoint. 1 = orthogonal, 0 = parallel."""
    na = a / np.linalg.norm(a, axis=1, keepdims=True)
    nb = b / np.linalg.norm(b, axis=1, keepdims=True)
    return 1 - np.abs(np.sum(na * nb, axis=1))


def orthogonality(dicts: list[dict], n_dim: int = N_DIM, win=ORTH_WIN,
                  n_shuffles: int | None = None) -> dict | None:
    """The published statistic, computed over whatever set of sessions is passed.

    Returns the observed value, the shuffle null, a one-sided permutation p, and the
    additivity measures -- all from a single pass, since they share the projection.
    """
    time_ax = dicts[0]["time"]
    T = len(time_ax)
    sel = (time_ax >= win[0]) & (time_ax <= win[1])
    if sel.sum() == 0:
        return None

    real = {s: np.vstack([d[s] for d in dicts]) for s in SPLITS}
    n_neurons = real["L_opto"].shape[0]
    if n_neurons < MIN_NEURONS:
        return None
    n_shuf = min(d["n_shuffles"] for d in dicts)
    if n_shuffles:
        n_shuf = min(n_shuf, n_shuffles)

    def stat(mats: dict[str, np.ndarray]) -> tuple[float, float, float]:
        stacked = np.vstack([mats[s].T for s in SPLITS])          # (4T, n_neurons)
        pcs = PCA(n_components=min(n_dim, stacked.shape[1])).fit_transform(stacked)
        l_c, r_c, o_c, n_c, L_o, R_o, L_n, R_n = _collapse(pcs, T)
        dot = _one_minus_cos(l_c - r_c, o_c - n_c)
        # additivity: is the 5-HT displacement the same in both choice contexts?
        dL, dR = L_o - L_n, R_o - R_n
        add_ang = _one_minus_cos(dL, dR)
        mag = (np.linalg.norm(dL, axis=1) /
               np.maximum(np.linalg.norm(dR, axis=1), 1e-12))
        return (float(np.mean(dot[sel])), float(np.mean(add_ang[sel])),
                float(np.mean(np.log2(np.maximum(mag[sel], 1e-12)))))

    obs, obs_add, obs_mag = stat(real)

    null, null_add = [], []
    for i in range(n_shuf):
        try:
            sh = {s: np.vstack([d["shuffle"][s][:, :, i] for d in dicts]) for s in SPLITS}
        except (KeyError, IndexError):
            continue
        if np.isnan(sh["L_opto"]).any():
            continue
        v, a, _ = stat(sh)
        null.append(v); null_add.append(a)
    if len(null) < 20:
        return None
    null = np.array(null); null_add = np.array(null_add)

    return dict(
        n_neurons=int(n_neurons), n_sessions=len(dicts), n_shuffles=len(null),
        dot=obs, null_mean=float(null.mean()), null_p97_5=float(np.quantile(null, .975)),
        p_perm=float((null >= obs).mean()),
        significant=bool(obs > np.quantile(null, .975)),
        z=float((obs - null.mean()) / (null.std() + 1e-12)),
        add_angle=obs_add, add_null_mean=float(null_add.mean()),
        add_p_perm=float((null_add >= obs_add).mean()),
        add_log2_mag_ratio=obs_mag)


# ------------------------------------------------------------------ analyses

def sign_p(k, n):
    return binomtest(k, n, 0.5, alternative="two-sided").pvalue if n else float("nan")


def run(sessions: list[dict], out: Path, ndims: list[int]) -> dict:
    sert = [d for d in sessions if d["sert-cre"] == 1]
    wt = [d for d in sessions if d["sert-cre"] == 0]
    print(f"  {len(sessions)} session files | SERT-cre {len(sert)} | wild-type {len(wt)}")
    print(f"  animals: SERT {len({d['subject'] for d in sert})} · "
          f"WT {len({d['subject'] for d in wt})}")
    summary: dict = {}

    # ---- 1. pooled, and the wild-type control the paper omits
    print("\n[1] Pooled statistic — published (SERT-cre) and the missing WT control")
    rows = []
    for label, group in (("SERT-cre (published)", sert), ("wild-type (CONTROL)", wt)):
        if not group:
            continue
        r = orthogonality(group)
        if r is None:
            print(f"  {label}: insufficient data"); continue
        rows.append(dict(group=label, **r))
        flag = "SIGNIFICANT" if r["significant"] else "n.s."
        print(f"  {label:22} 1-|cos| = {r['dot']:.4f}  null {r['null_mean']:.4f}  "
              f"z = {r['z']:+.2f}  p = {r['p_perm']:.4f}  {flag}   "
              f"({r['n_neurons']} neurons)")
    pooled = pd.DataFrame(rows)
    pooled.to_csv(out / "wp1_pooled_and_wt.csv", index=False)
    summary["pooled"] = rows

    # ---- 2. unpooling: per session, then per animal
    print("\n[2] Unpooling — per session and per animal (SERT-cre only)")
    per_ses = []
    for d in sert:
        r = orthogonality([d])
        if r:
            per_ses.append(dict(subject=d["subject"], eid=d.get("eid"),
                                file=d["_file"], **r))
    ps = pd.DataFrame(per_ses)
    ps.to_csv(out / "wp1_per_session.csv", index=False)
    if len(ps):
        k = int(ps.significant.sum())
        print(f"  per session: {k}/{len(ps)} individually significant "
              f"({100*k/len(ps):.0f}%) | median z = {ps.z.median():+.2f}")

    per_an = []
    for subj in sorted({d["subject"] for d in sert}):
        r = orthogonality([d for d in sert if d["subject"] == subj])
        if r:
            per_an.append(dict(subject=subj, **r))
    pa = pd.DataFrame(per_an)
    pa.to_csv(out / "wp1_per_animal.csv", index=False)
    if len(pa):
        k = int((pa.z > 0).sum()); n = len(pa)
        ksig = int(pa.significant.sum())
        print(f"  per animal : {ksig}/{n} individually significant | "
              f"{k}/{n} in the predicted direction | sign test p = {sign_p(k, n):.4f} "
              f"(ceiling {sign_p(n, n):.4f})")
        summary["per_animal"] = dict(n=n, n_significant=ksig, n_positive=k,
                                     sign_p=sign_p(k, n), ceiling=sign_p(n, n),
                                     median_z=float(pa.z.median()))

    # ---- 3. additivity across choice context
    print("\n[3] Additivity — is the 5-HT displacement the same for left and right choices?")
    if len(pa):
        k = int((pa.add_p_perm < 0.05).sum())
        print(f"  per animal: mean 1-|cos(dL,dR)| = {pa.add_angle.mean():.4f} "
              f"(null {pa.add_null_mean.mean():.4f}) | "
              f"{k}/{len(pa)} animals differ from shuffle at p<0.05")
        print(f"  magnitude ratio |dL|/|dR|: median {2**pa.add_log2_mag_ratio.median():.2f}x")
        print("  (1-|cos| near 0 = parallel = additive; near 1 = context-dependent)")
        summary["additivity"] = dict(mean_angle=float(pa.add_angle.mean()),
                                     mean_null=float(pa.add_null_mean.mean()),
                                     n_differing=k, n=len(pa))

    # ---- 4. dimensionality sweep
    print(f"\n[4] Dimensionality sweep — pooled SERT-cre, N_DIM in {ndims}")
    sweep = []
    for nd in ndims:
        r = orthogonality(sert, n_dim=nd, n_shuffles=200)
        if r:
            sweep.append(dict(n_dim=nd, **r))
            print(f"  N_DIM={nd:>3}: 1-|cos| = {r['dot']:.4f}  null {r['null_mean']:.4f}  "
                  f"z = {r['z']:+.2f}  p = {r['p_perm']:.4f}")
    pd.DataFrame(sweep).to_csv(out / "wp1_ndim_sweep.csv", index=False)
    summary["ndim_sweep"] = sweep
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=str(Path.home() /
                    "Projects/mainenlab/serotonin-analysis/Data"))
    ap.add_argument("--ndims", type=int, nargs="+", default=[3, 5, 10, 20])
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    data_dir, align = find_data(Path(args.data_root))
    print(f"Data: {data_dir}  (aligned to {align})")
    sessions = load_sessions(data_dir)
    if not sessions:
        print("no sessions loaded", file=sys.stderr); return 1

    summary = run(sessions, out, args.ndims)
    (out / "wp1_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nWritten to {out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
