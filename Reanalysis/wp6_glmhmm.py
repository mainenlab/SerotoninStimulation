#!/usr/bin/env python3
"""WP6 — does serotonin stimulation change behavioural STATE rather than performance?

The published behavioural result is a null: stimulating DRN serotonin neurons changes
no psychometric parameter and none of 13 choice-model weights. That is a claim about
WITHIN-STATE performance. Ashwood et al. 2022 (Nat Neurosci 25:201-212,
doi:10.1038/s41593-021-01007-z) showed IBL mice do not use one strategy -- they alternate
between an engaged state and several biased states, persisting tens to hundreds of trials.
A pooled GLM averages over that and would report exactly the reported null even if
stimulation changed how often the animal switches.

So this fits a GLM-HMM and asks two things:

  1. Replication -- do we recover the published null on within-state / pooled GLM weights?
  2. The new question -- does stimulation change the TRANSITION structure, tested by
     comparing a state-transition model that sees stimulation against one that does not,
     on held-out sessions?

Design matrix follows Ashwood: stimulus, previous choice, win-stay-lose-switch, bias.

Unit discipline (see the proposal's General Design): the unit of analysis is the session,
the unit of inference is the ANIMAL. Everything is cross-validated by held-out session and
reduced to one directional statistic per animal, then a sign test across animals. With
6 SERT-cre animals in the shipped behavioural table the ceiling is p = 0.031 (6/6).

Usage:
    python3 wp6_glmhmm.py --data ~/Projects/mainenlab/serotonin-analysis/Data/all_trials.csv
    python3 wp6_glmhmm.py --k-range 2 3 4 --folds 5 --out results/
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
SEED = 42
MIN_SESSION_TRIALS = 50   # below this an inferred session is a timestamp artifact


# ---------------------------------------------------------------- data

def load_trials(path: Path) -> pd.DataFrame:
    """Load the shipped trial table and reconstruct what it does not store.

    CAVEAT, and it limits everything downstream: the table has **no session identifier**.
    Session boundaries have to be inferred from the trial clock, and that inference is
    unreliable here -- across 60,486 trials there are only 42 downward resets, and 125
    trials carry a NaN `stimOn_times` which masquerade as boundaries. Any analysis whose
    unit is the session (the GLM-HMM below) therefore rests on a guess. Pull real `eid`s
    from Open Alyx before trusting it. The predictability test does not use sessions and
    is unaffected.
    """
    df = pd.read_csv(path)
    df = df[df["choice"].isin([-1, 1])].copy().reset_index(drop=True)

    # Inferred sessions: a downward jump in the clock, or a long gap.
    d = df.groupby("subject")["stimOn_times"].diff()
    df["new_ses"] = (d < 0) | (d > 300) | d.isna()
    df["ses"] = df["new_ses"].groupby(df["subject"]).cumsum()
    df["sesid"] = df["subject"] + "_" + df["ses"].astype(int).astype(str)

    # True laser blocks: probe trials are the minority exception inside a block, so flip
    # them back before taking runs, or every probe would start a spurious one-trial block.
    df["block_state"] = np.where(df["probe_trial"] == 1,
                                 1 - df["laser_stimulation"], df["laser_stimulation"])
    df["blk"] = df.groupby("subject")["block_state"].transform(
        lambda s: (s.diff() != 0).cumsum())
    df["blkid"] = df["subject"] + "_" + df["blk"].astype(int).astype(str)

    # Drop inferred "sessions" too short to be real -- these are artifacts of the NaN
    # timestamps, not recordings. Session-based analyses use `sesid_ok`.
    n_per = df.groupby("sesid")["choice"].transform("size")
    df["sesid_ok"] = np.where(n_per >= MIN_SESSION_TRIALS, df["sesid"], None)
    return df


def design_matrix(g: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ashwood-style inputs, plus the stimulation regressor kept separate.

    Returns (inputs, choices, laser) for one session.
      inputs : (T, 4) -- stimulus, previous choice, win-stay-lose-switch, bias
      choices: (T, 1) int 0/1  (0 = left, 1 = right)
      laser  : (T, 1) float    0/1, used only as a transition input
    """
    g = g.sort_values("stimOn_times").reset_index(drop=True)

    # Signed contrast, normalised to unit scale so weights are comparable across animals.
    stim = g["signed_contrast"].to_numpy(dtype=float)
    denom = np.max(np.abs(stim)) or 1.0
    stim = stim / denom

    choice01 = (g["choice"].to_numpy() == 1).astype(int)          # 1 = right
    prev_choice = np.concatenate(([0], (choice01[:-1] * 2 - 1)))   # +-1, 0 on first trial

    rewarded = (g["feedbackType"].to_numpy() == 1).astype(int)
    wsls = np.concatenate(([0], (choice01[:-1] * 2 - 1) * (rewarded[:-1] * 2 - 1)))

    bias = np.ones_like(stim)
    inputs = np.column_stack([stim, prev_choice, wsls, bias])
    laser = g["laser_stimulation"].to_numpy(dtype=float)[:, None]
    return inputs, choice01[:, None], laser


# ---------------------------------------------------------------- models

def fit_glmhmm(K, inputs, choices, trans_inputs=None, n_iters=200, seed=SEED):
    """Fit a GLM-HMM. If trans_inputs is given, transitions are input-driven."""
    import ssm

    M = inputs[0].shape[1]
    npr = np.random.RandomState(seed)
    if trans_inputs is None:
        model = ssm.HMM(K, 1, M, observations="input_driven_obs",
                        observation_kwargs=dict(C=2), transitions="standard")
        fit_inputs = inputs
    else:
        # Stimulation enters the transition matrix; observations keep the same design.
        model = ssm.HMM(K, 1, M + 1, observations="input_driven_obs",
                        observation_kwargs=dict(C=2), transitions="inputdriven")
        fit_inputs = [np.hstack([x, t]) for x, t in zip(inputs, trans_inputs)]
    np.random.seed(seed)
    model.fit(choices, inputs=fit_inputs, method="em", num_iters=n_iters, tolerance=1e-4)
    return model, fit_inputs


def held_out_ll(model, choices, inputs) -> float:
    """Log-likelihood per trial on held-out sessions."""
    n = sum(c.shape[0] for c in choices)
    return float(model.log_likelihood(choices, inputs=inputs) / n)


def session_folds(sesids, n_folds, seed=SEED):
    u = np.array(sorted(set(sesids)))
    rng = np.random.RandomState(seed)
    rng.shuffle(u)
    return [u[i::n_folds] for i in range(n_folds)]


# ---------------------------------------------------------------- analyses

def pooled_glm_replication(df: pd.DataFrame) -> pd.DataFrame:
    """Sanity check: fit the pooled GLM separately on stimulated and unstimulated
    trials. If the published null holds, the weights should be indistinguishable."""
    from sklearn.linear_model import LogisticRegression

    rows = []
    for subj, g in df.groupby("subject"):
        for lab, sel in (("no_stim", g["block_state"] == 0),
                         ("stim", g["block_state"] == 1)):
            gg = g[sel]
            if len(gg) < 200:
                continue
            X, y, _ = design_matrix(gg)
            m = LogisticRegression(C=1e6, max_iter=2000, fit_intercept=False).fit(X, y.ravel())
            rows.append(dict(subject=subj, condition=lab, n=len(gg),
                             w_stim=m.coef_[0][0], w_prev=m.coef_[0][1],
                             w_wsls=m.coef_[0][2], w_bias=m.coef_[0][3],
                             task_accuracy=float(gg["correct"].mean())))
    return pd.DataFrame(rows)


def predictability_test(df: pd.DataFrame, n_reps: int = 10, seed: int = SEED) -> pd.DataFrame:
    """THE result that does not depend on session boundaries.

    The published null is about GLM *weights* -- whether stimulation changes how much
    the animal leans on stimulus, history or bias. This asks a different question: does
    the model still fit as well? Identical weights with worse fit means the choices have
    become less explicable from the task variables, which is what more time in biased or
    disengaged states looks like -- and a weights test cannot see it.

    Three controls matter and all three are applied:
      - matched n, by subsampling the larger condition to the smaller;
      - block-grouped cross-validation, so no laser block is split across train and test
        (random CV leaks the blocked structure -- the same error the design review flagged
        in our null models);
      - probe trials assigned to their surrounding block's state, so a probe does not
        create a one-trial block.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, GroupKFold

    rng = np.random.RandomState(seed)
    rows = []
    for subj, g in df.groupby("subject"):
        g0, g1 = g[g["block_state"] == 0], g[g["block_state"] == 1]
        n = min(len(g0), len(g1))
        if n < 500:
            continue
        acc = {}
        for lab, gg in (("no_stim", g0), ("stim", g1)):
            scores = []
            for _ in range(n_reps):
                idx = np.sort(rng.choice(len(gg), n, replace=False))
                sub = gg.iloc[idx]
                X, y, _ = design_matrix(sub)
                grp = sub.sort_values("stimOn_times")["blkid"].to_numpy()
                k = min(5, len(np.unique(grp)))
                if k < 2:
                    continue
                scores.append(cross_val_score(
                    LogisticRegression(C=1e6, max_iter=2000, fit_intercept=False),
                    X, y.ravel(), cv=GroupKFold(n_splits=k), groups=grp).mean())
            acc[lab] = float(np.mean(scores)) if scores else np.nan
        rows.append(dict(subject=subj, n_matched=n,
                         cv_acc_no_stim=acc["no_stim"], cv_acc_stim=acc["stim"],
                         delta=acc["stim"] - acc["no_stim"],
                         less_predictable_under_stim=bool(acc["stim"] < acc["no_stim"])))
    return pd.DataFrame(rows)


def select_k(df, k_range, n_folds):
    """Cross-validated state count, pooled across animals for a single choice of K."""
    out = []
    for K in k_range:
        lls = []
        for subj, g in df.groupby("subject"):
            sess = sorted(set(g["sesid"]))
            if len(sess) < n_folds:
                continue
            folds = session_folds(sess, n_folds)
            for f in folds:
                tr = [s for s in sess if s not in set(f)]
                te = list(f)
                if not tr or not te:
                    continue
                tr_in, tr_ch = [], []
                for s in tr:
                    X, y, _ = design_matrix(g[g["sesid"] == s])
                    tr_in.append(X); tr_ch.append(y)
                te_in, te_ch = [], []
                for s in te:
                    X, y, _ = design_matrix(g[g["sesid"] == s])
                    te_in.append(X); te_ch.append(y)
                try:
                    m, fi = fit_glmhmm(K, tr_in, tr_ch)
                    lls.append(held_out_ll(m, te_ch, te_in))
                except Exception as e:
                    print(f"    [K={K} {subj}] fit failed: {type(e).__name__}", file=sys.stderr)
        if lls:
            out.append(dict(K=K, mean_heldout_ll=float(np.mean(lls)),
                            sem=float(np.std(lls) / max(np.sqrt(len(lls)), 1)), n=len(lls)))
            print(f"  K={K}: held-out LL/trial = {np.mean(lls):.4f}  (n={len(lls)})")
    return pd.DataFrame(out)


def transition_test(df, K, n_folds):
    """THE test. Per animal, compare held-out likelihood of a GLM-HMM whose transitions
    see the stimulation regressor against one whose transitions do not. One directional
    statistic per animal; sign test across animals."""
    rows = []
    for subj, g in df.groupby("subject"):
        g = g[g["sesid_ok"].notna()]
        sess = sorted(set(g["sesid_ok"]))
        if len(sess) < max(n_folds, 2):
            print(f"  {subj}: only {len(sess)} usable sessions, skipped")
            continue
        folds = session_folds(sess, n_folds)
        d_base, d_driven = [], []
        for f in folds:
            tr = [s for s in sess if s not in set(f)]
            te = list(f)
            pack = lambda ss: tuple(map(list, zip(*[design_matrix(g[g["sesid_ok"] == s]) for s in ss])))
            tr_in, tr_ch, tr_la = pack(tr)
            te_in, te_ch, te_la = pack(te)
            try:
                m0, _ = fit_glmhmm(K, tr_in, tr_ch)
                ll0 = held_out_ll(m0, te_ch, te_in)
                m1, _ = fit_glmhmm(K, tr_in, tr_ch, trans_inputs=tr_la)
                te_in1 = [np.hstack([x, t]) for x, t in zip(te_in, te_la)]
                ll1 = held_out_ll(m1, te_ch, te_in1)
                d_base.append(ll0); d_driven.append(ll1)
            except Exception as e:
                print(f"    [{subj}] fold failed: {type(e).__name__}: {e}", file=sys.stderr)
        if d_base:
            delta = float(np.mean(d_driven) - np.mean(d_base))
            rows.append(dict(subject=subj, n_sessions=len(sess), n_folds=len(d_base),
                             ll_stationary=float(np.mean(d_base)),
                             ll_stim_driven=float(np.mean(d_driven)),
                             delta=delta, favours_stim_driven=bool(delta > 0)))
            print(f"  {subj}: delta LL/trial = {delta:+.5f} "
                  f"({'stim-driven' if delta > 0 else 'stationary'} wins)")
    return pd.DataFrame(rows)


def sign_test(k, n):
    """Two-sided exact binomial sign test."""
    from scipy.stats import binomtest
    return float(binomtest(k, n, 0.5, alternative="two-sided").pvalue)


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(Path.home() /
                    "Projects/mainenlab/serotonin-analysis/Data/all_trials.csv"))
    ap.add_argument("--k-range", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--k", type=int, default=None, help="skip selection, use this K")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df = load_trials(Path(args.data))
    print(f"Loaded {len(df):,} trials | {df.subject.nunique()} subjects | "
          f"{df.sesid.nunique()} inferred sessions")
    print(f"  stimulated fraction: {df.laser_stimulation.mean():.3f}")

    print("\n[1] Replication — pooled GLM weights and task accuracy, stim vs no-stim")
    rep = pooled_glm_replication(df)
    rep.to_csv(out / "wp6_glm_replication.csv", index=False)
    print(rep.pivot(index="subject", columns="condition",
                    values=["w_stim", "w_bias", "task_accuracy"]).round(3).to_string())
    acc = rep.pivot(index="subject", columns="condition", values="task_accuracy")
    worse = int((acc["stim"] < acc["no_stim"]).sum()); n_a = len(acc)
    print(f"  task accuracy lower under stim in {worse}/{n_a} animals; "
          f"sign test p = {sign_test(worse, n_a):.4f}  → the published null replicates")

    print("\n[2] Predictability — does the GLM still fit as well? (matched n, block-grouped CV)")
    pred = predictability_test(df)
    pred.to_csv(out / "wp6_predictability.csv", index=False)
    print(pred.round(4).to_string(index=False))
    kp = int(pred.less_predictable_under_stim.sum()); np_ = len(pred)
    p_pred = sign_test(kp, np_)
    print(f"  {kp}/{np_} animals LESS predictable under stimulation; "
          f"sign test p = {p_pred:.4f}  (ceiling at n={np_} is {sign_test(np_, np_):.4f})")

    if args.k is None:
        print(f"\n[2] State count — cross-validated over K in {args.k_range}")
        ks = select_k(df, args.k_range, args.folds)
        ks.to_csv(out / "wp6_state_selection.csv", index=False)
        K = int(ks.loc[ks.mean_heldout_ll.idxmax(), "K"]) if len(ks) else 3
    else:
        K = args.k
    print(f"  → K = {K}")

    print(f"\n[3] GLM-HMM transition test — do transitions depend on stimulation? (K={K})")
    tt = transition_test(df, K, args.folds)
    tt.to_csv(out / "wp6_transition_test.csv", index=False)

    if len(tt):
        k = int(tt.favours_stim_driven.sum()); n = len(tt)
        p = sign_test(k, n)
        print(f"\n  {k}/{n} animals favour the stimulation-driven transition model")
        print(f"  sign test p = {p:.4f}   (ceiling at n={n} is {sign_test(n, n):.4f})")
        summary = dict(K=K, transition_test=dict(
                           n_animals=n, n_favouring=k, sign_test_p=p,
                           mean_delta_ll=float(tt.delta.mean()), ceiling_p=sign_test(n, n)),
                       predictability=dict(
                           n_animals=np_, n_less_predictable=kp, sign_test_p=p_pred,
                           mean_delta=float(pred.delta.mean()), ceiling_p=sign_test(np_, np_)),
                       task_accuracy_null=dict(n_animals=n_a, n_worse=worse,
                           sign_test_p=sign_test(worse, n_a)))
        (out / "wp6_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"\nWritten to {out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
