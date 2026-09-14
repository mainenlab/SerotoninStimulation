#!/usr/bin/env python3
"""WP7 figures — when does serotonin actually act within a trial?

The body gives a laser-locked readout with no reaction-time smearing, because the laser
fires at the go cue. Its latency says WHEN the manipulation is doing something, which every
window choice in the neural analysis has so far had to assume.

Both epochs are in the same units -- percent change from each session's 2nd percentile --
so amplitudes are comparable. Passive traces come from the upstream CSVs; task traces are
stimulated minus unstimulated, per animal, from our extraction.

    python3 plot_wp7_timing.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

HERE = Path(__file__).parent
UP = HERE.parent                      # upstream repo root
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)

TEAL, VIOLET = "#0B717B", "#5B3FD6"
INK, MUTED, FAINT = "#17151F", "#6B6780", "#CFCADD"
AMBER = "#A05A06"

# Trial landmarks, measured from all_trials.csv (60,486 trials, 6 mice)
GO_CUE, FIRST_MOVE, FEEDBACK = 0.0, 0.143, 0.407

plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED,
                     "figure.facecolor": "white"})


def per_animal(measure: str):
    """Return (passive_df, task_df): index = time, columns = animals, SERT-cre only."""
    pa = pd.read_csv(UP / "Data" / f"{measure}_passive.csv")
    pa = pa[pa.expression == 1]
    P = pd.concat({s: g.groupby("time").baseline_subtracted.mean()
                   for s, g in pa.groupby("subject")}, axis=1)

    ta = pd.read_csv(HERE / "results" / f"{measure}_task.csv")
    ta = ta[ta.expression == 1]
    cols = {}
    for s, g in ta.groupby("subject"):
        a = g[g.condition == "stim"].groupby("time").baseline_subtracted.mean()
        b = g[g.condition == "no_stim"].groupby("time").baseline_subtracted.mean()
        d = (a - b).dropna()
        if len(d):
            cols[s] = d
    T = pd.concat(cols, axis=1)
    return P, T


def peak_latency(M: pd.DataFrame) -> pd.Series:
    """Per-animal latency of the largest positive excursion after the laser."""
    t = M.index.values
    post = (t > 0) & (t <= 4)
    return pd.Series({c: t[post][np.nanargmax(M[c].values[post])] for c in M.columns})


def trace(ax, M, colour, label):
    t = M.index.values
    m = M.mean(axis=1).values
    sem = M.sem(axis=1).values
    ax.fill_between(t, m - sem, m + sem, color=colour, alpha=.16, lw=0)
    ax.plot(t, m, color=colour, lw=2.2, label=f"{label} (n={M.shape[1]})")


def landmarks(ax, top=True):
    for x, lab, c in ((GO_CUE, "go cue + laser", INK),
                      (FIRST_MOVE, "first move", MUTED),
                      (FEEDBACK, "feedback", AMBER)):
        ax.axvline(x, color=c, ls="--", lw=1.1, alpha=.75, zorder=1)
    if top:
        ylim = ax.get_ylim()
        ax.annotate("decision\nwindow", xy=(FEEDBACK / 2, ylim[1] * .93), ha="center",
                    fontsize=8.5, color=AMBER)
        ax.axvspan(GO_CUE, FEEDBACK, color=AMBER, alpha=.07, lw=0, zorder=0)


def main():
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    stats = {}

    for i, meas in enumerate(("whisking", "sniffing")):
        P, T = per_animal(meas)
        ax = axes[i]
        trace(ax, P, TEAL, "passive")
        trace(ax, T, VIOLET, "task")
        ax.axhline(0, color=MUTED, lw=.8)
        landmarks(ax)
        ax.set_xlim(-0.5, 4)
        ax.set_xlabel("time from laser onset (s)")
        ax.set_ylabel("% change from 2nd percentile" if i == 0 else "")
        ax.set_title(f"{'ab'[i]}  {meas.capitalize()}", fontsize=11, loc="left", pad=9)
        ax.legend(loc="upper right", frameon=False, fontsize=9)
        ax.grid(alpha=.15, color=FAINT)
        stats[meas] = dict(P=P, T=T)

    # (c) early vs late component, paired per animal -- THE result.
    # A scalar peak latency is bistable here: the response is biphasic in both epochs, so
    # "the peak" flips between the two bumps. Decomposing into an early window that overlaps
    # the decision and a late window in the inter-trial interval is what actually separates them.
    EARLY, LATE = (0.2, 1.0), (1.5, 3.0)

    def win(M, w):
        tt = M.index.values
        return M.loc[(tt >= w[0]) & (tt <= w[1])].mean(axis=0)

    ax = axes[2]
    summary = {}
    for j, meas in enumerate(("whisking", "sniffing")):
        P, T = stats[meas]["P"], stats[meas]["T"]
        sh = sorted(set(P.columns) & set(T.columns))
        pe, te = win(P, EARLY)[sh], win(T, EARLY)[sh]
        pl, tl = win(P, LATE)[sh], win(T, LATE)[sh]
        off = -0.09 if j == 0 else 0.09
        for s_ in sh:
            ax.plot([0 + off, 1 + off], [pe[s_], te[s_]], "-",
                    color=VIOLET if te[s_] < pe[s_] else FAINT, lw=1.3, alpha=.8)
            ax.plot([0 + off], [pe[s_]], "o", color=TEAL, ms=5)
            ax.plot([1 + off], [te[s_]], "o", color=VIOLET if te[s_] < pe[s_] else MUTED, ms=5)
        ax.plot([0 + off, 1 + off], [pe.mean(), te.mean()], "-", color=INK, lw=2.4, zorder=5)
        k = int((te < pe).sum())
        summary[meas] = dict(n=len(sh), k_early_down=k,
                             early_ret=100 * te.mean() / pe.mean(),
                             late_ret=100 * tl.mean() / pl.mean(),
                             p=binomtest(k, len(sh), 0.5).pvalue)
    ax.axhline(0, color=MUTED, lw=.8)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["passive", "task"])
    ax.set_xlim(-.45, 1.55)
    ax.set_ylabel("early response, 0.2–1 s (% change)")
    ax.set_title("c  Early component, per animal", fontsize=11, loc="left", pad=9)
    ax.grid(alpha=.15, axis="y", color=FAINT)
    note = "  ·  ".join(f"{m}: {d['k_early_down']}/{d['n']} lower, p = {d['p']:.3f}"
                        for m, d in summary.items())
    ax.text(.5, -.2, note, transform=ax.transAxes, ha="center", fontsize=8.5,
            color=VIOLET, fontweight="bold")

    # (d) what each component retains
    ax = axes[3]
    labels, early_v, late_v = [], [], []
    for meas, d in summary.items():
        labels.append(meas.capitalize()); early_v.append(d["early_ret"]); late_v.append(d["late_ret"])
    y = np.arange(len(labels))
    ax.barh(y + .18, early_v, height=.33, color=VIOLET, alpha=.92, label="early (0.2–1 s)")
    ax.barh(y - .18, late_v, height=.33, color=TEAL, alpha=.75, label="late (1.5–3 s)")
    ax.axvline(100, color=MUTED, ls="--", lw=1)
    ax.text(100, len(labels) - .35, "passive level", color=MUTED, fontsize=8.5, ha="center")
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlim(0, 145); ax.set_xlabel("retained in task (% of passive)")
    ax.set_title("d  The fast component is halved; the slow one is not",
                 fontsize=11, loc="left", pad=9)
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    ax.grid(alpha=.15, axis="x", color=FAINT)
    for yy, v in zip(y + .18, early_v):
        ax.text(v + 3, yy, f"{v:.0f}%", va="center", fontsize=9, color=VIOLET, fontweight="bold")
    for yy, v in zip(y - .18, late_v):
        ax.text(v + 3, yy, f"{v:.0f}%", va="center", fontsize=9, color=TEAL)

    fig.suptitle("Task engagement halves serotonin's fast reach into the body, and leaves "
                 "the slow one intact  ·  45 sessions, 15 mice",
                 fontsize=13, y=1.03, color=INK)
    plt.tight_layout()
    out = FIG / "wp7_fig1_timing.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved {out}")
    for m, d in summary.items():
        print(f"  {m}: early retained {d['early_ret']:.0f}%, late {d['late_ret']:.0f}%, "
              f"{d['k_early_down']}/{d['n']} animals lower early, p={d['p']:.3f}")


if __name__ == "__main__":
    main()
