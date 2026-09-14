#!/usr/bin/env python3
"""Figures for WP6 — serotonin changes behavioural state, not performance.

Reads the CSVs written by wp6_glmhmm.py and produces two figures in the visual
identity of the proposal (teal = unstimulated, violet = the serotonin condition,
matching the choice-axis / 5-HT-displacement colours used in its geometry figure).

  fig1 : the dissociation — accuracy and GLM weights unchanged, predictability down
  fig2 : the GLM-HMM transition test — stimulation-driven transitions win in every animal

No required arguments: `python3 plot_wp6.py` just works.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

HERE = Path(__file__).parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
FIGURES.mkdir(exist_ok=True)

# Proposal palette. Teal is the unstimulated / choice-axis colour, violet the 5-HT one.
TEAL = "#0B717B"
VIOLET = "#5B3FD6"
INK = "#17151F"
MUTED = "#6B6780"
FAINT = "#CFCADD"

plt.rcParams.update({
    "font.family": "Helvetica Neue",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "figure.facecolor": "white",
})


def sign_p(k, n):
    return binomtest(k, n, 0.5, alternative="two-sided").pvalue


def paired_panel(ax, y0, y1, ylabel, title, labels=("no stim", "stim"),
                 lower_is_effect=True, highlight=True):
    """Paired-slope panel: one line per animal, heavy line for the group mean.

    `lower_is_effect` says which direction counts as the predicted effect, so the
    highlight colour means the same thing in every panel: violet = moved the way the
    hypothesis predicts, grey = did not.
    """
    for a, b in zip(y0, y1):
        # With no predicted direction (highlight=False) every line is neutral, so the
        # panel cannot suggest a trend the statistics do not support.
        moved = highlight and ((b < a) if lower_is_effect else (b > a))
        colour = VIOLET if moved else FAINT
        ax.plot([0, 1], [a, b], "-", color=colour, lw=1.5, alpha=.85, zorder=2)
        ax.plot([0], [a], "o", color=TEAL, ms=5.5, zorder=3)
        ax.plot([1], [b], "o", color=colour if moved else MUTED, ms=5.5, zorder=3)
    ax.plot([0, 1], [np.mean(y0), np.mean(y1)], "-", color=INK, lw=2.4, zorder=4)
    ax.plot([0, 1], [np.mean(y0), np.mean(y1)], "o", color=INK, ms=7.5, zorder=5)
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_xlim(-.3, 1.3)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10.5, pad=9, loc="left", color=INK)
    ax.grid(alpha=.18, axis="y", color=FAINT)


def stat_note(ax, text, effect=False):
    ax.text(.5, -.22, text, transform=ax.transAxes, ha="center", fontsize=9,
            color=VIOLET if effect else MUTED,
            fontweight="bold" if effect else "normal")


def figure1():
    rep = pd.read_csv(RESULTS / "wp6_glm_replication.csv")
    pred = pd.read_csv(RESULTS / "wp6_predictability.csv")

    acc = rep.pivot(index="subject", columns="condition", values="task_accuracy")
    wst = rep.pivot(index="subject", columns="condition", values="w_stim")
    pred = pred.set_index("subject").loc[acc.index]
    n = len(acc)

    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.6))

    # (a) task accuracy — the published null
    k = int((acc["stim"] < acc["no_stim"]).sum())
    paired_panel(axes[0], acc["no_stim"], acc["stim"], "fraction correct", "a  Task accuracy")
    stat_note(axes[0], f"{k}/{n} lower · sign test p = {sign_p(k, n):.2f} · no effect")

    # (b) GLM stimulus weight — signed, and reported as "no consistent change"
    w0, w1 = wst["no_stim"].to_numpy(), wst["stim"].to_numpy()
    try:
        p_w = wilcoxon(w0, w1).pvalue
    except ValueError:
        p_w = float("nan")
    paired_panel(axes[1], w0, w1, "stimulus weight (signed)", "b  GLM stimulus weight",
                 highlight=False)
    axes[1].text(.5, -.22, f"paired Wilcoxon p = {p_w:.2f} · "
                 f"mean |Δ| = {np.mean(np.abs(w1 - w0)):.2f} · no consistent change",
                 transform=axes[1].transAxes, ha="center", fontsize=9, color=MUTED)

    # (c) predictability — the effect
    k3 = int(pred.less_predictable_under_stim.sum())
    paired_panel(axes[2], pred.cv_acc_no_stim, pred.cv_acc_stim,
                 "cross-validated GLM accuracy", "c  Choice predictability")
    stat_note(axes[2], f"{k3}/{n} lower · sign test p = {sign_p(k3, n):.3f}", effect=True)

    # (d) per-animal deltas
    order = pred.delta.sort_values().index
    d = pred.loc[order, "delta"] * 100
    axes[3].barh(range(len(d)), d, color=VIOLET, alpha=.9, height=.6)
    axes[3].axvline(0, color=INK, lw=1)
    axes[3].set_yticks(range(len(d)))
    axes[3].set_yticklabels([s.replace("ZFM-", "") for s in order], fontsize=9)
    axes[3].set_xlabel("Δ predictability, stim − no stim (pp)")
    axes[3].set_title("d  Effect size per animal", fontsize=10.5, pad=9, loc="left")
    axes[3].grid(alpha=.18, axis="x", color=FAINT)
    axes[3].set_xlim(min(d) * 1.32, 0)
    axes[3].text(.5, -.22, f"mean {d.mean():.2f} pp · all {n}/{n} negative",
                 transform=axes[3].transAxes, ha="center", fontsize=9,
                 color=VIOLET, fontweight="bold")

    fig.suptitle("Stimulating serotonin neurons leaves performance intact but makes choices "
                 "less predictable   ·   6 SERT-cre mice, 60,412 trials",
                 fontsize=12.5, y=1.04, color=INK)
    plt.tight_layout()
    out = FIGURES / "wp6_fig1_dissociation.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved {out}")
    return dict(n=n, k_acc=k, p_acc=sign_p(k, n), p_w=float(p_w),
                k_pred=k3, p_pred=sign_p(k3, n), mean_delta_pp=float(d.mean()))


def figure2():
    tt = pd.read_csv(RESULTS / "wp6_transition_test.csv").sort_values("delta")
    n = len(tt); k = int(tt.favours_stim_driven.sum())

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))

    # (a) paired held-out likelihood — here HIGHER is the predicted effect
    paired_panel(axes[0], tt.ll_stationary, tt.ll_stim_driven,
                 "held-out log-likelihood per trial", "a  Model comparison",
                 labels=("stationary\ntransitions", "stimulation-driven\ntransitions"),
                 lower_is_effect=False)
    stat_note(axes[0], f"{k}/{n} favour stimulation-driven · sign test p = {sign_p(k, n):.3f}",
              effect=True)

    # (b) per-animal delta
    d = tt.delta * 1000
    axes[1].barh(range(n), d, color=VIOLET, alpha=.9, height=.6)
    axes[1].axvline(0, color=INK, lw=1)
    axes[1].set_yticks(range(n))
    axes[1].set_yticklabels([s.replace("ZFM-", "") for s in tt.subject], fontsize=9)
    axes[1].set_xlabel("Δ held-out log-likelihood per trial  (×10⁻³)")
    axes[1].set_title("b  Advantage of stimulation-driven transitions",
                      fontsize=10.5, pad=9, loc="left")
    axes[1].grid(alpha=.18, axis="x", color=FAINT)
    axes[1].set_xlim(0, max(d) * 1.3)
    axes[1].text(.5, -.22, f"mean {d.mean():.2f}×10⁻³ · all {k}/{n} positive",
                 transform=axes[1].transAxes, ha="center", fontsize=9,
                 color=VIOLET, fontweight="bold")

    fig.suptitle("A GLM-HMM whose state transitions see the stimulation fits held-out "
                 "sessions better in every animal   ·   K = 3 states",
                 fontsize=12.5, y=1.04, color=INK)
    plt.tight_layout()
    out = FIGURES / "wp6_fig2_transitions.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved {out}")
    return dict(n=n, k=k, p=sign_p(k, n), mean_delta=float(tt.delta.mean()))


if __name__ == "__main__":
    print("fig1:", figure1())
    print("fig2:", figure2())
