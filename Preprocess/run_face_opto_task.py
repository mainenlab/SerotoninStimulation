#!/usr/bin/env python3
"""Extract laser-locked whisking, sniffing and pupil during the TASK.

Mirrors the existing `run_{whisking,sniffing,pupil}_opto_passive.py` scripts, but aligned
to the task's go cue instead of the passive stimulation trains. The passive and task epochs
are blocks within the SAME recording -- all 29 task eids are also passive eids, and all 44
task insertions are passive insertions -- so this is the same video, the same animals and
the same loaders, read over a different event table.

Why this is worth having. The laser fires at the go cue on stimulated trials, so a facial
response is trial-locked with no reaction-time smearing. Its latency and time course say
WHEN serotonin is actually acting within a trial, which every window choice in the neural
analysis has so far had to guess at. And the passive response is already characterised in
the shipped CSVs, giving a within-animal reference:

    passive, SERT-cre (n=10):  whisking peak +115 at 0.60 s, onset ~0.40 s, 9/10 animals
                               sniffing peak +219 at 0.70 s, onset ~0.40 s, 8/10 animals
                               pupil -- not a clean laser-locked readout (3/10)
    passive, wild-type (n=6):  flat and unsigned in all three

So the task question is simply: is it there, and is it bigger, smaller or the same?

Outputs one tidy CSV per measure, matching the passive files' columns so the two can be
concatenated directly:  [value, baseline_subtracted, time, subject, expression, eid]

Usage:
    conda activate ibl
    python3 run_face_opto_task.py --out ../Reanalysis/results
    python3 run_face_opto_task.py --limit 3        # smoke test on three sessions
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stim_functions import (init_one, load_subjects, load_trials, query_ephys_sessions,
                            get_dlc_XYs, get_pupil_diameter, smooth_interpolate_signal_sg)

T_BEFORE, T_AFTER = 0.5, 4.0     # matches the passive files' window
BIN = 0.1
BINS = np.arange(-T_BEFORE, T_AFTER + BIN, BIN)


def measures_for(one, eid: str) -> tuple[np.ndarray, dict[str, np.ndarray]] | None:
    """The three measures, defined exactly as the passive scripts define them.

    whisking : leftCamera.ROIMotionEnergy -- the whisker-pad ROI. NOT derivable from DLC,
               which carries no whisker keypoints (nose_tip, paws, pupil, tongue, tube only).
    sniffing : frame-to-frame displacement of nose_tip.
    pupil    : diameter from the four pupil keypoints.

    All three are then converted to PERCENT CHANGE FROM THE SESSION'S 2ND PERCENTILE,
    exactly as the passive scripts do:

        perc = ((x - p2) / p2) * 100

    This matters more than it looks. The denominator is computed over the WHOLE session,
    which contains both the task and the passive epochs, so it is a shared constant --
    which is precisely what makes the two epochs comparable in the same units. Omitting it
    (as a first pass here did) leaves task values in raw motion-energy units and passive
    values in percent, and any ratio between them is meaningless.
    """
    video_times, XYs = get_dlc_XYs(one, eid, view='left')
    if video_times is None or XYs is None:
        return None
    out: dict[str, np.ndarray] = {}

    def as_percent(x: np.ndarray) -> np.ndarray:
        """Percent change from the session's 2nd percentile -- the passive scripts' scaling."""
        good = x[~np.isnan(x)]
        if good.size == 0:
            return x
        p2 = np.percentile(good, 2)
        if not np.isfinite(p2) or p2 == 0:
            return x
        return ((x - p2) / p2) * 100

    try:
        roi = one.load_dataset(eid, dataset='leftCamera.ROIMotionEnergy.npy')
        out['whisking'] = as_percent(smooth_interpolate_signal_sg(roi))
    except Exception:
        pass
    try:
        d = np.sqrt(np.sum(np.diff(XYs['nose_tip'], axis=0) ** 2, axis=1))
        out['sniffing'] = as_percent(smooth_interpolate_signal_sg(d))
    except Exception:
        pass
    try:
        out['pupil'] = as_percent(smooth_interpolate_signal_sg(get_pupil_diameter(XYs)))
    except Exception:
        pass
    if not out:
        return None
    # Trim each measure and the clock to a common length, as the passive scripts do.
    n = min([len(video_times)] + [len(v) for v in out.values()])
    return video_times[:n], {k: v[:n] for k, v in out.items()}


def peri_event(signal: np.ndarray, times: np.ndarray, events: np.ndarray) -> np.ndarray:
    """Mean signal in each bin relative to each event; returns (n_events, n_bins)."""
    out = np.full((len(events), len(BINS) - 1), np.nan)
    for i, ev in enumerate(events):
        rel = times - ev
        m = (rel >= -T_BEFORE) & (rel < T_AFTER)
        if m.sum() < 5:
            continue
        idx = np.digitize(rel[m], BINS) - 1
        ok = (idx >= 0) & (idx < len(BINS) - 1)
        sig = signal[m][ok]
        out[i] = np.bincount(idx[ok], weights=sig, minlength=len(BINS) - 1) / np.maximum(
            np.bincount(idx[ok], minlength=len(BINS) - 1), 1)
    return out


def process(eid: str, one, sert: int) -> list[dict]:
    trials = load_trials(eid, laser_stimulation=True, one=one)
    if trials is None or "laser_stimulation" not in trials:
        return []
    # The laser fires at the go cue on stimulated trials.
    events = trials.loc[trials["laser_stimulation"] == 1, "goCue_times"].dropna().to_numpy()
    ctrl = trials.loc[trials["laser_stimulation"] == 0, "goCue_times"].dropna().to_numpy()
    if len(events) < 20:
        return []

    got = measures_for(one, eid)
    if got is None:
        return []
    ts, measures = got

    rows = []
    centres = BINS[:-1] + BIN / 2
    for name, sig in measures.items():
        sig, tt = np.asarray(sig, float), np.asarray(ts, float)
        for cond, evs in (("stim", events), ("no_stim", ctrl)):
            if len(evs) < 20:
                continue
            pe = peri_event(sig, tt, evs)
            mean = np.nanmean(pe, axis=0)
            base = np.nanmean(mean[centres < 0])
            for t, v in zip(centres, mean):
                rows.append(dict(measure=name, condition=cond, time=float(t),
                                 value=float(v), baseline_subtracted=float(v - base),
                                 eid=eid, expression=int(sert), n_events=int(len(evs))))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../Reanalysis/results")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    one = init_one()
    subjects = load_subjects()
    rec = query_ephys_sessions(one=one)
    eids = list(dict.fromkeys(rec["eid"].tolist()))
    if args.limit:
        eids = eids[: args.limit]
    print(f"{len(eids)} ephys sessions to process")

    all_rows = []
    for i, eid in enumerate(eids, 1):
        subj = rec.loc[rec["eid"] == eid, "subject"].iloc[0]
        if subj not in subjects["subject"].values:
            continue
        sert = int(subjects.loc[subjects["subject"] == subj, "sert-cre"].iloc[0])
        try:
            rows = process(eid, one, sert)
        except Exception as e:
            print(f"  [{i}/{len(eids)}] {subj} FAILED {type(e).__name__}: {e}")
            continue
        for r in rows:
            r["subject"] = subj
        all_rows += rows
        got = sorted({r["measure"] for r in rows})
        print(f"  [{i}/{len(eids)}] {subj} sert={sert} -> {got if got else 'no video'}")

    if not all_rows:
        print("nothing extracted", file=sys.stderr); return 1
    df = pd.DataFrame(all_rows)
    for m, g in df.groupby("measure"):
        p = out / f"{m}_task.csv"
        g.to_csv(p, index=False)
        print(f"wrote {p}  ({g.eid.nunique()} sessions, {g.subject.nunique()} animals)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
