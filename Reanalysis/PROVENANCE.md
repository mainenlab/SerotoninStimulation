# Provenance — where everything lives, and what it came from

Written 2026-09-13. Revised 2026-09-14. Update this file whenever data is fetched or the
upstream moves.

## The three locations, and why they are separate

| What | Where | Committed? |
|:--|:--|:--|
| **Our analysis code and results** | `Reanalysis/` (this directory) | Yes — branch `reanalysis/5ht-subspace` |
| **Our write-ups** | `mainenlab/5HT-programme`, in `results/` | Yes — in that repo |
| **Upstream code and large data** | the rest of this clone | **No — leave upstream files alone** |

⚠️ **Corrected 2026-09-14.** The first two rows were wrong. Our code was never in HaaK, and
there has never been a `5HT-talk/analysis/` directory in either copy of that project — the
write-ups in `results/` linked to a path that did not exist. This directory was also entirely
**untracked** until 2026-09-14, so the WP1/WP2/WP6 work had no history and no second copy
anywhere. It is now committed and pushed.

`origin` is `mainenlab/SerotoninStimulation`, **our own fork**, not Guido's repository. That
is the change that makes committing here correct rather than rude: the rule below — a branch
on a fork, never a commit into someone else's clone — is satisfied by `reanalysis/5ht-subspace`
on the fork. Upstream files still get left alone; only `Reanalysis/` and our own new scripts
under `Preprocess/` are ours to commit. Downloaded data lives here because that is where the
scripts look, and the upstream `.gitignore` already covers those paths.

## Upstream

- Upstream: <https://github.com/guidomeijer/SerotoninStimulation>
- Our fork (`origin` here): <https://github.com/mainenlab/SerotoninStimulation>
- Clone: `~/Projects/mainenlab/serotonin-analysis`, branch `reanalysis/5ht-subspace`
- **Upstream pinned at commit `db29cd1`** — re-pin here if the clone is updated, since
  analyses run against its `stim_functions.py` loaders.
- Upstream files are left untouched. One untracked file, `Figures/figure6/run_glmm.py`,
  predates our work and is not ours — leave it untracked.

## Data

| Dataset | Path | Size | Source |
|:--|:--|:--|:--|
| Trial table | `Data/all_trials.csv` | 9 MB | ships with the repo |
| Per-neuron ROCs | `Data/task_modulated_neurons.csv` | 1 MB | ships with the repo |
| Passive face/pupil | `Data/{whisking,sniffing,pupil}_passive.csv` | ~800 KB | ships with the repo |
| **Manifold PSTHs, movement-aligned** | `Data/firstMovement_times/` | **1.6 GB, 38 sessions** | figshare file `57359308`, fetched 2026-09-13 |
| Manifold PSTHs, stimulus-aligned | `Data/stimOn_times/` | not yet fetched | figshare file `57359311` |

Both manifold paths are covered by the upstream `.gitignore` (lines 12–13), so they do not
dirty the clone. They are **not** in HaaK either — 1.6 GB does not belong in a git repo.
Re-fetch with the command below rather than backing them up.

### The download URL in the published scripts does not work

The repo fetches from `https://figshare.com/ndownloader/files/<id>`, which returns **HTTP 202
with an empty body, indefinitely** — 50 polls over 21 minutes, both file ids, never any data.
A browser User-Agent makes it worse (403). Anyone running `manifold_analysis_all_together.py`
from a clean checkout gets a hang, not a figure.

The working host is **`ndownloader.figshare.com`**, which the figshare API redirects to:

```bash
curl -L -o f.zip "https://ndownloader.figshare.com/files/57359308"   # movement-aligned
unzip f.zip -d ~/Projects/mainenlab/serotonin-analysis/Data/
```

**This is worth reporting upstream — it is a two-character fix and it blocks reproduction.**
Include it in the eventual pull request.

### Which alignment is which

- `57359308` → `Data/firstMovement_times/` → used by `Figures/figure6/` — **the main
  orthogonality result**. Time axis −0.294 to −0.006 s, 24 bins of 12.5 ms, entirely
  pre-movement.
- `57359311` → `Data/stimOn_times/` → used by `Figures/supp_figure_manifold/` — the
  supplementary version. Since the laser fires at the go cue and `goCue − stimOn` is 16 ms,
  **this is effectively the laser-locked alignment** and is the right one for anything about
  the serotonin response itself.

## Environment

Base conda environment provides `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`
and `ssm` (Linderman, for the GLM-HMM). Sufficient for WP1, WP2 and WP6.

`ONE-api` and `ibllib` are **not** installed and are required for anything touching raw
spikes, video, or real session identifiers — which is WP7 face extraction, the null swap,
and WP6's session ids. ONE resolves to the public Open Alyx instance
(`https://openalyx.internationalbrainlab.org`, password `international`), so no account is
needed; `stim_functions.init_one()` handles it.

## Layout, and the eventual pull request

Our scripts are named to mirror upstream conventions so that contributing them back is a
copy rather than a rewrite:

| Ours | Upstream home in a future PR |
|:--|:--|
| `wp1_orthogonality.py` | `Figures/figure6/` — per-session and per-animal variants |
| `wp2_condition_dependence.py` | `Figures/figure6/` |
| `wp6_glmhmm.py` | `Figures/figure5/` or a new behavioural figure |
| `../Preprocess/run_face_opto_task.py` | already in place, mirroring the existing `_passive` scripts |

**On the fork, not yet a PR.** Our work lives on `reanalysis/5ht-subspace` on
`mainenlab/SerotoninStimulation`, which is the branch-on-a-fork route this section always
prescribed — a pull request to Guido is then a branch diff, opened when the analyses are
ready to stand. The URL fix above should go in that PR regardless of whether the analyses do.

## Reproducing

```bash
conda activate base
cd <this directory>
python3 wp6_glmhmm.py --k 3 --folds 3 --out results     # behaviour only, no download needed
python3 wp1_orthogonality.py --out results              # needs Data/firstMovement_times/
python3 wp2_condition_dependence.py --out results       # same
python3 plot_wp6.py                                     # figures
```

All scripts take `--data-root` and default to the clone path above.
