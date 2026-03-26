# -*- coding: utf-8 -*-
"""
Created on Wed May 17 12:05:36 2023

By Guido Meijer
"""

import numpy as np
from os.path import join
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
import pandas as pd
from brainbox.io.one import SpikeSortingLoader
from stim_functions import paths, load_trials, figure_style, peri_multiple_events_time_histogram
from one.api import ONE
from iblatlas.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()
fig_path, save_path = paths()
colors, dpi = figure_style()

# Settings
PID = '307f9e04-5beb-4a91-a3be-fe3b5c37ed7c'
NEURON_ID = 298
T_BEFORE = 0.5  # for plotting
T_AFTER = 1
BIN_SIZE = 0.025
SMOOTHING = 0.01

# Load in trials
trials = load_trials(one.pid2eid(PID)[0], laser_stimulation=True, one=one)

# Generate an array with 1, 2, 3, 4 for four conditions: laser_stimulation 0 or 1 and choice -1 or 1
trials['conditions'] = 0
trials.loc[(trials['laser_stimulation'] == 1) & (trials['choice'] == -1), 'conditions'] = 1
trials.loc[(trials['laser_stimulation'] == 0) & (trials['choice'] == -1), 'conditions'] = 2
trials.loc[(trials['laser_stimulation'] == 1) & (trials['choice'] == 1), 'conditions'] = 3
trials.loc[(trials['laser_stimulation'] == 0) & (trials['choice'] == 1), 'conditions'] = 4
trials = trials[(trials['conditions'] != 0) & (np.abs(trials['signed_contrast']) > 0) & (np.abs(trials['signed_contrast']) < 1)]

# Load in spikes
sl = SpikeSortingLoader(pid=PID, one=one, atlas=ba)
spikes, clusters, channels = sl.load_spike_sorting()
clusters = sl.merge_clusters(spikes, clusters, channels)

# %% Plot PSTH
p, ax = plt.subplots(1, 1, figsize=(2.3, 1.75), dpi=dpi)
peri_multiple_events_time_histogram(
    spikes.times, spikes.clusters, trials['goCue_times'], trials['conditions'],
    NEURON_ID, t_before=T_BEFORE, t_after=T_AFTER, bin_size=BIN_SIZE, ax=ax,
    pethline_kwargs=[{'color': colors['left-stim'], 'lw': 1},
                     {'color': colors['left-no-stim'], 'lw': 1},
                     {'color': colors['right-stim'], 'lw': 1},
                     {'color': colors['right-no-stim'], 'lw': 1}],
    errbar_kwargs=[{'color': colors['left-stim'], 'alpha': 0.3, 'lw': 0},
                   {'color': colors['left-no-stim'], 'alpha': 0.3, 'lw': 0},
                   {'color': colors['right-stim'], 'alpha': 0.3, 'lw': 0},
                   {'color': colors['right-no-stim'], 'alpha': 0.3, 'lw': 0}],
    raster_kwargs=[{'color': colors['left-stim'], 'lw': 0.5},
                   {'color': colors['left-no-stim'], 'lw': 0.5},
                   {'color': colors['right-stim'], 'lw': 0.5},
                   {'color': colors['right-no-stim'], 'lw': 0.5}],
    eventline_kwargs={'lw': 0}, include_raster=False, labels=['L w. 5-HT', 'L', 'R w. 5-HT', 'R'], include_legend=True)
ax.set(ylabel='Firing rate (spikes/s)', xlabel='Time from trial start (s)',
       yticks=[0, 5, 10, 15, 20, 25], xticks=[-0.5, 0, 0.5, 1], xticklabels=[-0.5, 0, 0.5, 1],
       title='Example M2 neuron')
ax.legend(bbox_to_anchor=(0.9, 1))
sns.despine(trim=False)
plt.tight_layout()
plt.savefig(join(fig_path, 'figure6', f'example_neuron_{NEURON_ID}.pdf'))
plt.show()