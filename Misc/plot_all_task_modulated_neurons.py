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
PATH = r'C:\Users\guido\Figures\5HT\Ephys\SingleNeurons\TaskModNeurons'
T_BEFORE = 0.5  # for plotting
T_AFTER = 1
BIN_SIZE = 0.025
SMOOTHING = 0.01
CENTER_ON = 'stimOn_times'

# Load in data
all_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))

for i, pid in enumerate(np.unique(all_neurons['pid'])):

    # Get eid
    eid = np.unique(all_neurons.loc[all_neurons['pid'] == pid, 'eid'])[0]
    probe = np.unique(all_neurons.loc[all_neurons['pid'] == pid, 'probe'])[0]
    subject = np.unique(all_neurons.loc[all_neurons['pid'] == pid, 'subject'])[0]
    date = np.unique(all_neurons.loc[all_neurons['pid'] == pid, 'date'])[0]
    print(f'Starting {subject}, {date}, {probe}')

    # Load in trials
    try:
        trials = load_trials(eid, laser_stimulation=True, one=one)
    except:
        print('Could not load trials')
        continue

    # Drop NaNs
    if CENTER_ON == 'firstMovement_times':
        trials = trials[~np.isnan(trials['firstMovement_times'])]

    # Generate an array with 1, 2, 3, 4 for four conditions: laser_stimulation 0 or 1 and choice -1 or 1
    trials['conditions'] = 0
    trials.loc[(trials['laser_stimulation'] == 1) & (trials['choice'] == -1), 'conditions'] = 1
    trials.loc[(trials['laser_stimulation'] == 0) & (trials['choice'] == -1), 'conditions'] = 2
    trials.loc[(trials['laser_stimulation'] == 1) & (trials['choice'] == 1), 'conditions'] = 3
    trials.loc[(trials['laser_stimulation'] == 0) & (trials['choice'] == 1), 'conditions'] = 4
    #trials = trials[(trials['conditions'] != 0) & (np.abs(trials['signed_contrast']) > 0) & (np.abs(trials['signed_contrast']) < 1)]

    # Load in spikes
    sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
    spikes, clusters, channels = sl.load_spike_sorting()
    clusters = sl.merge_clusters(spikes, clusters, channels)

    if 'acronym' not in clusters.keys():
        print(f'No brain regions found for {eid}')
        continue
        
    modulated = all_neurons[(all_neurons['pid'] == pid) & (all_neurons['modulated'] == 1)]
    
    for n, ind in enumerate(modulated.index.values):
        region = modulated.loc[ind, 'region']
        subject = modulated.loc[ind, 'subject']
        date = modulated.loc[ind, 'date']
        neuron_id = modulated.loc[ind, 'neuron_id']
        
        # Plot PSTH
        try:
            p, ax = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
            peri_multiple_events_time_histogram(
                spikes.times, spikes.clusters, trials[CENTER_ON],
                trials['conditions'],
                neuron_id, t_before=T_BEFORE, t_after=T_AFTER, bin_size=BIN_SIZE, ax=ax,
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
                eventline_kwargs={'lw': 0}, include_raster=False)
            ax.set(ylabel='Firing rate (spikes/s)', xlabel='Time from trial start (s)',
                   yticks=np.linspace(0, np.round(ax.get_ylim()[1]), 3), xticks=[-T_BEFORE, 0, T_AFTER])
            if np.round(ax.get_ylim()[1]) % 2 == 0:
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            else:
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            sns.despine(trim=False)
            plt.tight_layout()
            plt.savefig(
                join(PATH, f'{region}_{pid}_neuron{neuron_id}.jpg'), dpi=300)
            plt.close(p)
        except:
            continue