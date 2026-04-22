# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 10:03:56 2024

By Guido Meijer
"""

import numpy as np
import os
from os import path
import pandas as pd
import seaborn as sns
from scipy import stats
from pathlib import Path
import matplotlib.pyplot as plt
from stim_functions import paths, figure_style, load_subjects

# Paths
f_path, save_path = paths()
try:
    current_dir = Path(__file__).resolve().parent
except NameError:
    current_dir = Path(os.getcwd())
fig_path = Path(f_path) / current_dir.name

# Initialize
_, data_path = paths()
subjects = load_subjects()
colors, dpi = figure_style()

# Load in all trials 
all_trials = pd.read_csv(path.join(save_path, 'all_trials.csv'))

# Loop over subjects
rt_df = pd.DataFrame()
for i, nickname in enumerate(np.unique(all_trials['subject'])):
    print(f'Subject {nickname} ({i} of {np.unique(all_trials["subject"]).shape[0]})')

    # Get trials DataFrame
    trials = all_trials[all_trials['subject'] == nickname].copy()
    trials = trials.drop(columns=['subject'])
    trials = trials.reset_index(drop=True)
   
    # Get reaction time medians
    rt_opto_median = trials.loc[trials['laser_stimulation'] == 1, 'time_to_choice'].median()
    rt_no_opto_median = trials.loc[trials['laser_stimulation'] == 0, 'time_to_choice'].median()
             
    # Add to dataframe
    rt_df = pd.concat((rt_df, pd.DataFrame(index=[rt_df.shape[0]], data={
        'subject': nickname, 
        'rt_opto': rt_opto_median, 'rt_no_opto': rt_no_opto_median})))

# %%

t, p = stats.ttest_rel(rt_df['rt_opto'], rt_df['rt_no_opto'])
print(f'DF = {rt_df.shape[0]-1}, t = {t:.3f}, p = {p:.3f}')

f, ax1 = plt.subplots(1, 1, figsize=(1.2, 1.75), dpi=dpi)
for i in rt_df.index:
    ax1.plot([0, 1], [rt_df.loc[i, 'rt_opto'], rt_df.loc[i, 'rt_no_opto']], marker='o', color='k',
             markersize=2)
ax1.set(xticks=[0, 1], xticklabels=['5-HT', 'No 5-HT'], ylabel='Median reaction time (s)',
        yticks=np.arange(0.3, 0.61, 0.1), xlim=[-0.2, 1.2])
ax1.text(0.5, 0.58, 'n.s.', ha='center', va='center')

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(path.join(fig_path, 'reaction_time_median.pdf'))
plt.show()

