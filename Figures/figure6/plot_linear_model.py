# -*- coding: utf-8 -*-
"""
Author: Guido Meijer
Date: 03/03/2026
"""
# %%

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from stim_functions import paths, figure_style, remap, combine_regions, load_subjects
colors, dpi = figure_style()

# Get path
f_path,data_path = paths()
data_path = Path(data_path)
try:
    current_dir = Path(__file__).resolve().parent
except NameError:
    current_dir = Path(os.getcwd())
fig_path = Path(f_path) / current_dir.name

# Load in data
linear_df = pd.read_csv(data_path / 'linear_model_results.csv')
linear_df['region'] = combine_regions(remap(linear_df['acronym']))
linear_df = linear_df[linear_df['region'] != 'root']
linear_df['coef_choice_abs'] = np.abs(linear_df['coef_choice'])
linear_df['coef_stim_abs'] = np.abs(linear_df['coef_stim'])

# Add genotype and subject number
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    linear_df.loc[linear_df['subject'] == nickname,
                    'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]
linear_df = linear_df[linear_df['sert-cre'] == 1]

# %% Plot results per subject
f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)

summary_df = linear_df.groupby(['subject'])[['coef_choice_abs', 'coef_stim_abs', 'coef_interaction']].mean().reset_index()
summary_long_df = pd.melt(summary_df, id_vars=['subject'], value_vars=['coef_choice_abs', 'coef_stim_abs', 'coef_interaction'])
sns.barplot(x='variable', y='value', data=summary_long_df, errorbar=None, ax=ax1, color='gray')
sns.swarmplot(x='variable', y='value', data=summary_long_df, ax=ax1, color='k', s=3)
ax1.set(ylabel='GLM coefficient', yticks=[-1, 0, 1, 2, 3, 4],
        xlabel='', xticks=[0, 1, 2], xticklabels=['Choice', '5-HT', 'Interaction'])

plt.tight_layout()
sns.despine(trim=True)
plt.savefig(fig_path / 'linear_model_per_subject.jpg', dpi=600)
plt.savefig(fig_path / 'linear_model_per_subject.pdf')
plt.show()

# %% Plot results per neuron
f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)

long_df = pd.melt(linear_df, value_vars=['coef_choice_abs', 'coef_stim_abs', 'coef_interaction'])
ax1.plot([-0.5, 2.5], [0, 0], ls='--', color='red')
sns.boxplot(x='variable', y='value', data=long_df, ax=ax1, color='gray', fliersize=0)
ax1.set(ylabel='GLM coefficient', yticks=[-1, 0, 1, 2, 3, 4], ylim=[-1, 1],
        xlabel='', xticks=[0, 1, 2], xticklabels=['Choice', '5-HT', 'Interaction'])

plt.tight_layout()
sns.despine(trim=True)
plt.savefig(fig_path / 'linear_model_per_neuron.jpg', dpi=600)
plt.savefig(fig_path / 'linear_model_per_neuron.pdf')
plt.show()

