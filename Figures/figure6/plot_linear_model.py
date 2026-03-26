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
from scipy.stats import ttest_1samp
import pingouin as pg
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
linear_df = pd.read_csv(data_path / 'linear_model_results_3-8.csv')
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

# Statistics versus zero
for var in ['coef_choice_abs', 'coef_stim_abs', 'coef_interaction']:
    t_stat, p_val = ttest_1samp(summary_df[var], 0)
    bf = pg.bayesfactor_ttest(t_stat, nx=summary_df[var].shape[0])
    print(f'{var}: t({len(summary_df)-1}) = {t_stat:.3f}, p = {p_val:.3f}, BF10 = {bf:.3f}')

summary_long_df = pd.melt(summary_df, id_vars=['subject'], value_vars=['coef_choice_abs', 'coef_stim_abs', 'coef_interaction'])
props = {'boxprops':{'facecolor':'none', 'edgecolor':'none'}, 'medianprops':{'color':'none'},
         'whiskerprops':{'color':'none'}, 'capprops':{'color':'none'}}
ax1.plot([-0.5, 2.5], [0, 0], ls='--', color='grey', zorder=0, lw=0.75)
sns.swarmplot(x='variable', y='value', data=summary_long_df, ax=ax1, color='k', s=3, zorder=1)
sns.boxplot(x='variable', y='value', data=summary_long_df, showmeans=True, fliersize=0, zorder=2,
            meanprops={"marker": "_", "markeredgecolor": "red", "markersize": "10"}, **props)
ax1.text(0, 4, '*', fontsize=12, ha='center', va='center')
ax1.text(1, 4, '*', fontsize=12, ha='center', va='center')
ax1.set(ylabel='GLM coefficient', yticks=[-1, 0, 1, 2, 3, 4],
        xlabel='', xticks=[0, 1, 2], xticklabels=['Choice', '5-HT', 'Interaction'])

plt.tight_layout()
sns.despine(trim=True)
plt.savefig(fig_path / 'linear_model_per_subject.jpg', dpi=600)
plt.savefig(fig_path / 'linear_model_per_subject.pdf')
plt.show()
