# -*- coding: utf-8 -*-
"""
Created on Mon May  5 17:01:49 2025 by Guido Meijer
"""


import numpy as np
import pandas as pd
from os.path import join
from stim_functions import paths, remap
from iblatlas.regions import BrainRegions
from iblbrainviewer.api import FeatureUploader
up = FeatureUploader()
br = BrainRegions()

# Get paths
_, save_path = paths()

# Load in results
expression_df = pd.read_csv(join(save_path, 'receptor_expression.csv'))
expression_df = expression_df[~np.isin(expression_df['acronym'], ['MMme', 'CUL4, 5', 'CUL4, 5gr', 'CUL4, 5mo'])]

# Put into numpy arrays
acronyms, values = dict(), dict()
receptors = np.unique(expression_df['receptor'])
for receptor in receptors:
    acronyms[receptor] = expression_df.loc[expression_df['receptor'] == receptor, 'acronym'].values.astype(str)
    values[receptor] = expression_df.loc[expression_df['receptor'] == receptor, 'expression_energy'].values
    up.local_features(receptor, acronyms[receptor], values[receptor], hemisphere='left', output_dir=save_path)

# Load bucket
up = FeatureUploader('meijer_5ht_receptors')

# Descriptions
short_desc = 'Expression of serotonin receptors across the brain'
long_desc = 'For each brain region the expression energy of the selected serotonergic receptor is shown'
up.patch_bucket(short_desc=short_desc, long_desc=long_desc)

# Upload the features
for receptor in receptors:
    if up.features_exist(receptor):
        up.patch_features(receptor, acronyms[receptor], values[receptor], hemisphere='left')
    else:
        up.create_features(receptor, acronyms[receptor], values[receptor], hemisphere='left')

# Generate url
url = up.get_buckets_url(['meijer_5ht_receptors'])
print(url)

# Get token
print(up.token)



     

