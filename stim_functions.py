# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:22:01 2020

By: Guido Meijer
"""

import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import tkinter as tk
import statsmodels.api as sm
from sklearn.model_selection import KFold
from scipy.stats import binned_statistic
from scipy.signal import convolve
from scipy.stats import binned_statistic_2d
from scipy.signal.windows import gaussian
import pathlib
from brainbox import singlecell
from os.path import join, realpath, dirname, isfile
from matplotlib import colors as matplotlib_colors
from scipy.interpolate import interp1d
import json
from pathlib import Path
from brainbox.io.one import SessionLoader
from brainbox.io.spikeglx import spikeglx
from brainbox.metrics.single_units import spike_sorting_metrics
from brainbox.io.one import SpikeSortingLoader
from iblutil.numerical import ismember
from iblatlas.regions import BrainRegions
from iblatlas.atlas import AllenAtlas
from one.api import ONE


def init_one(local=False, open_one=True):
    """Initialize an instance of the ONE class with specified configuration.

    Parameters
    ----------
    local : bool, optional
        If True, initializes ONE in 'local' mode. Defaults to False, which sets the mode to 'remote'.
    open_one : bool, optional
        If True, initializes ONE with a specific base URL and credentials for the Open Alyx
        instance. Defaults to True.

    Returns
    -------
    one.api.ONE
        An instance of the ONE class configured based on the provided parameters.
    """
    if local:
        mode='local'
    else:
        mode='remote'
    if open_one:
        return ONE(mode=mode, base_url='https://openalyx.internationalbrainlab.org',
                   password='international', silent=True)
    else:
        return ONE(mode=mode)


def load_subjects():
    """Load subject information from the subjects.csv file.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing subject information with columns from the CSV file.
        The 'subject_nr' column is cast to an integer.
    """
    subjects = pd.read_csv(join(pathlib.Path(__file__).parent.resolve(), 'subjects.csv'),
                           delimiter=';|,', engine='python')
    subjects['subject_nr'] = subjects['subject_nr'].astype(int)
    subjects = subjects.reset_index(drop=True)
    return subjects


def paths(save_dir='repo'):
    """Get paths for saving figures and data.

    Loads paths from 'paths.json'. If the file doesn't exist, it prompts the user for input
    and creates it. The save directory can be specified to be within the repository
    or the ONE cache directory.

    Parameters
    ----------
    save_dir : {'repo', 'cache'}, optional
        Specifies where to save data.
        'repo': saves in the 'Data' directory within the repository (for small files).
        'cache': saves in a 'serotonin' subdirectory of the ONE cache (for large files).
        Defaults to 'repo'.

    Returns
    -------
    fig_path : str
        Path to where to save the figures.
    save_path : str or pathlib.Path
        Path to where to save the output data.
    """
    if not isfile(join(dirname(realpath(__file__)), 'paths.json')):
        path_dict = dict()
        path_dict['fig_path'] = input('Path folder to save figures: ')
        path_dict['save_path'] = join(dirname(realpath(__file__)), 'Data')
        path_file = open(join(dirname(realpath(__file__)), 'paths.json'), 'w')
        json.dump(path_dict, path_file)
        path_file.close()
    with open(join(dirname(realpath(__file__)), 'paths.json')) as json_file:
        path_dict = json.load(json_file)
    if save_dir == 'cache':
        one = ONE(mode='local')
        save_dir = Path(one.cache_dir, 'serotonin')
        save_dir.mkdir(exist_ok=True)
    elif save_dir == 'repo':
        save_dir = path_dict['save_path']
    else:
        print('save_dir must be either repo or cache')
    return path_dict['fig_path'], save_dir


def figure_style():
    """Set a standard style for plotting figures.

    This function configures seaborn and matplotlib to a standard style for this project.
    It also defines a color palette for different brain regions, experimental conditions, etc.

    Returns
    -------
    colors : dict
        A dictionary of colors for plotting.
    dpi : float
        The dots per inch (DPI) for the screen, calculated based on screen width.
    """
    sns.set(style="ticks", context="paper",
            font="Arial",
            rc={"font.size": 7,
                "figure.titlesize": 7,
                "axes.titlesize": 7,
                "axes.labelsize": 7,
                "axes.linewidth": 0.5,
                "lines.linewidth": 1,
                "lines.markersize": 3,
                "xtick.labelsize": 7,
                "ytick.labelsize": 7,
                "savefig.transparent": True,
                "xtick.major.size": 2.5,
                "ytick.major.size": 2.5,
                "xtick.major.width": 0.5,
                "ytick.major.width": 0.5,
                "xtick.minor.size": 2,
                "ytick.minor.size": 2,
                "xtick.minor.width": 0.5,
                "ytick.minor.width": 0.5,
                'legend.fontsize': 7,
                'legend.title_fontsize': 7,
                'legend.frameon': False,
                })
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    subject_pal = sns.color_palette(
        np.concatenate((sns.color_palette('tab20'),
                        [matplotlib_colors.to_rgb('maroon'), np.array([0, 0, 0])])))
    frontal = sns.color_palette('Dark2')[1]
    sensory = sns.color_palette('Dark2')[5]
    hipp = sns.color_palette('Dark2')[4]
    amygdala = sns.color_palette('Dark2')[2]
    thalamus = sns.color_palette('Dark2')[0]
    striatum = sns.color_palette('Set1')[7]
    midbrain = sns.color_palette('Dark2')[3]

    colors = {'subject_palette': subject_pal,
              'grey': [0.7, 0.7, 0.7],
              'sert': sns.color_palette('Dark2')[0],
              'wt': [0.6, 0.6, 0.6],
              'enhanced': sns.color_palette('colorblind')[3],
              'suppressed': sns.color_palette('colorblind')[0],
              'stim': 'dodgerblue',
              'no-stim': [0.65, 0.65, 0.65],
              'NS': sns.color_palette('Set2')[0],
              'WS': sns.color_palette('Set2')[1],
              'WS1': sns.color_palette('Set2')[1],
              'WS2': sns.color_palette('Set2')[2],

              'OFC': sns.color_palette('Set1')[7],
              'mPFC': sns.color_palette('Dark2')[1],
              'M2': sns.color_palette('Dark2')[2],
              'Amyg.': sns.color_palette('Dark2')[6],
              'HPC': sns.color_palette('Dark2')[3],
              'VIS': sns.color_palette('Dark2')[5],
              'Pir.': sns.color_palette('Dark2')[4],
              'SC': sns.color_palette('Dark2')[7],
              'Thal.': sns.color_palette('tab10')[9],
              'PAG': sns.color_palette('Dark2')[0],
              'Str.': sns.color_palette('Accent')[1],
              'MRN': sns.color_palette('Accent')[2],
              'OLF': sns.color_palette('tab10')[8],
              'RSP': 'r',
              'Orbitofrontal cortex': sns.color_palette('Set1')[7],
              'Medial prefrontal cortex': sns.color_palette('Dark2')[1],
              'Secondary motor cortex': sns.color_palette('Dark2')[2],
              'Amygdala': sns.color_palette('Dark2')[6],
              'Hippocampus': sns.color_palette('Dark2')[3],
              'Visual cortex': sns.color_palette('Dark2')[5],
              'Piriform cortex': sns.color_palette('Dark2')[4],
              'Superior colliculus': sns.color_palette('Dark2')[7],
              'Thalamus': sns.color_palette('tab10')[9],
              'Periaqueductal gray': sns.color_palette('Dark2')[0],
              'Striatum': sns.color_palette('Accent')[1],
              'Olfactory areas': sns.color_palette('tab10')[8],
              'Retrosplenial cortex': 'r',
              'left-stim': 'royalblue',
              'left-no-stim': 'rebeccapurple',
              'right-stim': 'darkorange',
              'right-no-stim': 'firebrick'}
    screen_width = tk.Tk().winfo_screenwidth()
    dpi = screen_width / 10
    return colors, dpi


def add_significance(x, p_values, ax, alpha=0.05):
    """Add significance markers (horizontal lines) to a plot.

    This function identifies contiguous regions where `p_values < alpha` and
    draws horizontal lines over these regions on the provided axes.

    Parameters
    ----------
    x : array_like
        The x-coordinates corresponding to the p-values.
    p_values : array_like
        The p-values to evaluate for significance.
    ax : matplotlib.axes.Axes
        The axes object to which the significance markers will be added.
    alpha : float, optional
        The significance threshold. P-values below this are considered significant.
        Default is 0.05.
    """
    p_sig = p_values < alpha
    start_end = np.where(np.concatenate(([0], np.diff(p_sig).astype(int))))[0]
    if p_sig[0] == True:
        start_end = np.concatenate(([0], start_end))
    if p_sig[-1] == True:
        start_end = np.concatenate((start_end, [p_sig.shape[0]-1]))
    y = ax.get_ylim()[1]
    for (i, ind) in zip(np.arange(0, (start_end.shape[0] // 2) + 1, 2), start_end[::2]):
        ax.plot([x[ind], x[start_end[i+1]]], [y + (y*0.05), y + (y*0.05)], color='k', lw=1.5,
                clip_on=False)


def get_artifact_neurons():
    """Load a dataframe of neurons that are considered artifacts.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing information about artifact neurons, loaded from
        'artifact_neurons.csv'.
    """
    artifact_neurons = pd.read_csv(
        join(pathlib.Path(__file__).parent.resolve(), 'artifact_neurons.csv'))
    return artifact_neurons


def remove_artifact_neurons(df):
    """Remove artifact neurons from a DataFrame.

    This function loads a list of known artifact neurons from 'artifact_neurons.csv'
    and removes them from the input DataFrame `df` by performing an outer merge
    and keeping only the entries that are unique to the input DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame containing neuron data. It must contain columns to identify
        neurons, either ['pid', 'neuron_id', 'subject', 'probe', 'date'] or
        ['subject', 'probe', 'date', 'neuron_id'].

    Returns
    -------
    pandas.DataFrame
        The input DataFrame with artifact neurons removed.
    """
    artifact_neurons = pd.read_csv(
        join(pathlib.Path(__file__).parent.resolve(), 'artifact_neurons.csv'))
    for i, column in enumerate(df.columns):
        if df[column].dtype == bool:
            df[column] = df[column].astype('boolean')
    if 'pid' in df.columns:
        df = pd.merge(df, artifact_neurons, indicator=True, how='outer',
                      on=['pid', 'neuron_id', 'subject', 'probe', 'date']).query('_merge=="left_only"').drop('_merge', axis=1)
    else:
        df = pd.merge(df, artifact_neurons, indicator=True, how='outer',
                      on=['subject', 'probe', 'date', 'neuron_id']).query('_merge=="left_only"').drop('_merge', axis=1)
    return df


def query_ephys_sessions(acronym=None, one=None):
    """Query ephys recording sessions from the database.

    Queries for sessions that are part of the 'serotonin_inference' project,
    have a session QC less than 50, and have an alignment count greater than 0.
    It also filters to include only subjects listed in 'subjects.csv'.

    Parameters
    ----------
    acronym : str or list of str, optional
        If provided, only returns sessions that have recordings in the brain region(s)
        with the given Allen acronym(s). Defaults to None.
    one : one.api.ONE, optional
        An initialized ONE instance. If None, a new one is initialized.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with identifiers for each matching ephys recording, with columns:
        'pid', 'eid', 'probe', 'subject', 'date'.
    """
    if one is None:
        one = init_one()

    # Construct django query string
    DJANGO_STR = ('session__projects__name__icontains,serotonin_inference,'
                  'session__qc__lt,50,json__extended_qc__alignment_count__gt,0')

    # Query sessions
    if acronym is None:
        ins = one.alyx.rest('insertions', 'list', django=DJANGO_STR)
    elif type(acronym) is str:
        ins = one.alyx.rest('insertions', 'list', django=DJANGO_STR, atlas_acronym=acronym)
    else:
        ins = []
        for i, ac in enumerate(acronym):
            ins = ins + one.alyx.rest('insertions', 'list', django=DJANGO_STR, atlas_acronym=ac)

    # Only include subjects from subjects.csv
    incl_subjects = load_subjects()
    ins = [i for i in ins if i['session_info']['subject'] in incl_subjects['subject'].values]

    # Get list of eids and probes
    rec = pd.DataFrame()
    rec['pid'] = np.array([i['id'] for i in ins])
    rec['eid'] = np.array([i['session'] for i in ins])
    rec['probe'] = np.array([i['name'] for i in ins])
    rec['subject'] = np.array([i['session_info']['subject'] for i in ins])
    rec['date'] = np.array([i['session_info']['start_time'][:10] for i in ins])
    rec = rec.drop_duplicates('pid', ignore_index=True)
    return rec


def remap(acronyms, source='Allen', dest='Beryl', combine=False, split_thalamus=False,
          abbreviate=True, brainregions=None):
    """Remap a list of brain region acronyms from one mapping to another.

    Parameters
    ----------
    acronyms : list or array_like
        A list of brain region acronyms to be remapped.
    source : str, optional
        The source mapping to use for remapping. Default is 'Allen'.
    dest : str, optional
        The destination mapping to remap to. Default is 'Beryl'.
    combine : bool, optional
        If True, combines remapped regions into broader categories using `combine_regions`.
        Default is False.
    split_thalamus : bool, optional
        If True and `combine` is True, splits thalamus regions into subcategories.
        Default is False.
    abbreviate : bool, optional
        If True and `combine` is True, abbreviates combined region names.
        Default is True.
    brainregions : iblatlas.regions.BrainRegions, optional
        An instance of the BrainRegions class. If None, a new instance is created.
        Default is None.

    Returns
    -------
    numpy.ndarray
        The remapped acronyms. If `combine` is True, returns combined region names.
    """
    br = brainregions or BrainRegions()
    _, inds = ismember(br.acronym2id(acronyms), br.id[br.mappings[source]])
    remapped_acronyms = br.get(br.id[br.mappings[dest][inds]])['acronym']
    if combine:
        return combine_regions(remapped_acronyms, split_thalamus=split_thalamus, abbreviate=abbreviate)
    else:
        return remapped_acronyms


def combine_regions(acronyms, split_thalamus=False, abbreviate=True):
    """Combine Beryl atlas acronyms into broader, functionally-defined region groups.

    It is recommended to first remap acronyms to the Beryl atlas using the `remap` function.

    Parameters
    ----------
    acronyms : array_like
        An array of brain region acronyms (should be in Beryl mapping).
    split_thalamus : bool, optional
        If True, thalamic subregions are kept separate instead of being grouped into 'Thal.'.
        Default is False.
    abbreviate : bool, optional
        If True, returns abbreviated region names (e.g., 'mPFC').
        If False, returns full region names (e.g., 'Medial prefrontal cortex').
        Default is True.

    Returns
    -------
    numpy.ndarray
        An array of the same length as `acronyms` containing the combined region names.
        Regions not part of any group are labeled as 'root'.
    """
    regions = np.array(['root'] * len(acronyms), dtype=object)
    if abbreviate:
        regions[np.in1d(acronyms, ['ILA', 'PL', 'ACAd', 'ACAv'])] = 'mPFC'
        regions[np.in1d(acronyms, ['MOs'])] = 'M2'
        regions[np.in1d(acronyms, ['ORBl', 'ORBm'])] = 'OFC'
        if split_thalamus:
            regions[np.in1d(acronyms, ['PO'])] = 'PO'
            regions[np.in1d(acronyms, ['LP'])] = 'LP'
            regions[np.in1d(acronyms, ['LD'])] = 'LD'
            regions[np.in1d(acronyms, ['RT'])] = 'RT'
            regions[np.in1d(acronyms, ['VAL'])] = 'VAL'
        else:
            regions[np.in1d(acronyms, ['PO', 'LP', 'LD', 'RT', 'VAL'])] = 'Thal.'
        regions[np.in1d(acronyms, ['SCm', 'SCs', 'SCig', 'SCsg', 'SCdg'])] = 'SC'
        regions[np.in1d(acronyms, ['RSPv', 'RSPd'])] = 'RSP'
        # regions[np.in1d(acronyms, ['ZI'])] = 'ZI'
        regions[np.in1d(acronyms, ['PAG'])] = 'PAG'
        # regions[np.in1d(acronyms, ['SSp-bfd'])] = 'BC'
        # regions[np.in1d(acronyms, ['LGv', 'LGd'])] = 'LG'
        regions[np.in1d(acronyms, ['PIR'])] = 'Pir.'
        # regions[np.in1d(acronyms, ['SNr', 'SNc', 'SNl'])] = 'SN'
        regions[np.in1d(acronyms, ['VISa', 'VISam', 'VISp', 'VISpm'])] = 'VIS'
        # regions[np.in1d(acronyms, ['AId', 'AIv', 'AIp'])] = 'AI'
        regions[np.in1d(acronyms, ['MEA', 'CEA', 'BLA', 'COAa'])] = 'Amyg.'
        regions[np.in1d(acronyms, ['AON', 'TTd', 'DP'])] = 'OLF'
        regions[np.in1d(acronyms, ['CP', 'STR', 'STRd', 'STRv'])] = 'Str.'
        regions[np.in1d(acronyms, ['CA1', 'CA3', 'DG'])] = 'HPC'
    else:
        regions[np.in1d(acronyms, ['ILA', 'PL', 'ACAd', 'ACAv'])] = 'Medial prefrontal cortex'
        regions[np.in1d(acronyms, ['MOs'])] = 'Secondary motor cortex'
        regions[np.in1d(acronyms, ['ORBl', 'ORBm'])] = 'Orbitofrontal cortex'
        if split_thalamus:
            regions[np.in1d(acronyms, ['PO'])] = 'Thalamus (PO)'
            regions[np.in1d(acronyms, ['LP'])] = 'Thalamus (LP)'
            regions[np.in1d(acronyms, ['LD'])] = 'Thalamus (LD)'
            regions[np.in1d(acronyms, ['RT'])] = 'Thalamus (RT)'
            regions[np.in1d(acronyms, ['VAL'])] = 'Thalamus (VAL)'
        else:
            regions[np.in1d(acronyms, ['PO', 'LP', 'LD', 'RT', 'VAL'])] = 'Thalamus'
        regions[np.in1d(acronyms, ['SCm', 'SCs', 'SCig', 'SCsg', 'SCdg'])] = 'Superior colliculus'
        regions[np.in1d(acronyms, ['RSPv', 'RSPd'])] = 'Retrosplenial cortex'
        regions[np.in1d(acronyms, ['AON', 'TTd', 'DP'])] = 'Olfactory areas'
        # regions[np.in1d(acronyms, ['ZI'])] = 'Zona incerta'
        regions[np.in1d(acronyms, ['PAG'])] = 'Periaqueductal gray'
        # regions[np.in1d(acronyms, ['AId', 'AIv', 'AIp'])] = 'Insular cortex'
        # regions[np.in1d(acronyms, ['SSp-bfd'])] = 'Barrel cortex'
        # regions[np.in1d(acronyms, ['LGv', 'LGd'])] = 'Lateral geniculate'
        regions[np.in1d(acronyms, ['PIR'])] = 'Piriform cortex'
        # regions[np.in1d(acronyms, ['SNr', 'SNc', 'SNl'])] = 'Substantia nigra'
        regions[np.in1d(acronyms, ['VISa', 'VISam', 'VISp', 'VISpm'])] = 'Visual cortex'
        regions[np.in1d(acronyms, ['MEA', 'CEA', 'BLA', 'COAa'])] = 'Amygdala'
        regions[np.in1d(acronyms, ['CP', 'STR', 'STRd', 'STRv'])] = 'Striatum'
        regions[np.in1d(acronyms, ['CA1', 'CA3', 'DG'])] = 'Hippocampus'
    return regions


def high_level_regions(acronyms, merge_cortex=False, only_vis=False, input_atlas='Allen'):
    """Map brain region acronyms to high-level functional groups.

    This function first remaps acronyms to a standard set (if `input_atlas` is 'Allen'),
    then combines them into first-level regions (e.g., 'mPFC', 'VIS'), and finally
    groups them into high-level categories like 'Frontal cortex', 'Midbrain', etc.

    Parameters
    ----------
    acronyms : list or array_like
        List of brain region acronyms to be mapped.
    merge_cortex : bool, optional
        If True, merges various cortical regions into a single 'Cortex' category.
        Default is False.
    only_vis : bool, optional
        If True and `merge_cortex` is False, maps only 'VIS' to 'Visual cortex' and other
        sensory areas are not grouped into 'Sensory cortex'. Default is False.
    input_atlas : str, optional
        The atlas of the input `acronyms`. If 'Allen', they will be remapped.
        Default is False.

    Returns
    -------
    numpy.ndarray
        Array of high-level brain region labels corresponding to the input acronyms.
        Regions not mapped are labeled 'root'.
    """

    if input_atlas == 'Allen':
        acronyms = remap(acronyms)
    first_level_regions = combine_regions(acronyms, abbreviate=True)
    cosmos_regions = remap(acronyms, dest='Cosmos')
    regions = np.array(['root'] * len(first_level_regions), dtype=object)
    if merge_cortex:
        # regions[cosmos_regions == 'Isocortex'] = 'Cortex'
        # regions[first_level_regions == 'Pir'] = 'Cortex'
        regions[np.in1d(first_level_regions, ['mPFC', 'OFC', 'M2', 'Pir', 'BC', 'VIS'])] = 'Cortex'
    else:
        regions[np.in1d(first_level_regions, ['mPFC', 'OFC', 'M2'])] = 'Frontal cortex'
        if only_vis:
            regions[np.in1d(first_level_regions, ['VIS'])] = 'Visual cortex'
        else:
            regions[np.in1d(first_level_regions, ['Pir', 'BC', 'VIS'])] = 'Sensory cortex'
    regions[cosmos_regions == 'MB'] = 'Midbrain'
    regions[cosmos_regions == 'HPF'] = 'Hippocampus'
    regions[cosmos_regions == 'TH'] = 'Thalamus'
    regions[np.in1d(first_level_regions, ['Amyg'])] = 'Amygdala'
    regions[np.in1d(acronyms, ['CP', 'ACB', 'FS'])] = 'Striatum'
    return regions


def get_full_region_name(acronyms):
    """Retrieve the full region names for a list of brain region acronyms.

    This function maps each acronym to its full region name using the Allen Brain Atlas.
    If an acronym cannot be found, it is returned as-is.

    Parameters
    ----------
    acronyms : list of str
        A list of brain region acronyms.

    Returns
    -------
    str or list of str
        If the input list contains a single acronym, returns its full name as a string.
        Otherwise, returns a list of full region names.
    """
    brainregions = BrainRegions()
    full_region_names = []
    for i, acronym in enumerate(acronyms):
        try:
            regname = brainregions.name[np.argwhere(brainregions.acronym == acronym).flatten()][0]
            full_region_names.append(regname)
        except IndexError:
            full_region_names.append(acronym)
    if len(full_region_names) == 1:
        return full_region_names[0]
    else:
        return full_region_names


def load_passive_opto_times(eid, one=None, freq=25):
    """Load passive optogenetic stimulation times from pre-extracted files.

    This function loads the timestamps of optogenetic stimulation trains and individual
    pulses for a given session and stimulation frequency. It expects the files to be
    pre-extracted and saved in a specific location.

    Parameters
    ----------
    eid : str
        The experiment ID of the session.
    one : one.api.ONE, optional
        An initialized ONE instance. If None, a new one is initialized. Used to get
        session details.
    freq : int, optional
        The frequency of the stimulation to load. Defaults to 25 Hz.

    Returns
    -------
    opto_train_times : numpy.ndarray
        Timestamps of the start of each pulse train. Returns an empty list if files
        are not found.
    opto_pulse_times : numpy.ndarray
        Timestamps of all individual pulses. Returns an empty list if files
        are not found.
    """
    if one is None:
        one = init_one()
    ses_details = one.get_details(eid)
    subject = ses_details['subject']
    date = ses_details['date']

    # Load in pulses from disk if already extracted
    _, save_path = paths()
    save_path = join(save_path, 'OptoTimes')
    if isfile(join(save_path, f'{subject}_{date}_pulse_trains_{freq}hz.npy')):
        opto_train_times = np.load(join(save_path, f'{subject}_{date}_pulse_trains_{freq}hz.npy'))
        opto_on_times = np.load(join(save_path, f'{subject}_{date}_ind_pulses_{freq}hz.npy'))
        return opto_train_times, opto_on_times
    else:
        print(f'Extracted opto pulse times not found for {subject} {date}!')
        return [], []


def load_trials(eid, laser_stimulation=False, invert_choice=False, invert_stimside=False, one=None):
    """Load and process trial data for a given experiment session.

    This function loads trial data from ONE and computes several additional
    behavioral variables.

    Parameters
    ----------
    eid : str
        Experiment ID for the session to load.
    laser_stimulation : bool, optional
        If True, includes laser stimulation data in the trials DataFrame. Default is False.
    invert_choice : bool, optional
        If True, inverts the choice values (-1 becomes 1, 1 becomes -1). Default is False.
    invert_stimside : bool, optional
        If True, inverts the stimulus side and signed contrast values. Default is False.
    one : one.api.ONE, optional
        An instance of the ONE API. If None, a new instance is created.

    Returns
    -------
    pd.DataFrame or None
        A DataFrame containing processed trial data. Returns None if no trials are
        found for the session. The DataFrame includes loaded data plus derived columns:
        - 'signed_contrast': Contrast with sign indicating side.
        - 'correct': 1 for correct, 0 for incorrect.
        - 'right_choice': 1 for right choice, 0 for left.
        - 'stim_side': -1 for left, 1 for right.
        - 'time_to_choice': Time from stimulus onset to feedback.
        - 'reaction_times': Time from stimulus onset to first movement.
        If `laser_stimulation` is True, it also adds:
        - 'laser_stimulation': 1 if laser was on, 0 otherwise.
        - 'laser_probability': Probability of laser stimulation.
        - 'probe_trial': Indicator for probe trials.

    Notes
    -----
    - If `laser_stimulation` is True and the laser probability dataset is unavailable,
      the function estimates the laser probability based on signed contrast and stimulation data.
    """

    one = one or ONE()

    data = one.load_object(eid, 'trials')
    data = {your_key: data[your_key] for your_key in [
        'stimOn_times', 'feedback_times', 'goCue_times', 'probabilityLeft', 'contrastLeft',
        'contrastRight', 'feedbackType', 'choice', 'firstMovement_times']}
    trials = pd.DataFrame(data=data)
    if trials.shape[0] == 0:
        return
    trials['signed_contrast'] = trials['contrastRight']
    trials.loc[trials['signed_contrast'].isnull(), 'signed_contrast'] = -trials['contrastLeft']
    if laser_stimulation:
        trials['laser_stimulation'] = one.load_dataset(
            eid, dataset='_ibl_trials.laserStimulation.npy')
        try:
            trials['laser_probability'] = one.load_dataset(
                eid, dataset='_ibl_trials.laserProbability.npy')
            trials['probe_trial'] = ((trials['laser_stimulation'] == 0) & (trials['laser_probability'] == 0.75)
                                     | (trials['laser_stimulation'] == 1) & (trials['laser_probability'] == 0.25)).astype(int)
        except:
            trials['laser_probability'] = trials['laser_stimulation'].copy()
            trials.loc[(trials['signed_contrast'] == 0)
                       & (trials['laser_stimulation'] == 0), 'laser_probability'] = 0.25
            trials.loc[(trials['signed_contrast'] == 0)
                       & (trials['laser_stimulation'] == 1), 'laser_probability'] = 0.75

    trials['correct'] = trials['feedbackType']
    trials.loc[trials['correct'] == -1, 'correct'] = 0
    trials['right_choice'] = -trials['choice']
    trials.loc[trials['right_choice'] == -1, 'right_choice'] = 0
    trials['stim_side'] = (trials['signed_contrast'] > 0).astype(int)
    trials.loc[trials['stim_side'] == 0, 'stim_side'] = -1
    trials.loc[(trials['signed_contrast'] == 0) & (trials['contrastLeft'].isnull()),
               'stim_side'] = 1
    trials.loc[(trials['signed_contrast'] == 0) & (trials['contrastRight'].isnull()),
               'stim_side'] = -1

    trials['time_to_choice'] = trials['feedback_times'] - trials['stimOn_times']
    if 'firstMovement_times' in trials.columns.values:
        trials['reaction_times'] = trials['firstMovement_times'] - trials['stimOn_times']
    if invert_choice:
        trials['choice'] = -trials['choice']
    if invert_stimside:
        trials['stim_side'] = -trials['stim_side']
        trials['signed_contrast'] = -trials['signed_contrast']
    return trials


def get_neuron_qc(pid, one=None, ba=None, force_rerun=False):
    """Compute or load neuron quality control (QC) metrics for a probe insertion.

    This function calculates single-unit QC metrics using `brainbox.metrics.single_units.spike_sorting_metrics`.
    If the metrics have already been computed and saved to a CSV file in the session's ALF
    directory, it loads them from disk unless `force_rerun` is True.

    Parameters
    ----------
    pid : str
        The probe insertion ID (e.g., '6c511e7c-add8-4833-8f55-9103813a21aa').
    one : one.api.ONE, optional
        An instance of the ONE API. If not provided, a new instance is created.
    ba : iblatlas.atlas.AllenAtlas, optional
        An instance of the AllenAtlas. If provided, it is used by the SpikeSortingLoader.
    force_rerun : bool, optional
        If True, forces recalculation of QC metrics even if they are already saved.
        Default is False.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the neuron QC metrics for all clusters.
    """

    one = one or ONE()

    # Check if QC is already computed
    eid, probe = one.pid2eid(pid)
    session_path = one.eid2path(eid)
    if isfile(join(session_path, 'alf', probe, 'neuron_qc_metrics.csv')) & ~force_rerun:
        print('Neuron QC metrics loaded from disk')
        qc_metrics = pd.read_csv(join(session_path, 'alf', probe, 'neuron_qc_metrics.csv'))
        return qc_metrics

    # Load in spikes
    if ba is None:
        sl = SpikeSortingLoader(pid=pid, one=one)
    else:
        sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
    spikes, clusters, channels = sl.load_spike_sorting()
    clusters = sl.merge_clusters(spikes, clusters, channels)

    # Calculate QC metrics
    print('Calculating neuron QC metrics')
    qc_metrics, _ = spike_sorting_metrics(spikes.times, spikes.clusters,
                                          spikes.amps, spikes.depths,
                                          cluster_ids=np.arange(clusters.channels.size))
    qc_metrics.to_csv(join(session_path, 'alf', probe, 'neuron_qc_metrics.csv'))
    return qc_metrics


def load_lfp(eid, probe, time_start, time_end, relative_to='begin', destriped=False, one=None):
    """Load a slice of local field potential (LFP) data for a specified time range.

    Parameters
    ----------
    eid : str
        Experiment ID for the session.
    probe : str
        Name of the probe (e.g., 'probe00').
    time_start : float
        Start time of the LFP slice in seconds.
    time_end : float
        End time of the LFP slice in seconds.
    relative_to : {'begin', 'end'}, optional
        Reference point for the time range. 'begin' means times are relative to the
        start of the recording. 'end' means times are relative to the end.
        Default is 'begin'.
    destriped : bool, optional
        If True, loads pre-destriped LFP data from a local cache. If False, downloads
        the raw LFP data using ONE. Default is False.
    one : one.api.ONE, optional
        Instance of the ONE API. If None, a new instance is created.

    Returns
    -------
    signal : numpy.ndarray
        The LFP signal for the specified time range, with shape (n_channels, n_samples).
    time : numpy.ndarray
        Array of time points corresponding to the LFP signal samples.
    """
    one = one or ONE()
    destriped_lfp_path = join(paths()[1], 'LFP')

    # Download LFP data
    if destriped:
        ses_details = one.get_details(eid)
        subject = ses_details['subject']
        date = ses_details['start_time'][:10]
        lfp_path = join(destriped_lfp_path, f'{subject}_{date}_{probe}_destriped_lfp.cbin')
    else:
        lfp_paths, _ = one.load_datasets(eid, download_only=True, datasets=[
            '_spikeglx_ephysData_g*_t0.imec*.lf.cbin', '_spikeglx_ephysData_g*_t0.imec*.lf.meta',
            '_spikeglx_ephysData_g*_t0.imec*.lf.ch'], collections=[f'raw_ephys_data/{probe}'] * 3)
        lfp_path = lfp_paths[0]
    sr = spikeglx.Reader(lfp_path)

    # Convert time to samples
    if relative_to == 'begin':
        samples_start = int(time_start * sr.fs)
        samples_end = int(time_end * sr.fs)
    elif relative_to == 'end':
        samples_start = sr.shape[0] - int(time_start * sr.fs)
        samples_end = sr.shape[0] - int(time_end * sr.fs)

    # Load in lfp slice
    signal = sr.read(nsel=slice(samples_start, samples_end, None), csel=slice(None, None, None))[0]
    signal = signal.T
    time = np.arange(samples_start, samples_end) / sr.fs

    return signal, time


def plot_scalar_on_slice(
        regions, values, coord=-1000, slice='coronal', mapping='Beryl', hemisphere='left',
        cmap='viridis', background='boundary', clevels=None, brain_atlas=None, colorbar=False, ax=None):
    """Plot scalar values for brain regions on a 2D atlas slice.

    Parameters
    ----------
    regions : array_like
        Array of acronyms of Allen regions.
    values : array_like
        Array of scalar values, one per region in `regions`. If `hemisphere` is 'both'
        and different values are desired for each hemisphere, `values` should be a
        2D array with shape (n_regions, 2), with column 0 for left and column 1 for right.
    coord : int, optional
        Coordinate of the slice in micrometers (AP for coronal, ML for sagittal,
        DV for horizontal). Not used if `slice` is 'top'. Default is -1000.
    slice : {'coronal', 'sagittal', 'horizontal', 'top'}, optional
        Orientation of the slice. 'top' provides a top-down view of the brain.
        Default is 'coronal'.
    mapping : {'Allen', 'Beryl', 'Cosmos'}, optional
        Atlas mapping to use. Default is 'Beryl'.
    hemisphere : {'left', 'right', 'both'}, optional
        Hemisphere to display. Default is 'left'.
    background : {'image', 'boundary'}, optional
        Background to plot data on. 'image' uses the atlas image, 'boundary' uses
        region outlines. Default is 'boundary'.
    cmap : str or matplotlib.colors.Colormap, optional
        Colormap for the scalar values. Default is 'viridis'.
    clevels : tuple, optional
        Tuple of (min, max) for the color levels. If None, determined from `values`.
        Default is None.
    brain_atlas : iblatlas.atlas.AllenAtlas, optional
        An AllenAtlas instance. If None, a new one is created.
    colorbar : bool, optional
        Whether to plot a colorbar. Default is False.
    ax : matplotlib.axes.Axes, optional
        An existing axes object to plot on. If None, a new figure and axes are created.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object with the plot.
    """
    if clevels is None:
        clevels = (np.min(values), np.max(values))

    ba = brain_atlas or AllenAtlas()
    br = ba.regions

    # Find the mapping to use
    map_ext = '-lr'
    map = mapping + map_ext

    region_values = np.zeros_like(br.id) * np.nan

    if len(values.shape) == 2:
        for r, vL, vR in zip(regions, values[:, 0], values[:, 1]):
            region_values[np.where(br.acronym[br.mappings[map]] == r)[0][0]] = vR
            region_values[np.where(br.acronym[br.mappings[map]] == r)[0][1]] = vL
    else:
        for r, v in zip(regions, values):
            region_values[np.where(br.acronym[br.mappings[map]] == r)[0]] = v

        lr_divide = int((br.id.shape[0] - 1) / 2)
        if hemisphere == 'left':
            region_values[0:lr_divide] = np.nan
        elif hemisphere == 'right':
            region_values[lr_divide:] = np.nan
            region_values[0] = np.nan

    if ax:
        fig = ax.get_figure()
    else:
        fig, ax = plt.subplots()

    if background == 'boundary':
        cmap_bound = matplotlib.cm.get_cmap("bone_r").copy()
        cmap_bound.set_under([1, 1, 1], 0)

    if slice == 'coronal':

        if background == 'image':
            ba.plot_cslice(coord / 1e6, volume='image', mapping=map, ax=ax)
            ba.plot_cslice(
                coord / 1e6, volume='value', region_values=region_values, mapping=map, cmap=cmap,
                vmin=clevels[0], vmax=clevels[1], ax=ax)
        else:
            ba.plot_cslice(
                coord / 1e6, volume='value', region_values=region_values, mapping=map, cmap=cmap,
                vmin=clevels[0], vmax=clevels[1], ax=ax)
            ba.plot_cslice(
                coord / 1e6, volume='boundary', mapping=map, ax=ax, cmap=cmap_bound, vmin=0.01,
                vmax=0.8)

    elif slice == 'sagittal':
        if background == 'image':
            ba.plot_sslice(coord / 1e6, volume='image', mapping=map, ax=ax)
            ba.plot_sslice(
                coord / 1e6, volume='value', region_values=region_values, mapping=map, cmap=cmap,
                vmin=clevels[0], vmax=clevels[1], ax=ax)
        else:
            ba.plot_sslice(
                coord / 1e6, volume='value', region_values=region_values, mapping=map, cmap=cmap,
                vmin=clevels[0], vmax=clevels[1], ax=ax)
            ba.plot_sslice(
                coord / 1e6, volume='boundary', mapping=map, ax=ax, cmap=cmap_bound, vmin=0.01,
                vmax=0.8)

    elif slice == 'horizontal':
        if background == 'image':
            ba.plot_hslice(coord / 1e6, volume='image', mapping=map, ax=ax)
            ba.plot_hslice(
                coord / 1e6, volume='value', region_values=region_values, mapping=map, cmap=cmap,
                vmin=clevels[0], vmax=clevels[1], ax=ax)
        else:
            ba.plot_hslice(
                coord / 1e6, volume='value', region_values=region_values, mapping=map, cmap=cmap,
                vmin=clevels[0], vmax=clevels[1], ax=ax)
            ba.plot_hslice(
                coord / 1e6, volume='boundary', mapping=map, ax=ax, cmap=cmap_bound, vmin=0.01,
                vmax=0.8)

    elif slice == 'top':
        if background == 'image':
            ba.plot_top(volume='image', mapping=map, ax=ax)
            ba.plot_top(
                volume='value', region_values=region_values, mapping=map, cmap=cmap,
                vmin=clevels[0], vmax=clevels[1], ax=ax)
        else:
            ba.plot_top(
                volume='value', region_values=region_values, mapping=map, cmap=cmap,
                vmin=clevels[0], vmax=clevels[1], ax=ax)
            ba.plot_top(
                volume='boundary', mapping=map, ax=ax, cmap=cmap_bound, vmin=0.01, vmax=0.8)

    return fig, ax


def make_bins(signal, timestamps, start_times, stop_times, binsize):
    """Bin a signal into time intervals and compute the mean for each bin.

    For each interval defined by a `start_time` and `stop_time`, this function
    divides the interval into bins of `binsize` and computes the mean of the
    `signal` within each bin.

    Parameters
    ----------
    signal : array_like
        The signal values to be binned.
    timestamps : array_like
        The timestamps corresponding to the signal values.
    start_times : array_like
        The start times of the intervals to bin.
    stop_times : array_like
        The stop times of the intervals to bin. Must be same length as `start_times`.
    binsize : float
        The size of each bin in seconds.

    Returns
    -------
    list of numpy.ndarray
        A list where each element is a NumPy array of the binned signal means for
        the corresponding interval.
    """
    # Loop over start times
    binned_signal = []
    for (start, end) in np.vstack((start_times, stop_times)).T:
        binned_signal.append(binned_statistic(timestamps, signal, bins=int((end-start)*(1/binsize)),
                                              range=(start, end), statistic=np.nanmean)[0])
    return binned_signal


def calculate_peths(
        spike_times, spike_clusters, cluster_ids, align_times, pre_time=0.2,
        post_time=0.5, bin_size=0.025, smoothing=0.025, return_fr=True):
    """Calculate peri-event time histograms (PETHs).

    Computes PETHs for multiple clusters aligned to a set of event times.

    Parameters
    ----------
    spike_times : array_like
        Spike times (in seconds).
    spike_clusters : array_like
        Cluster IDs corresponding to each spike.
    cluster_ids : array_like
        The subset of cluster IDs for which to calculate PETHs.
    align_times : array_like
        Times (in seconds) to align the PETHs to.
    pre_time : float, optional
        Time (in seconds) to include before the align times. Default is 0.2.
    post_time : float, optional
        Time (in seconds) to include after the align times. Default is 0.5.
    bin_size : float, optional
        Width of PETH bins in seconds. Default is 0.025.
    smoothing : float, optional
        Standard deviation (in seconds) of Gaussian kernel for smoothing.
        Set to 0 for no smoothing. Default is 0.025.
    return_fr : bool, optional
        If True, return firing rates (Hz). If False, return spike counts.
        Default is True.

    Returns
    -------
    peths : dict
        A dictionary containing:
        - 'means': PETH means (n_clusters, n_bins).
        - 'stds': PETH standard deviations (n_clusters, n_bins).
        - 'tscale': Time vector for the PETH bins.
        - 'cscale': The cluster IDs.
    binned_spikes : numpy.ndarray
        The binned spike counts or rates for each trial, with shape
        (n_align_times, n_clusters, n_bins).
    """
    # initialize containers
    n_offset = 5 * int(np.ceil(smoothing / bin_size))  # get rid of boundary effects for smoothing
    n_bins_pre = int(np.ceil(pre_time / bin_size)) + n_offset
    n_bins_post = int(np.ceil(post_time / bin_size)) + n_offset
    n_bins = n_bins_pre + n_bins_post
    binned_spikes = np.zeros(shape=(len(align_times), len(cluster_ids), n_bins))

    # build gaussian kernel if requested
    if smoothing > 0:
        w = n_bins - 1 if n_bins % 2 == 0 else n_bins
        window = gaussian(w, std=smoothing / bin_size)
        # half (causal) gaussian filter
        # window[int(np.ceil(w/2)):] = 0
        window /= np.sum(window)
        binned_spikes_conv = np.copy(binned_spikes)

    ids = np.unique(cluster_ids)

    # filter spikes outside of the loop
    idxs = np.bitwise_and(spike_times >= np.min(align_times) - (n_bins_pre + 1) * bin_size,
                          spike_times <= np.max(align_times) + (n_bins_post + 1) * bin_size)
    idxs = np.bitwise_and(idxs, np.isin(spike_clusters, cluster_ids))
    spike_times = spike_times[idxs]
    spike_clusters = spike_clusters[idxs]

    # compute floating tscale
    tscale = np.arange(-n_bins_pre, n_bins_post + 1) * bin_size
    # bin spikes
    for i, t_0 in enumerate(align_times):
        # define bin edges
        ts = tscale + t_0
        # filter spikes
        idxs = np.bitwise_and(spike_times >= ts[0], spike_times <= ts[-1])
        i_spikes = spike_times[idxs]
        i_clusters = spike_clusters[idxs]

        # bin spikes similar to bincount2D: x = spike times, y = spike clusters
        xscale = ts
        xind = (np.floor((i_spikes - np.min(ts)) / bin_size)).astype(np.int64)
        yscale, yind = np.unique(i_clusters, return_inverse=True)
        nx, ny = [xscale.size, yscale.size]
        ind2d = np.ravel_multi_index(np.c_[yind, xind].transpose(), dims=(ny, nx))
        r = np.bincount(ind2d, minlength=nx * ny, weights=None).reshape(ny, nx)

        # store (ts represent bin edges, so there are one fewer bins)
        bs_idxs = np.isin(ids, yscale)
        binned_spikes[i, bs_idxs, :] = r[:, :-1]

        # smooth
        if smoothing > 0:
            idxs = np.where(bs_idxs)[0]
            for j in range(r.shape[0]):
                binned_spikes_conv[i, idxs[j], :] = convolve(
                    r[j, :], window, mode='same', method='auto')[:-1]

    # average
    if smoothing > 0:
        binned_spikes_ = np.copy(binned_spikes_conv)
    else:
        binned_spikes_ = np.copy(binned_spikes)
    if return_fr:
        binned_spikes_ /= bin_size

    peth_means = np.mean(binned_spikes_, axis=0)
    peth_stds = np.std(binned_spikes_, axis=0)

    if smoothing > 0:
        peth_means = peth_means[:, n_offset:-n_offset]
        peth_stds = peth_stds[:, n_offset:-n_offset]
        binned_spikes = binned_spikes_[:, :, n_offset:-n_offset]
        tscale = tscale[n_offset:-n_offset]

    # package output
    tscale = (tscale[:-1] + tscale[1:]) / 2
    peths = dict({'means': peth_means, 'stds': peth_stds, 'tscale': tscale, 'cscale': ids})
    return peths, binned_spikes


def binned_rate_timewarped(spike_times, spike_clusters, trials_df, start='stimOn_times',
        end='firstMovement_times', n_bins=10):
    """Compute time-warped binned firing rates for neurons across trials.

    For each trial, the interval between a start and end event is divided into
    `n_bins` equal-width bins. The firing rate of each neuron is calculated for
    each of these time-warped bins.

    Parameters
    ----------
    spike_times : array_like
        1D array of spike times (in seconds).
    spike_clusters : array_like
        1D array of cluster IDs corresponding to each spike time.
    trials_df : pandas.DataFrame
        DataFrame containing trial information. Must include columns specified
        by the `start` and `end` parameters.
    start : str, optional
        Column name in `trials_df` for the start times of trial intervals.
        Default is 'stimOn_times'.
    end : str, optional
        Column name in `trials_df` for the end times of trial intervals.
        Default is 'firstMovement_times'.
    n_bins : int, optional
        Number of bins to divide each trial interval into. Default is 10.

    Returns
    -------
    binned_rate : numpy.ndarray
        3D array of shape (n_trials, n_neurons, n_bins) containing the firing rates (in Hz).
    neuron_ids : numpy.ndarray
        1D array of unique neuron IDs, corresponding to the second dimension of `binned_rate`.
    """
    # Precompute unique neuron IDs and number of neurons
    neuron_ids = np.unique(spike_clusters)
    n_neurons = neuron_ids.shape[0]
    n_trials = trials_df.shape[0]

    # Initialize binned_rate array
    binned_rate = np.zeros((n_trials, n_neurons, n_bins))

    # Loop over trials
    for i, (start_time, end_time) in enumerate(zip(trials_df[start], trials_df[end])):

        # Define bin edges
        bin_edges = np.linspace(start_time, end_time, n_bins+1)
        bin_width = (end_time - start_time) / n_bins  # Compute bin width

        # Use digitize to assign each spike to a bin
        bin_indices = np.digitize(spike_times, bin_edges, right=True) - 1  # Subtract 1 to get 0-based index

        # Mask to keep only spikes within the trial interval
        valid_spike_mask = (spike_times >= start_time) & (spike_times <= end_time)

        # Filter spike data
        valid_clusters = spike_clusters[valid_spike_mask]
        valid_bins = bin_indices[valid_spike_mask]

        # Use binned_statistic_2d to count spikes per neuron per bin
        spike_counts, _, _, _ = binned_statistic_2d(
            valid_clusters, valid_bins, None, statistic='count',
            bins=[n_neurons, n_bins], range=[[0, n_neurons], [0, n_bins]]
        )

        # Convert to firing rate
        binned_rate[i, :, :] = spike_counts / bin_width

    return binned_rate, neuron_ids


def peri_multiple_events_time_histogram(
        spike_times, spike_clusters, events, event_ids, cluster_id,
        t_before=0.2, t_after=0.5, bin_size=0.025, smoothing=0.025, as_rate=True,
        include_raster=False, error_bars='sem', ax=None,
        pethline_kwargs=[{'color': 'blue', 'lw': 2}, {'color': 'red', 'lw': 2}],
        errbar_kwargs=[{'color': 'blue', 'alpha': 0.5}, {'color': 'red', 'alpha': 0.5}],
        raster_kwargs=[{'color': 'blue', 'lw': 0.5}, {'color': 'red', 'lw': 0.5}],
        eventline_kwargs={'color': 'black', 'alpha': 0.5},
        labels=None, include_legend=False, **kwargs):
    """Plot PETHs for a single neuron aligned to multiple event types.

    This function plots the mean firing rate of a neuron aligned to different
    sets of events, distinguished by `event_ids`. It can optionally include
    error bars and a spike raster plot.

    Parameters
    ----------
    spike_times : array_like
        Spike times (in seconds).
    spike_clusters : array-like
        Cluster identities for each spike.
    events : array-like
        Times to align the histogram(s) to.
    event_ids : array-like
        Identities of events, used to group `events`.
    cluster_id : int
        Identity of the cluster for which to plot a PETH.
    t_before : float, optional
        Time before event to plot (in seconds). Default is 0.2.
    t_after : float, optional
        Time after event to plot (in seconds). Default is 0.5.
    bin_size : float, optional
        Width of histogram bins (in seconds). Default is 0.025.
    smoothing : float, optional
        Sigma of Gaussian smoothing kernel (in seconds). Default is 0.025.
    as_rate : bool, optional
        If True, y-axis is firing rate (Hz). If False, it's spike counts. Default is True.
    include_raster : bool, optional
        If True, adds a raster plot below the PETH. Default is False.
    error_bars : {'std', 'sem', 'none'}, optional
        Type of error bars to plot: 'std' for standard deviation, 'sem' for
        standard error of the mean. Default is 'sem'.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure and axes are created.
    pethline_kwargs : list of dict, optional
        List of dictionaries with keyword arguments for the PETH line plots (e.g., `color`, `lw`),
        one for each unique event ID.
    errbar_kwargs : list of dict, optional
        List of dictionaries with keyword arguments for the error bar fill.
    raster_kwargs : list of dict, optional
        List of dictionaries with keyword arguments for the raster plot lines.
    eventline_kwargs : dict, optional
        Dictionary with keyword arguments for the vertical line at time 0.
    labels : list of str, optional
        List of labels for each event type to be used in the legend. Default is None.
    include_legend : bool, optional
        Whether to include a legend in the plot. Default is False.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object with the plot.
    """

    # Check to make sure if we fail, we fail in an informative way
    if not len(spike_times) == len(spike_clusters):
        raise ValueError('Spike times and clusters are not of the same shape')
    if len(events) == 1:
        raise ValueError('Cannot make a PETH with only one event.')
    if error_bars not in ('std', 'sem', 'none'):
        raise ValueError('Invalid error bar type was passed.')
    if not all(np.isfinite(events)):
        raise ValueError('There are NaN or inf values in the list of events passed. '
                         ' Please remove non-finite data points and try again.')

    # Construct an axis object if none passed
    if ax is None:
        plt.figure()
        ax = plt.gca()
    # Plot the curves and add error bars
    mean_max, bars_max = [], []
    for i, event_id in enumerate(np.unique(event_ids)):
        # Compute peths
        peths, binned_spikes = singlecell.calculate_peths(spike_times, spike_clusters, [cluster_id],
                                                          events[event_ids == event_id], t_before,
                                                          t_after, bin_size, smoothing, as_rate)
        mean = peths.means[0, :]
        label = labels[i] if labels is not None else None
        ax.plot(peths.tscale, mean, label=label, **pethline_kwargs[i])
        if error_bars == 'std':
            bars = peths.stds[0, :]
        elif error_bars == 'sem':
            bars = peths.stds[0, :] / np.sqrt(np.sum(event_ids == event_id))
        else:
            bars = np.zeros_like(mean)
        if error_bars != 'none':
            ax.fill_between(peths.tscale, mean - bars, mean + bars, **errbar_kwargs[i])
        mean_max.append(mean.max())
        bars_max.append(bars[mean.argmax()])

    # Plot the event marker line. Extends to 5% higher than max value of means plus any error bar.
    plot_edge = (np.max(mean_max) + bars_max[np.argmax(mean_max)]) * 1.05
    ax.vlines(0., 0., plot_edge, **eventline_kwargs)
    # Set the limits on the axes to t_before and t_after. Either set the ylim to the 0 and max
    # values of the PETH, or if we want to plot a spike raster below, create an equal amount of
    # blank space below the zero where the raster will go.
    ax.set_xlim([-t_before, t_after])
    ax.set_ylim([-plot_edge if include_raster else 0., plot_edge])
    # Put y ticks only at min, max, and zero
    if mean.min() != 0:
        ax.set_yticks([0, mean.min(), mean.max()])
    else:
        ax.set_yticks([0., mean.max()])
    # Move the x axis line from the bottom of the plotting space to zero if including a raster,
    # Then plot the raster
    if include_raster:
        ax.axhline(0., color='black', lw=0.5)
        tickheight = plot_edge / len(events)  # How much space per trace
        tickedges = np.arange(0., -plot_edge - 1e-5, -tickheight)
        clu_spks = spike_times[spike_clusters == cluster_id]
        ii = 0
        for k, event_id in enumerate(np.unique(event_ids)):
            for i, t in enumerate(events[event_ids == event_id]):
                idx = np.bitwise_and(clu_spks >= t - t_before, clu_spks <= t + t_after)
                event_spks = clu_spks[idx]
                ax.vlines(event_spks - t, tickedges[i + ii + 1], tickedges[i + ii],
                          **raster_kwargs[k])
            ii += np.sum(event_ids == event_id)
        ax.set_ylabel('Firing Rate' if as_rate else 'Number of spikes', y=0.75)
    else:
        ax.set_ylabel('Firing Rate' if as_rate else 'Number of spikes')
    if include_legend:
        ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('Time (s) after event')
    return ax


def calculate_mi(spike_counts_A, spike_counts_B):
    """Calculate the mutual information between two discrete variables.

    Parameters
    ----------
    spike_counts_A : array_like
        A 1D array of observations for the first variable.
    spike_counts_B : array_like
        A 1D array of observations for the second variable. Must be the same
        length as `spike_counts_A`.

    Returns
    -------
    float
        The mutual information in bits.

    Notes
    -----
    The number of bins for the 2D histogram is set to the number of observations,
    which may not be optimal for continuous data. This function is best suited for
    discrete data where `spike_counts_A` and `spike_counts_B` represent counts or categories.
    """
    # Calculate joint probability distribution
    joint_hist, _, _ = np.histogram2d(spike_counts_A, spike_counts_B, bins=spike_counts_A.shape[0])
    joint_prob = joint_hist / np.sum(joint_hist)

    # Calculate marginal probabilities
    marg_prob_A = np.sum(joint_prob, axis=1)
    marg_prob_B = np.sum(joint_prob, axis=0)

    # Calculate mutual information
    non_zero_indices = joint_prob > 0
    mutual_info = np.sum(joint_prob[non_zero_indices] * \
                         np.log2(joint_prob[non_zero_indices] / \
                                 (np.outer(marg_prob_A, marg_prob_B))[non_zero_indices]))

    return mutual_info


def get_dlc_XYs_old(one, eid, view='left', likelihood_thresh=0.9):
    """Load DeepLabCut (DLC) tracking data from a .pqt file (legacy).

    .. deprecated::
        Use `get_dlc_XYs` instead, which uses the more modern `SessionLoader`.

    This function loads DLC coordinates and camera timestamps for a given session.
    It filters coordinates based on a likelihood threshold.

    Parameters
    ----------
    one : one.api.ONE
        An initialized ONE instance.
    eid : str
        The experiment ID of the session.
    view : {'left', 'right', 'body'}, optional
        The camera view to load. Default is 'left'.
    likelihood_thresh : float, optional
        The likelihood threshold below which coordinates are set to NaN. Default is 0.9.

    Returns
    -------
    times : numpy.ndarray or None
        An array of camera frame timestamps. None if timestamps can't be loaded.
    XYs : dict or None
        A dictionary where keys are tracked body parts and values are (n_frames, 2)
        numpy arrays of (x, y) coordinates. None if DLC data can't be loaded.
    """
    ses_details = one.get_details(eid)
    subject = ses_details['subject']
    date = ses_details['date']
    _, save_path = paths()
    if f'alf/_ibl_{view}Camera.times.npy' in one.list_datasets(eid):
        times = one.load_dataset(eid, f'_ibl_{view}Camera.times.npy')
    elif isfile(join(save_path, 'CameraTimestamps', f'{subject}_{date}_{view}Camera.npy')):
        times = np.load(join(save_path, 'CameraTimestamps', f'{subject}_{date}_{view}Camera.npy'))
    else:
        print('could not load camera timestamps')
        return None, None
    try:
        cam = one.load_dataset(eid, '_ibl_%sCamera.dlc.pqt' % view)
    except KeyError:
        print('not all dlc data available')
        return None, None
    points = np.unique(['_'.join(x.split('_')[:-1]) for x in cam.keys()])
    # Set values to nan if likelyhood is too low # for pqt: .to_numpy()
    XYs = {}
    for point in points:
        x = np.ma.masked_where(cam[point + '_likelihood'] < likelihood_thresh, cam[point + '_x'])
        x = x.filled(np.nan)
        y = np.ma.masked_where(cam[point + '_likelihood'] < likelihood_thresh, cam[point + '_y'])
        y = y.filled(np.nan)
        XYs[point] = np.array([x, y]).T
    return times, XYs


def get_dlc_XYs(one, eid, view='left'):
    """Load DeepLabCut (DLC) tracking data using SessionLoader.

    This function uses `brainbox.io.one.SessionLoader` to load pose estimation data
    and returns it in a format compatible with older functions.

    Parameters
    ----------
    one : one.api.ONE
        An initialized ONE instance.
    eid : str
        The experiment ID of the session.
    view : {'left', 'right', 'body'}, optional
        The camera view to load. Default is 'left'.

    Returns
    -------
    times : numpy.ndarray
        An array of camera frame timestamps.
    XYs : dict
        A dictionary where keys are tracked body parts and values are (n_frames, 2)
        numpy arrays of (x, y) coordinates.
    """
    # Load in DLC
    sl = SessionLoader(one=one, eid=eid)
    sl.load_pose(views=[view])
    dlc_df = sl.pose[f'{view}Camera']

    # Transform to dict for backwards compatibility reasons
    X = [dlc_df[i].values for i in dlc_df.columns if i[-1] == 'x']
    Y = [dlc_df[i].values for i in dlc_df.columns if i[-1] == 'y']
    key_names = [i[:-2] for i in dlc_df.columns if i[-1:] == 'x']

    XYs = {}
    for i, this_key in enumerate(key_names):
        XYs[this_key] = np.array([X[i], Y[i]]).T

    return dlc_df['times'].values, XYs


def smooth_interpolate_signal_sg(signal, window=31, order=3, interp_kind='cubic'):
    """Smooth and interpolate a signal using a Savitzky-Golay filter.

    This function handles NaNs in the input signal by applying a non-uniform
    Savitzky-Golay filter (`non_uniform_savgol`) to the valid data points,
    and then interpolating the missing (NaN) values.

    Parameters
    ----------
    signal : np.ndarray
        1D array representing the signal to be smoothed. May contain NaNs.
    window : int, optional
        The length of the filter window for the Savitzky-Golay filter. Must be an
        odd integer. Default is 31.
    order : int, optional
        The order of the polynomial used to fit the samples. Default is 3.
    interp_kind : str, optional
        The kind of interpolation to use for filling NaNs, passed to
        `scipy.interpolate.interp1d`. Default is 'cubic'.

    Returns
    -------
    np.ndarray
        The smoothed and interpolated signal.
    """
    signal_noisy_w_nans = np.copy(signal)
    timestamps = np.arange(signal_noisy_w_nans.shape[0])
    good_idxs = np.where(~np.isnan(signal_noisy_w_nans))[0]
    # perform savitzky-golay filtering on non-nan points
    signal_smooth_nonans = non_uniform_savgol(
        timestamps[good_idxs], signal_noisy_w_nans[good_idxs], window=window, polynom=order)
    signal_smooth_w_nans = np.copy(signal_noisy_w_nans)
    signal_smooth_w_nans[good_idxs] = signal_smooth_nonans
    # interpolate nan points
    interpolater = interp1d(
        timestamps[good_idxs], signal_smooth_nonans, kind=interp_kind, fill_value='extrapolate')

    signal = interpolater(timestamps)

    return signal


def non_uniform_savgol(x, y, window, polynom):
    """Apply a Savitzky-Golay filter to non-uniformly spaced data.

    This implementation is based on the method described on StackExchange for data
    with non-uniform spacing.
    Source: https://dsp.stackexchange.com/questions/1676/savitzky-golay-smoothing-filter-for-not-equally-spaced-data

    Parameters
    ----------
    x : array_like
        The x-coordinates of the data points (timestamps).
    y : array_like
        The y-coordinates of the data points (signal values). Must have same length as x.
    window : int
        The length of the filter window. Must be an odd integer.
    polynom : int
        The order of the polynomial to use. Must be less than `window`.

    Returns
    -------
    np.ndarray
        The smoothed y values.
    """

    if len(x) != len(y):
        raise ValueError('"x" and "y" must be of the same size')

    if len(x) < window:
        raise ValueError('The data size must be larger than the window size')

    if type(window) is not int:
        raise TypeError('"window" must be an integer')

    if window % 2 == 0:
        raise ValueError('The "window" must be an odd integer')

    if type(polynom) is not int:
        raise TypeError('"polynom" must be an integer')

    if polynom >= window:
        raise ValueError('"polynom" must be less than "window"')

    half_window = window // 2
    polynom += 1

    # Initialize variables
    A = np.empty((window, polynom))  # Matrix
    tA = np.empty((polynom, window))  # Transposed matrix
    t = np.empty(window)  # Local x variables
    y_smoothed = np.full(len(y), np.nan)

    # Start smoothing
    for i in range(half_window, len(x) - half_window, 1):
        # Center a window of x values on x[i]
        for j in range(0, window, 1):
            t[j] = x[i + j - half_window] - x[i]

        # Create the initial matrix A and its transposed form tA
        for j in range(0, window, 1):
            r = 1.0
            for k in range(0, polynom, 1):
                A[j, k] = r
                tA[k, j] = r
                r *= t[j]

        # Multiply the two matrices
        tAA = np.matmul(tA, A)

        # Invert the product of the matrices
        tAA = np.linalg.inv(tAA)

        # Calculate the pseudoinverse of the design matrix
        coeffs = np.matmul(tAA, tA)

        # Calculate c0 which is also the y value for y[i]
        y_smoothed[i] = 0
        for j in range(0, window, 1):
            y_smoothed[i] += coeffs[0, j] * y[i + j - half_window]

        # If at the end or beginning, store all coefficients for the polynom
        if i == half_window:
            first_coeffs = np.zeros(polynom)
            for j in range(0, window, 1):
                for k in range(polynom):
                    first_coeffs[k] += coeffs[k, j] * y[j]
        elif i == len(x) - half_window - 1:
            last_coeffs = np.zeros(polynom)
            for j in range(0, window, 1):
                for k in range(polynom):
                    last_coeffs[k] += coeffs[k, j] * y[len(y) - window + j]

    # Interpolate the result at the left border
    for i in range(0, half_window, 1):
        y_smoothed[i] = 0
        x_i = 1
        for j in range(0, polynom, 1):
            y_smoothed[i] += first_coeffs[j] * x_i
            x_i *= x[i] - x[half_window]

    # Interpolate the result at the right border
    for i in range(len(x) - half_window, len(x), 1):
        y_smoothed[i] = 0
        x_i = 1
        for j in range(0, polynom, 1):
            y_smoothed[i] += last_coeffs[j] * x_i
            x_i *= x[i] - x[-half_window - 1]

    return y_smoothed


def get_pupil_diameter(XYs):
    """Estimate pupil diameter from four tracked points.

    This function computes the pupil diameter in multiple ways and returns the median
    estimate. It calculates the direct distance between top-bottom and left-right
    markers, and also estimates the diameter from pairs of adjacent markers assuming
    a circular pupil.

    Author: Michael Schartner

    Parameters
    ----------
    XYs : dict
        A dictionary of DLC coordinates. Must contain keys: 'pupil_top_r',
        'pupil_bottom_r', 'pupil_left_r', 'pupil_right_r'. Each value should be
        an (n_frames, 2) array of (x, y) coordinates.

    Returns
    -------
    np.ndarray
        The estimated pupil diameter for each time point (n_frames,).
    """

    # direct diameters
    t = XYs['pupil_top_r'][:, :2]
    b = XYs['pupil_bottom_r'][:, :2]
    l = XYs['pupil_left_r'][:, :2]
    r = XYs['pupil_right_r'][:, :2]

    def distance(p1, p2):
        return ((p1[:, 0] - p2[:, 0]) ** 2 + (p1[:, 1] - p2[:, 1]) ** 2) ** 0.5

    # get diameter via top-bottom and left-right
    ds = []
    ds.append(distance(t, b))
    ds.append(distance(l, r))

    def dia_via_circle(p1, p2):
        # only valid for non-crossing edges
        u = distance(p1, p2)
        return u * (2 ** 0.5)

    # estimate diameter via circle assumption
    for side in [[t, l], [t, r], [b, l], [b, r]]:
        ds.append(dia_via_circle(side[0], side[1]))
    diam = np.nanmedian(ds, axis=0)

    return diam


def get_raw_smooth_pupil_diameter(XYs):
    """Compute raw and smoothed pupil diameter from DLC coordinates.

    This function first computes a raw pupil diameter estimate using `get_pupil_diameter`.
    It then applies a two-pass smoothing and outlier removal procedure:
    1. Smooth the raw signal to identify outliers.
    2. Remove outliers from the raw signal.
    3. Smooth the cleaned signal again.
    Long gaps of NaNs are not interpolated.

    Parameters
    ----------
    XYs : dict
        A dictionary of DLC coordinates passed to `get_pupil_diameter`.

    Returns
    -------
    diam0 : np.ndarray
        The raw, noisy pupil diameter estimate.
    diam_sm1 : np.ndarray
        The final smoothed and cleaned pupil diameter estimate.
    """
    # threshold (in standard deviations) beyond which a point is labeled as an outlier
    std_thresh = 5

    # threshold (in seconds) above which we will not interpolate nans, but keep them
    # (for long stretches interpolation may not be appropriate)
    nan_thresh = 1

    # compute framerate of camera
    fr = 60  # set by hardware
    window = 61  # works well empirically

    # compute diameter using raw values of 4 markers (will be noisy and have missing data)
    diam0 = get_pupil_diameter(XYs)

    # run savitzy-golay filter on non-nan timepoints to denoise
    diam_sm0 = smooth_interpolate_signal_sg(
        diam0, window=window, order=3, interp_kind='linear')

    # find outliers, set to nan
    errors = diam0 - diam_sm0
    std = np.nanstd(errors)
    diam1 = np.copy(diam0)
    diam1[(errors < (-std_thresh * std)) | (errors > (std_thresh * std))] = np.nan
    # run savitzy-golay filter again on (possibly reduced) non-nan timepoints to denoise
    diam_sm1 = smooth_interpolate_signal_sg(
        diam1, window=window, order=3, interp_kind='linear')

    # don't interpolate long strings of nans
    t = np.diff(1 * np.isnan(diam1))
    begs = np.where(t == 1)[0]
    ends = np.where(t == -1)[0]
    if begs.shape[0] > ends.shape[0]:
        begs = begs[:ends.shape[0]]
    for b, e in zip(begs, ends):
        if (e - b) > (fr * nan_thresh):
            diam_sm1[(b + 1):(e + 1)] = np.nan  # offset by 1 due to earlier diff

    # diam_sm1 is the final smoothed pupil diameter estimate
    return diam0, diam_sm1


def SNR(diam0, diam_sm1):
    """Calculate the Signal-to-Noise Ratio (SNR) of a signal.

    The SNR is computed as the ratio of the variance of the smoothed signal (the "signal")
    to the variance of the residual (raw - smoothed, the "noise").

    Parameters
    ----------
    diam0 : np.ndarray
        The raw signal, which may contain NaNs.
    diam_sm1 : np.ndarray
        The smoothed signal, which may contain NaNs.

    Returns
    -------
    float
        The computed SNR.
    """
    # compute signal to noise ratio between raw and smooth dia
    good_idxs = np.where(~np.isnan(diam_sm1) & ~np.isnan(diam0))[0]
    snr = (np.var(diam_sm1[good_idxs]) /
           np.var(diam_sm1[good_idxs] - diam0[good_idxs]))

    return snr


def query_opto_sessions(subject, include_ephys=False, one=None):
    """Query optogenetic sessions for a given subject.

    Parameters
    ----------
    subject : str
        The name of the subject.
    include_ephys : bool, optional
        If True, queries for all opto task protocols (`_iblrig_tasks_opto_`).
        If False, queries only for the biased choice world opto task
        (`_iblrig_tasks_opto_biasedChoiceWorld`). Default is False.
    one : one.api.ONE, optional
        An initialized ONE instance. If None, a new one is created.

    Returns
    -------
    list of str
        A list of session EIDs for the matching sessions.
    """
    one = one or ONE()
    if include_ephys:
        sessions = one.alyx.rest('sessions', 'list', subject=subject,
                                 task_protocol='_iblrig_tasks_opto_')
    else:
        sessions = one.alyx.rest('sessions', 'list', subject=subject,
                                 task_protocol='_iblrig_tasks_opto_biasedChoiceWorld')
    return [sess['url'][-36:] for sess in sessions]


def behavioral_criterion(eids, min_perf=0.7, min_trials=200, max_rt=0.7, return_excluded=False,
                         verbose=True, one=None):
    """Filter sessions based on behavioral performance criteria.

    This function iterates through a list of session EIDs and keeps only those that
    meet specified criteria for performance, number of trials, and reaction time.

    Parameters
    ----------
    eids : list of str
        A list of session EIDs to filter.
    min_perf : float, optional
        Minimum performance on high-contrast trials (contrast=1). Default is 0.7.
    min_trials : int, optional
        Minimum total number of trials in the session. Default is 200.
    max_rt : float, optional
        Maximum median reaction time (feedback_time - goCue_time). Default is 0.7s.
    return_excluded : bool, optional
        If True, also returns a list of the excluded EIDs. Default is False.
    verbose : bool, optional
        If True, prints information about excluded sessions. Default is True.
    one : one.api.ONE, optional
        An initialized ONE instance. If None, a new one is created.

    Returns
    -------
    list of str or (list of str, list of str)
        If `return_excluded` is False, returns a list of EIDs that passed the criteria.
        If `return_excluded` is True, returns a tuple of (passed_eids, excluded_eids).
    """
    if one is None:
        one = ONE()
    use_eids, excl_eids = [], []
    for j, eid in enumerate(eids):
        try:
            trials = load_trials(eid, one=one)
            trials['rt'] = trials['feedback_times'] - trials['goCue_times']
            perf = (np.sum(trials.loc[np.abs(trials['signed_contrast']) == 1, 'feedbackType'] == 1)
                    / trials[np.abs(trials['signed_contrast']) == 1].shape[0])
            details = one.get_details(eid)
            if (perf > min_perf) & (trials.shape[0] > min_trials) & (trials['rt'].median() < max_rt):
                use_eids.append(eid)
            else:
                if verbose:
                    print('%s %s excluded (perf: %.2f, n_trials: %d, rt: %.2f)'
                          % (details['subject'], details['start_time'][:10], perf, trials.shape[0],
                             trials['rt'].median()))
                excl_eids.append(eid)
        except Exception:
            if verbose:
                print('Could not load session %s' % eid)
    if return_excluded:
        return use_eids, excl_eids
    else:
        return use_eids


def fit_psychfunc(stim_levels, n_trials, proportion, transform_slope=False):
    """Fit a psychometric function to behavioral data.

    This function uses `psychofit.mle_fit_psycho` to fit a 4-parameter psychometric
    function (bias, threshold, low-lapse, high-lapse) to choice data.

    Parameters
    ----------
    stim_levels : array_like
        The stimulus levels (e.g., contrast). Assumed to be in range [-100, 100].
    n_trials : array_like
        The number of trials at each stimulus level.
    proportion : array_like
        The proportion of 'rightward' choices at each stimulus level.
    transform_slope : bool, optional
        If True, transforms the threshold parameter to be a slope parameter (1/threshold * 100).
        Default is False.

    Returns
    -------
    numpy.ndarray
        A 4-element array containing the fitted parameters:
        [bias, threshold/slope, lapse_low, lapse_high].
    """
    import psychofit as psy
    assert (stim_levels.shape == n_trials.shape == proportion.shape)
    if stim_levels.max() <= 1:
        stim_levels = stim_levels * 100

    # Fit psychometric function
    pars, _ = psy.mle_fit_psycho(np.vstack((stim_levels, n_trials, proportion)),
                                 P_model='erf_psycho_2gammas',
                                 parstart=np.array([0, 20, 0.05, 0.05]),
                                 parmin=np.array([-100, 5, 0, 0]),
                                 parmax=np.array([100, 100, 1, 1]))

    # Transform the slope paramter such that large values = steeper slope
    if transform_slope:
        pars[1] = (1/pars[1])*100

    return pars


def plot_psychometric(trials, ax, color='b', linestyle='solid', fraction=True):
    """Plot a psychometric function for a set of trials.

    This function fits a psychometric curve to the provided trial data using
    `fit_psychfunc` and plots the fitted curve along with the raw data points
    (mean choice proportion with standard error bars). It also handles a
    broken x-axis for 100% contrast trials.

    Parameters
    ----------
    trials : pandas.DataFrame
        A trials DataFrame, must contain 'signed_contrast' and 'right_choice' columns.
    ax : matplotlib.axes.Axes
        The axes on which to plot.
    color : str, optional
        The color for the plot elements. Default is 'b'.
    linestyle : str, optional
        The linestyle for the fitted curve. Default is 'solid'.
    fraction : bool, optional
        If True, the y-axis is labeled as a fraction. If False, it's labeled as a percentage.
        Default is True.
    """
    import psychofit as psy
    if trials['signed_contrast'].max() <= 1:
        trials['signed_contrast'] = trials['signed_contrast'] * 100

    stim_levels = np.sort(trials['signed_contrast'].unique())
    pars = fit_psychfunc(stim_levels, trials.groupby('signed_contrast').size(),
                         trials.groupby('signed_contrast').mean()['right_choice'])

    # plot psychfunc
    sns.lineplot(x=np.arange(-27, 27), y=psy.erf_psycho_2gammas(pars, np.arange(-27, 27)),
                 ax=ax, color=color, linestyle=linestyle)

    # plot psychfunc: -100, +100
    sns.lineplot(x=np.arange(-36, -31), y=psy.erf_psycho_2gammas(pars, np.arange(-103, -98)),
                 ax=ax, color=color, linestyle=linestyle)
    sns.lineplot(x=np.arange(31, 36), y=psy.erf_psycho_2gammas(pars, np.arange(98, 103)),
                 ax=ax, color=color, linestyle=linestyle)

    # now break the x-axis
    trials['signed_contrast'].replace(-100, -35)
    trials['signed_contrast'].replace(100, 35)

    # plot datapoints with errorbars on top
    sns.lineplot(x=trials['signed_contrast'], y=trials['right_choice'], ax=ax,
                 **{**{'err_style': "bars",
                       'linewidth': 0, 'linestyle': 'None', 'mew': 0.5,
                       'marker': 'o', 'errorbar': 'se'}}, color=color)

    ax.set(xticks=[-35, -25, -12.5, 0, 12.5, 25, 35], xlim=[-40, 40], ylim=[0, 1.02],
           xlabel='Contrast (%)')
    ax.set_xticklabels(['100', '25', '12.5', '0', '12.5', '25', '100'])
    if fraction:
        ax.set(yticks=[0, 0.25, 0.5, 0.75, 1], yticklabels=['0', '.25', '.5', '.75', '1'])
        ax.set_ylabel('Fraction of rightward choices', labelpad=1)
    else:
        ax.set(yticks=[0, 0.25, 0.5, 0.75, 1], yticklabels=['0', '25', '50', '75', '100'],
               ylabel='Rightward choices (%)')
    # break_xaxis()


def break_xaxis(y=-0.004, **kwargs):
    """Add break marks to the x-axis of a psychometric plot.

    This is a helper function for `plot_psychometric` to indicate a discontinuity
    on the x-axis, typically between +/-25% and +/-100% contrast.

    Parameters
    ----------
    y : float, optional
        The y-coordinate at which to draw the break marks. Default is -0.004.
    **kwargs
        Additional keyword arguments are passed to `plt.text`, but are currently unused.
    """
    # axisgate: show axis discontinuities with a quick hack
    # https://twitter.com/StevenDakin/status/1313744930246811653?s=19
    # first, white square for discontinuous axis
    plt.text(-30, y, '-', fontsize=14, fontweight='bold',
             horizontalalignment='center', verticalalignment='center',
             color='w')
    plt.text(30, y, '-', fontsize=14, fontweight='bold',
             horizontalalignment='center', verticalalignment='center',
             color='w')

    # put little dashes to cut axes
    plt.text(-30, y, '/ /', horizontalalignment='center',
             verticalalignment='center', fontsize=12, fontweight='bold')
    plt.text(30, y, '/ /', horizontalalignment='center',
             verticalalignment='center', fontsize=12, fontweight='bold')


def get_bias(trials):
    """Calculate task bias from trial data.

    The bias is calculated by fitting separate psychometric functions to trials from
    the 80/20 and 20/80 probability blocks. The bias is defined as the difference
    in the choice probability at 0% contrast between these two blocks.

    Parameters
    ----------
    trials : pandas.DataFrame
        A trials DataFrame containing 'probabilityLeft', 'signed_contrast', and
        'right_choice' columns.

    Returns
    -------
    float
        The calculated bias value. Returns np.nan if there are no trials.
    """
    import psychofit as psy
    if len(trials) == 0:
        return np.nan

    # 20/80 blocks
    these_trials = trials[trials['probabilityLeft'] == 0.2]
    stim_levels = np.sort(these_trials['signed_contrast'].unique())
    pars_right = fit_psychfunc(stim_levels, these_trials.groupby('signed_contrast').size(),
                               these_trials.groupby('signed_contrast').mean()['right_choice'])
    bias_right = psy.erf_psycho_2gammas(pars_right, 0)

    # 80/20 blocks
    these_trials = trials[trials['probabilityLeft'] == 0.8]
    stim_levels = np.sort(these_trials['signed_contrast'].unique())
    pars_left = fit_psychfunc(stim_levels, these_trials.groupby('signed_contrast').size(),
                              these_trials.groupby('signed_contrast').mean()['right_choice'])
    bias_left = psy.erf_psycho_2gammas(pars_left, 0)

    return bias_right - bias_left


def fit_glm(behav, prior_blocks=True, opto_stim=False, folds=3):
    """Fit a logistic regression model (GLM) to behavioral choice data.

    This function constructs a design matrix with regressors for previous choice,
    block bias (prior), and stimulus contrast, separated for optogenetic stimulation
    and no-stimulation trials. It fits a logistic regression model to predict the
    current choice and evaluates its accuracy using k-fold cross-validation.

    Parameters
    ----------
    behav : pandas.DataFrame
        A DataFrame containing behavioral data for a session. Must include columns:
        'signed_contrast', 'laser_stimulation', 'previous_choice', 'block_id',
        'stim_side', 'trial_feedback_type', 'choice'.
    prior_blocks : bool, optional
        This parameter is unused.
    opto_stim : bool, optional
        This parameter is unused.
    folds : int, optional
        The number of folds for cross-validation to calculate model accuracy.
        Default is 3.

    Returns
    -------
    pandas.DataFrame
        A single-row DataFrame containing the fitted model parameters (coefficients)
        as columns, along with 'pseudo_rsq', 'condition_number', and cross-validated
        'accuracy'.
    """
    # drop trials with contrast-level 50, only rarely present (should not be its own regressor)
    behav = behav[np.abs(behav.signed_contrast) != 50]

    # create extra columns for 5-HT and no 5-HT trials
    behav['previous_choice_0'] = behav['previous_choice'] * (1 - behav['laser_stimulation'])
    behav['previous_choice_1'] = behav['previous_choice'] * behav['laser_stimulation']
    behav['prior_0'] = behav['block_id'] * (1 - behav['laser_stimulation'])
    behav['prior_1'] = behav['block_id'] * behav['laser_stimulation']

    # Loop through unique contrast values
    for contrast in behav['contrast'].unique():
        behav[f'{int(contrast)}_0'] = np.zeros(behav.shape[0])
        behav.loc[behav['contrast'] == contrast, f'{int(contrast)}_0'] = (
            behav.loc[behav['contrast'] == contrast, 'stim_side']
            * (1 - behav['laser_stimulation']))
        behav[f'{int(contrast)}_1'] = np.zeros(behav.shape[0])
        behav.loc[behav['contrast'] == contrast, f'{int(contrast)}_1'] = (
            behav.loc[behav['contrast'] == contrast, 'stim_side']
            * behav['laser_stimulation'])

    # drop NaNs
    behav = behav.dropna(subset=['trial_feedback_type', 'choice', 'previous_choice']).reset_index(drop=True)

    # create input for GLM (0 = no 5HT, 1 = 5HT)
    endog = pd.DataFrame(data={'choice': behav['choice']})
    exog = behav[[
        'previous_choice_0', 'previous_choice_1', 'prior_0', 'prior_1',
        '6_0', '6_1', '12_0', '12_1', '25_0', '25_1', '100_0', '100_1']].copy()
    exog['bias'] = 1
    exog['opto'] = behav['laser_stimulation']

    # recode choices for logistic regression
    endog['choice'] = endog['choice'].map({-1:0, 1:1})

    # NOW FIT THIS WITH STATSMODELS - ignore NaN choices
    logit_model = sm.Logit(endog, exog)
    res = logit_model.fit_regularized(disp=False) # run silently

    # what do we want to keep?
    params = pd.DataFrame(res.params).T
    params['pseudo_rsq'] = res.prsquared # https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.LogitResults.prsquared.html?highlight=pseudo
    params['condition_number'] = np.linalg.cond(exog)

    # ===================================== #
    # ADD MODEL ACCURACY - cross-validate

    kf = KFold(n_splits=folds, shuffle=True)
    acc = np.array([])
    for train, test in kf.split(endog):
        X_train, X_test, y_train, y_test = exog.loc[train], exog.loc[test], \
                                           endog.loc[train], endog.loc[test]
        # fit again
        logit_model = sm.Logit(y_train, X_train)
        res = logit_model.fit_regularized(disp=False)  # run silently

        # compute the accuracy on held-out data [from Luigi]:
        # suppose you are predicting Pr(Left), let's call it p,
        # the % match is p if the actual choice is left, or 1-p if the actual choice is right
        # if you were to simulate it, in the end you would get these numbers
        y_test['pred'] = res.predict(X_test)
        y_test.loc[y_test['choice'] == 0, 'pred'] = 1 - y_test.loc[y_test['choice'] == 0, 'pred']
        acc = np.append(acc, y_test['pred'].mean())

    # average prediction accuracy over the K folds
    params['accuracy'] = np.mean(acc)


    return params  # wide df