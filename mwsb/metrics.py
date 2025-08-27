#!/usr/bin/python

"""
Set of methods developed to follow the Round Robin workbench for SAR WV SWH predictions defined in CCE sea-state ESA project
https://climate.esa.int/media/documents/Sea_State_cci_PVASR_v3.0-signed.pdf
"""
import numpy as np
import os
import pandas as pd
import copy
from yaml import load
from yaml import CLoader as Loader
import logging

CATEGORIES = {
    'low': (0, 1.5),
    'medium': (1.5, 3.0),
    'high': (3.0, 6.0),
    'very high': (6.0, 30.0)
}

MIN_MAX_BOUNDARIES_4_NORMALIZATION = {
    'low': {
        'rmse': {'min': 0.2, 'max': 0.6},
        'no-data-pct': {'min': 0.0, 'max': 30.0},
        'outlier-pct': {'min': 0.0, 'max': 5.0},
        'median_abs_bias': {'min': 0.0, 'max': 0.5},
    },
    'medium': {
        'rmse': {'min': 0.2, 'max': 0.6},
        'no-data-pct': {'min': 0.0, 'max': 30.0},
        'outlier-pct': {'min': 0.0, 'max': 5.0},
        'median_abs_bias': {'min': 0.0, 'max': 0.5},
    },
    'high': {
        'rmse': {'min': 0.2, 'max': 1.0},
        'no-data-pct': {'min': 0.0, 'max': 50.0},
        'outlier-pct': {'min': 0.0, 'max': 10.0},
        'median_abs_bias': {'min': 0.0, 'max': 0.5},
    },
    'very high': {
        'rmse': {'min': 0.2, 'max': 1.0},
        'no-data-pct': {'min': 0.0, 'max': 60.0},
        'outlier-pct': {'min': 0.0, 'max': 10.0},
        'median_abs_bias': {'min': 0.0, 'max': 0.5},
    }
}

WEIGHTINGS_FACTOR_K = {
    'low': {
        'rmse': {'cmems': 0.12, 'ndbc': 0.12},
        'no-data-pct': {'cmems': 0.12, 'ndbc': 0.12},
        'outlier-pct': {'cmems': 0.12, 'ndbc': 0.12},
        'median_abs_bias': {'cmems': 0.12, 'ndbc': 0.12},
    },
    'medium': {
        'rmse': {'cmems': 0.62, 'ndbc': 0.62},
        'no-data-pct': {'cmems': 0.62, 'ndbc': 0.62},
        'outlier-pct': {'cmems': 0.62, 'ndbc': 0.62},
        'median_abs_bias': {'cmems': 0.62, 'ndbc': 0.62},
    },
    'high': {
        'rmse': {'cmems': 0.24, 'ndbc': 0.24},
        'no-data-pct': {'cmems': 0.24, 'ndbc': 0.24},
        'outlier-pct': {'cmems': 0.24, 'ndbc': 0.24},
        'median_abs_bias': {'cmems': 0.24, 'ndbc': 0.24},
    },
    'very high': {
        'rmse': {'cmems': 0.02, 'ndbc': 0.02},
        'no-data-pct': {'cmems': 0.02, 'ndbc': 0.02},
        'outlier-pct': {'cmems': 0.02, 'ndbc': 0.02},
        'median_abs_bias': {'cmems': 0.02, 'ndbc': 0.02},
    }
}


def load_config(config_file):
    """
    Load the configuration from a YAML file.

    Args:
        config_file (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration parameters loaded from the file.
    """
    with open(config_file, 'r') as stream:
        config = load(stream, Loader=Loader)
    return config


# Load configuration
config_file_path = os.path.join(os.path.dirname(__file__), 'config.yml')
config = load_config(config_file_path)

# Extract variable names from the configuration
S1_WV_SWH_VARNAME = config.get('S1-WV-SWH-VARNAME', 'swhs1')
GROUND_TRUTH_VARNAME = config.get('GROUND_TRUTH_VARNAME', 'swh_ground_truth')
GROUND_TRUTH_VARNAME_CMEMS = 'cmems_vhm0'
INCIDENCE_ANGLE_VARNAME = config.get('INCIDENCE_ANGLE_VARNAME', 'incidence_angle')
LATITUDE_VARNAME = config.get('LATITUDE_VARNAME', 'latitude')
LONGITUDE_VARNAME = config.get('LONGITUDE_VARNAME', 'longitude')

HS_BIN_DELTA = 1.0
HS_BINS = np.arange(0, 25.0, HS_BIN_DELTA)  # :TODO: to be revised once Tab.X2 found


def get_nb_outliers(df):
    nb_outliers = 0
    df = df.copy()
    df['diffswh'] = np.abs(df[S1_WV_SWH_VARNAME] - df[GROUND_TRUTH_VARNAME])
    for rmsebin in HS_BINS:
        sub = df[(df[GROUND_TRUTH_VARNAME] > rmsebin) & (df[GROUND_TRUTH_VARNAME] < rmsebin + HS_BIN_DELTA)]
        if len(sub) == 0:
            logging.debug('No data for bin {}'.format(rmsebin))
            continue
        rmse = np.sqrt(np.mean(sub['diffswh'] ** 2))

        if rmse == 0:
            pass
        else:
            tmp_nb_outliers = len(sub[sub['diffswh'] > 3 * rmse])
            logging.debug('RMSE for bin {}: {}, nb outliers: {}'.format(rmsebin, rmse, tmp_nb_outliers))
            nb_outliers += tmp_nb_outliers
    return nb_outliers


def get_pct_outliers(df):
    nb_outliers = get_nb_outliers(df)
    return 100.0 * nb_outliers / len(df) if len(df) > 0 else 0.0


def get_rmse(df):
    """
    Calculate the RMSE (Root Mean Square Error) of the SWHs.

    Args:
        df (pd.DataFrame): DataFrame containing SWH and ground truth values.

    Returns:
        float: RMSE value.
    """
    df = df.copy()
    df = df.dropna()
    df['diffswh'] = np.abs(df[S1_WV_SWH_VARNAME] - df[GROUND_TRUTH_VARNAME])
    return float(np.sqrt(np.nanmean(df['diffswh'] ** 2)))


def get_median_bias(df):
    """
    Calculate the median absolute bias of the SWHs.

    Args:
        df (pd.DataFrame): DataFrame containing SWH and ground truth values.

    Returns:
        float: Median absolute bias value.
    """
    df = df.copy()
    df = df.dropna()
    df['diffswh'] = np.abs(df[S1_WV_SWH_VARNAME] - df[GROUND_TRUTH_VARNAME])
    return float(np.nanmedian(df['diffswh']))


def get_number_no_data(df):
    """
    Count the number of no data points in the SWHs.

    Args:
        df (pd.DataFrame): DataFrame containing SWH values.

    Returns:
        int: Number of no data points.
    """
    df = df.copy()
    return int(df[S1_WV_SWH_VARNAME].isnull().sum())


def get_pct_no_data(df):
    """
    Calculate the percentage of no data points in the SWHs.

    Args:
        df (pd.DataFrame): DataFrame containing SWH values.

    Returns:
        float: Percentage of no data points.
    """
    nbnodata = get_number_no_data(df)
    return 100.0 * nbnodata / len(df) if len(df) > 0 else 0.0


def compute_metrics(df_wv_ndbc, ds_wv_cmems, alternative_names=None):
    """
    Compute metrics for the SWHs based on the given DataFrame.

    Args:
        df_wv_ndbc (pd.DataFrame): DataFrame containing the SWH data.
        alternative_names (dict, optional): Alternative variable names.

    Returns:
        tuple: (metrics_ndbc dict, metrics_cmems dict, total_score float, inc_scores dict)
    """
    # remove the data at high latitudes
    df_wv_ndbc = df_wv_ndbc[(df_wv_ndbc[INCIDENCE_ANGLE_VARNAME] > -60) & (df_wv_ndbc[INCIDENCE_ANGLE_VARNAME] < 60)]
    ds_wv_cmems = ds_wv_cmems.where(abs(ds_wv_cmems['lat'])<60,drop=True)

    metrics_ndbc = {}
    metrics_cmems = {}
    for wv in ['wv1', 'wv2']:
        metrics_ndbc[wv] = {}
        metrics_cmems[wv] = {}
        if wv == 'wv1':
            condinc = df_wv_ndbc[INCIDENCE_ANGLE_VARNAME] < 30
            cond_cmems_inc = ds_wv_cmems['angle_of_incidence']<30
        else:
            condinc = df_wv_ndbc[INCIDENCE_ANGLE_VARNAME] > 30
            cond_cmems_inc = ds_wv_cmems['angle_of_incidence']>30
        sub = df_wv_ndbc[condinc]
        sub_cmems = ds_wv_cmems.where(cond_cmems_inc,drop=True)
        for seastate in CATEGORIES:
            metrics_ndbc[wv][seastate] = {}
            metrics_cmems[wv][seastate] = {}
            subsub = sub[(sub[GROUND_TRUTH_VARNAME] > CATEGORIES[seastate][0]) &
                         (sub[GROUND_TRUTH_VARNAME] < CATEGORIES[seastate][1])]
            tmp_cond = (sub_cmems[GROUND_TRUTH_VARNAME_CMEMS] > CATEGORIES[seastate][0]) & (sub_cmems[GROUND_TRUTH_VARNAME_CMEMS] < CATEGORIES[seastate][1])
            subsub_cmems = sub_cmems.where(tmp_cond,drop=True).to_dataframe()
            if len(subsub) == 0:
                print('warning, no data in', wv, seastate)
            if len(subsub_cmems) == 0:
                print('warning, no data CMEMS colocs in', wv, seastate)
            # WV vs NDBC buoys metrics
            nb_out = get_nb_outliers(df=subsub)
            pct_outlier = get_pct_outliers(df=subsub)
            nb_no_data = get_number_no_data(df=subsub)
            pct_no_data = get_pct_no_data(df=subsub)
            rmse = get_rmse(df=subsub)
            median_bias = get_median_bias(df=subsub)

            metrics_ndbc[wv][seastate]['nb_no_data'] = nb_no_data
            metrics_ndbc[wv][seastate]['rmse'] = rmse
            metrics_ndbc[wv][seastate]['median_abs_bias'] = median_bias
            metrics_ndbc[wv][seastate]['nb_outliers'] = nb_out
            metrics_ndbc[wv][seastate]['outlier-pct'] = pct_outlier
            metrics_ndbc[wv][seastate]['no-data-pct'] = pct_no_data

            # WV vs CMEMS model metrics
            nb_out = get_nb_outliers(df=subsub_cmems)
            pct_outlier = get_pct_outliers(df=subsub_cmems)
            nb_no_data = get_number_no_data(df=subsub_cmems)
            pct_no_data = get_pct_no_data(df=subsub_cmems)
            rmse = get_rmse(df=subsub_cmems)
            median_bias = get_median_bias(df=subsub_cmems)

            metrics_cmems[wv][seastate]['nb_no_data'] = nb_no_data
            metrics_cmems[wv][seastate]['rmse'] = rmse
            metrics_cmems[wv][seastate]['median_abs_bias'] = median_bias
            metrics_cmems[wv][seastate]['nb_outliers'] = nb_out
            metrics_cmems[wv][seastate]['outlier-pct'] = pct_outlier
            metrics_cmems[wv][seastate]['no-data-pct'] = pct_no_data
    metrics_ndbc = compute_normalized_parameters(metrics_ndbc)
    metrics_cmems = compute_normalized_parameters(metrics_cmems)
    total_score, inc_scores = compute_score(metrics_ndbc=metrics_ndbc,metrics_cmems=metrics_cmems)
    return metrics_ndbc,metrics_cmems, total_score, inc_scores


def compute_normalized_parameters(metrics):
    """
    Add normalized parameters to metrics for scoring.

    Args:
        metrics (dict): Dictionary containing parameters per incidence angle and sea state class.

    Returns:
        dict: Metrics with normalized parameters added.
    """
    four_params = ['no-data-pct', 'outlier-pct', 'rmse', 'median_abs_bias']
    for wv in metrics:
        for cat in metrics[wv]:
            for parm in four_params:
                metrics[wv][cat][parm + '_normalized'] = (
                    (metrics[wv][cat][parm] - MIN_MAX_BOUNDARIES_4_NORMALIZATION[cat][parm]['min']) /
                    (MIN_MAX_BOUNDARIES_4_NORMALIZATION[cat][parm]['max'] - MIN_MAX_BOUNDARIES_4_NORMALIZATION[cat][parm]['min'])
                )
    return metrics


def compute_score(metrics_ndbc,metrics_cmems):
    """
    Compute the total score and scores per incidence angle and sea state class.

    Args:
        metrics_ndbc (dict): Dictionary containing normalized parameters WV vs NDBC.
        metrics_cmems (dict): Dictionary containing normalized parameters WV vs CMEMS.

    Returns:
        tuple: (total_score float, scores dict)
    """
    scores = {}
    for wv in metrics_ndbc:
        scores[wv] = {}
        for cat in metrics_ndbc[wv]:
            ndbc_score_part = WEIGHTINGS_FACTOR_K[cat]['rmse']['ndbc'] * (
                metrics_ndbc[wv][cat]['no-data-pct_normalized'] +
                metrics_ndbc[wv][cat]['rmse_normalized'] +
                metrics_ndbc[wv][cat]['median_abs_bias_normalized'] +
                metrics_ndbc[wv][cat]['outlier-pct_normalized']
            )
            cmems_score_part = WEIGHTINGS_FACTOR_K[cat]['rmse']['ndbc'] * (
                metrics_cmems[wv][cat]['no-data-pct_normalized'] +
                metrics_cmems[wv][cat]['rmse_normalized'] +
                metrics_cmems[wv][cat]['median_abs_bias_normalized'] +
                metrics_cmems[wv][cat]['outlier-pct_normalized']
            )
            scores[wv][cat] = ndbc_score_part*0.3 + cmems_score_part*0.7
    # here not sure whether the total score is a sum or a mean
    # total_score = 0.5 * (np.sum([scores['wv1'][ss] for ss in scores['wv1']])) + \
    #               0.5 * (np.sum([scores['wv2'][ss] for ss in scores['wv2']]))
    total_score = 0.5 * (np.mean([scores['wv1'][ss] for ss in scores['wv1']])) + \
                   0.5 * (np.mean([scores['wv2'][ss] for ss in scores['wv2']]))
    return total_score, scores


def make_df(swhs, ground_truth):
    """
    Create a DataFrame from SWH and ground truth arrays.

    Args:
        swhs (array-like): Predicted SWH values.
        ground_truth (array-like): Ground truth SWH values.

    Returns:
        pd.DataFrame: DataFrame with SWH and ground truth columns.
    """
    return pd.DataFrame({
        S1_WV_SWH_VARNAME: swhs,
        GROUND_TRUTH_VARNAME: ground_truth
    })


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    print('start')
    # One value far from ground truth, others close
    np.random.seed(42)
    predictions = np.random.uniform(0, 10, 500)
    ground_truth = copy.copy(predictions)
    predictions[95] = 200.0
    df = make_df(predictions, ground_truth)
    print(get_nb_outliers(df))