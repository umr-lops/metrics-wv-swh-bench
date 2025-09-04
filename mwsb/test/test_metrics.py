import pytest
import pandas as pd
import numpy as np
import copy
from mwsb.metrics import get_nb_outliers, S1_WV_SWH_VARNAME, GROUND_TRUTH_VARNAME
from mwsb.metrics import get_rmse, get_median_bias
import mwsb.metrics as metrics

# File: mwsb/test_metrics.py

def make_df(swhs, ground_truth):
    return pd.DataFrame({
        S1_WV_SWH_VARNAME: swhs,
        GROUND_TRUTH_VARNAME: ground_truth
    })

def test_no_outliers():
    # All values close to ground truth, RMSE > 0
    df = make_df([1.0, 1.1, 1.2, 1.05], [1.0, 1.1, 1.2, 1.05])
    assert get_nb_outliers(df) == 0

def test_one_outlier():
    # One value far from ground truth, others close
    # df = make_df([1.0, 1.1, 1.2, 10.0], [1.0, 1.1, 1.2, 1.0])
    np.random.seed(42)
    predictions = np.random.uniform(0, 10, 500)
    # ground_truth = np.random.uniform(0, 10, 100)
    ground_truth = copy.copy(predictions)
    predictions[95] = 200.
    df = make_df(predictions, ground_truth)
    # print(df)
    # df = make_df([1.0, 1.1, 1.2, 10.0], [1.0, 1.1, 1.2, 1.0])
    # print(get_nb_outliers(df))
    actual = get_nb_outliers(df)
    assert actual == 1


def test_empty_df():
    df = make_df([], [])
    assert get_nb_outliers(df) == 0

def test_with_nan():
    df = make_df([1.0, np.nan, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0])
    assert isinstance(get_nb_outliers(df), int)

def test_rmse_perfect_match():
    df = make_df([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert get_rmse(df) == 0.0

def test_rmse_simple():
    df = make_df([2.0, 2.0, 2.0], [1.0, 1.0, 1.0])
    # RMSE = sqrt(mean((2-1)^2)) = sqrt(1) = 1.0
    assert get_rmse(df) == 1.0

def test_median_bias_perfect_match():
    df = make_df([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert get_median_bias(df) == 0.0

def test_median_bias_simple():
    df = make_df([2.0, 2.0, 2.0], [1.0, 1.0, 1.0])
    # All diffs are 1.0, so median is 1.0
    assert get_median_bias(df) == 1.0

def test_load_config(tmp_path):
    # Crée un fichier YAML temporaire
    yaml_content = "S1-WV-SWH-VARNAME: test_swh\nGROUND_TRUTH_VARNAME: test_gt"
    config_file = tmp_path / "config.yml"
    config_file.write_text(yaml_content)
    config = metrics.load_config(str(config_file))
    assert config["S1-WV-SWH-VARNAME"] == "test_swh"
    assert config["GROUND_TRUTH_VARNAME"] == "test_gt"


def test_get_rmse():
    df = pd.DataFrame({
        metrics.S1_WV_SWH_VARNAME: [1, 2, 3],
        metrics.GROUND_TRUTH_VARNAME: [1, 2, 4]
    })
    rmse = metrics.get_rmse(df)
    assert np.isclose(rmse, np.sqrt(((0)**2 + (0)**2 + (1)**2)/3))

def test_get_median_bias():
    df = pd.DataFrame({
        metrics.S1_WV_SWH_VARNAME: [1, 2, 3],
        metrics.GROUND_TRUTH_VARNAME: [1, 2, 4]
    })
    bias = metrics.get_median_bias(df)
    assert bias == np.median([0, 0, 1])

def test_get_number_no_data():
    df = pd.DataFrame({
        metrics.S1_WV_SWH_VARNAME: [1, np.nan, 3],
        metrics.GROUND_TRUTH_VARNAME: [1, 2, 3]
    })
    assert metrics.get_number_no_data(df) == 1

def test_get_pct_no_data():
    df = pd.DataFrame({
        metrics.S1_WV_SWH_VARNAME: [1, np.nan, 3, np.nan],
        metrics.GROUND_TRUTH_VARNAME: [1, 2, 3, 4]
    })
    pct = metrics.get_pct_no_data(df)
    assert np.isclose(pct, 50.0)

def test_make_df():
    swhs = [1, 2, 3]
    gt = [1, 2, 4]
    df = metrics.make_df(swhs, gt)
    assert metrics.S1_WV_SWH_VARNAME in df.columns
    assert metrics.GROUND_TRUTH_VARNAME in df.columns
    assert len(df) == 3

def test_compute_normalized_parameters():
    # Structure simplifiée
    metrics_dict = {
        'wv1': {
            'low': {'no-data-pct': 0, 'outlier-pct': 0, 'rmse': 0.2, 'median_abs_bias': 0}
        }
    }
    norm = metrics.compute_normalized_parameters(metrics_dict)
    assert 'no-data-pct_normalized' in norm['wv1']['low']

def test_compute_score():
    # Structure simplifiée
    metrics_ndbc = {
        'wv1': {'low': {'no-data-pct_normalized': 0, 'rmse_normalized': 0, 'median_abs_bias_normalized': 0, 'outlier-pct_normalized': 0}},
        'wv2': {'low': {'no-data-pct_normalized': 0, 'rmse_normalized': 0, 'median_abs_bias_normalized': 0, 'outlier-pct_normalized': 0}}
    }
    metrics_cmems = {
        'wv1': {'low': {'no-data-pct_normalized': 0, 'rmse_normalized': 0, 'median_abs_bias_normalized': 0, 'outlier-pct_normalized': 0}},
        'wv2': {'low': {'no-data-pct_normalized': 0, 'rmse_normalized': 0, 'median_abs_bias_normalized': 0, 'outlier-pct_normalized': 0}}
    }
    total_score, scores = metrics.compute_score(metrics_ndbc, metrics_cmems)
    assert isinstance(total_score, float)
    assert isinstance(scores, dict)