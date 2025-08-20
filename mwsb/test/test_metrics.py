import pytest
import pandas as pd
import numpy as np
import copy
from mwsb.metrics import get_nb_outliers, S1_WV_SWH_VARNAME, GROUND_TRUTH_VARNAME
from mwsb.metrics import get_rmse, get_median_bias

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