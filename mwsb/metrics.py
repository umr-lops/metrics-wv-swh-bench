import numpy as np
import os
from yaml import load
from yaml import CLoader as Loader
import logging

def load_config(config_file):
    """
    Load the configuration from a YAML file.
    
    Args:
        config_file (str): Path to the YAML configuration file.
        
    Returns:
        dict: Configuration parameters loaded from the file.
    """
    # with open(config_file, 'r') as file:
        # config = safe_load(file)
    stream = open(config_file, 'r')
    config = load(stream, Loader=Loader)
    return config

# Load configuration
config_file_path = os.path.join(os.path.dirname(__file__), 'config.yml')
config = load_config(config_file_path)

# Extract variable names from the configuration
S1_WV_SWH_VARNAME = config.get('S1-WV-SWH-VARNAME', 'swhs1')
GROUND_TRUTH_VARNAME = config.get('GROUND_TRUTH_VARNAME', 'swh_ground_truth')
INCIDENCE_ANGLE_VARNAME = config.get('INCIDENCE_ANGLE_VARNAME', 'incidence_angle')
LATITUDE_VARNAME = config.get('LATITUDE_VARNAME', 'latitude')
LONGITUDE_VARNAME = config.get('LONGITUDE_VARNAME', 'longitude')

HS_BIN_DELTA = 1.0
HS_BINS = np.arange(0,25.,HS_BIN_DELTA) # :TODO: to be revised once Tab.X2 found

def get_nb_outliers(df):
    nb_outliers = 0
    df = df.copy()
    df['diffswh'] = np.abs(df[S1_WV_SWH_VARNAME] - df[GROUND_TRUTH_VARNAME])
    for rmsebin in HS_BINS:
        sub = df[(df[GROUND_TRUTH_VARNAME]>rmsebin) & (df[GROUND_TRUTH_VARNAME]<rmsebin+HS_BIN_DELTA)]
        if len(sub) == 0:
            logging.debug('No data for bin {}'.format(rmsebin))
            continue
        rmse = np.sqrt(np.mean(sub['diffswh']**2))
        
        if rmse == 0:
            #nb_outliers += len(sub[sub['diffswh'] > 0])
            pass
        else:
            tmp_nb_outliers = len(sub[sub['diffswh'] > 3 * rmse])
            logging.debug('RMSE for bin {}: {}, nb outliers: {}'.format(rmsebin, rmse,tmp_nb_outliers))
            nb_outliers = nb_outliers+tmp_nb_outliers
    return nb_outliers

def get_rmse(df):
    """
    Returns the RMSE of the SWHs.
    """
    df = df.copy()
    df['diffswh'] = np.abs(df[S1_WV_SWH_VARNAME] - df[GROUND_TRUTH_VARNAME])
    return np.sqrt(np.mean(df['diffswh']**2))

def get_median_bias(df):
    """
    Returns the median bias of the SWHs.
    """
    df = df.copy()
    df['diffswh'] = np.abs(df[S1_WV_SWH_VARNAME] - df[GROUND_TRUTH_VARNAME])
    return np.median(df['diffswh'])

def get_number_no_data(df):
    """
    Returns the number of no data points in the SWHs.
    """
    df = df.copy()
    return len(df[S1_WV_SWH_VARNAME].isnull())
    

def compute_metrics(df):
    """
    Computes the metrics for the SWHs based on the given DataFrame.

    The DataFrame should contain the following columns:
    - 'swhs1': SWH from the first source
    - 'swh_ground_truth': Ground truth SWH
    - 'incidence': Incidence angle (e.g., 'wv1', 'wv2')
    - 'seastate': Sea state value
    - 'lat': Latitude of the measurement

    Returns a dictionary with metrics for each incidence angle and sea state category.

    Args:
        df (DataFrame): DataFrame containing the SWH data.

    Returns:
        dict: A dictionary with metrics for each incidence angle and sea state category.    

    """


    # remove the data at high latitudes
    df = df[(df[INCIDENCE_ANGLE_VARNAME] > -60) & (df[INCIDENCE_ANGLE_VARNAME] < 60)]

    categories = {'low':(0,1.5),
                  'medium':(1.5,3.0),
                  'high':(3.0,6.0),
                  'very high':(6.0,30.0) }
    metrics = {}
    for wv in ['wv1','wv2']:
        metrics[wv] = {}
        sub = df[df[INCIDENCE_ANGLE_VARNAME] == wv ]
        for seastate in categories:
            metrics[wv][seastate] = {}
            subsub = sub[(sub['seastate'] > categories[seastate][0]) &
                         (sub['seastate'] < categories[seastate][1])]
            nb_out = get_nb_outliers(df=subsub)
            nb_no_data = get_number_no_data(df=subsub)
            rmse = get_rmse(df=subsub)
            median_bias = get_median_bias(df=subsub)

            metrics[wv][seastate]['nb_no_data'] = nb_no_data
            metrics[wv][seastate]['rmse'] = rmse
            metrics[wv][seastate]['median_bias'] = median_bias
            metrics[wv][seastate]['nb_outliers'] = nb_out

    return metrics

def make_df(swhs, ground_truth):
    return pd.DataFrame({
        S1_WV_SWH_VARNAME: swhs,
        GROUND_TRUTH_VARNAME: ground_truth
    })

if __name__ == "__main__":
    import pandas as pd
    import sys
    import copy
    logging.basicConfig(level=logging.DEBUG)
    print('start')
        # One value far from ground truth, others close
    np.random.seed(42)
    predictions = np.random.uniform(0, 10, 500)
    # ground_truth = np.random.uniform(0, 10, 100)
    ground_truth = copy.copy(predictions)
    predictions[95] = 200.
    df = make_df(predictions, ground_truth)
    # print(df)
    # df = make_df([1.0, 1.1, 1.2, 10.0], [1.0, 1.1, 1.2, 1.0])
    print(get_nb_outliers(df))