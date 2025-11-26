#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
script to build co-aligned between altimeter datasets CMEMS WAV L3 and Sentinel-1 WV SWH predictions.
creation: November 2025


Need a CSV coaligned dataset between WV Sentinel-1 and altimeters from CMEMS database

I want to have recent data for both S1C and recent altimeters data such as SWOT.
CMEMS is not a first choice. So first target would be one month such as 2025-August.
The amount should not be too large for allow rapid test exercize and challengers to get the dataset without a huge infrastructure for reprocessing.
This dataset could be a subset a dataset updated daily.


Once CCI sea state will be available for 2025.


inspired from notebook: sentinel1/hs_total/round_robin_benchmark/build_alti_wv_coaligned_dataset.ipynb


First idea is to take benefit from unified colocs generated daily at Ifremer
These colocs take in inputs: CCI sea state Quach files (one nc per SAFE) + L3 CMEMS Alti database
"""
import glob
import os
import time
import collections
from tqdm import tqdm
import numpy as np
import xarray as xr
import datetime
import pandas as pd
import s1ifr.get_full_path_from_measurement
import s1ifr.match_measu_l1_l2
from dateutil import rrule
import logging
import argparse

DIR_COLOCS = '/home/datawork-cersat-public/project/mpc-sentinel1/analysis/s1_data_analysis/hs_nn/unified_colocs_wv_alti/'

def list_nc_coloc_files_wv_alti(start_date,stop_date,dir_colocs=DIR_COLOCS)->list:
    """
    list all coloc files between start_date and stop_date in dir_colocs
    1 coloc file per day
    1 coloc file contains all coloc between S1 WV and altimeters for this day
    1 coloc file named as:for instance: coloc_20250803_S1A_WV_cmems_Jason-3_3_hours_2_degree.nc

    Arguments:
        start_date: datetime.date
        stop_date: datetime.date
        dir_colocs: str, directory where coloc files are stored 
    
    
    Returns:
        list of str: list of coloc files paths

    """
    logging.info('input directory for coloc files: '+dir_colocs)
    lst = []
    for dd in rrule.rrule(rrule.DAILY,dtstart=start_date,until=stop_date):
        lst += glob.glob(os.path.join(dir_colocs,'S1*_cmems*',dd.strftime('%Y'),'coloc_'+dd.strftime('%Y%m%d')+'*.nc'))
    logging.info(f'found {len(lst)} coloc files between {start_date} and {stop_date}')
    logging.debug('example file: '+str(lst[0]) if len(lst)>0 else 'no file found')
    return lst

def read_and_concat_coloc_files(list_nc_files)->xr.Dataset:
    """
    read and concat all coloc files in list_nc_files into a single xarray.Dataset

    Arguments:
        list_nc_files: list of str, paths to coloc netcdf files

    Returns:
        xr.Dataset: concatenated dataset
    """
    concat = []
    for ii in tqdm(range(len(list_nc_files))):
        tmpds = xr.open_dataset(list_nc_files[ii],decode_timedelta=True)
        unit = os.path.basename(list_nc_files[ii]).split('_')[2]
        alti_name = os.path.basename(list_nc_files[ii]).split('_')[5]
        tmpds['SAR-unit'] = xr.DataArray(np.tile(unit,len(tmpds['time_sar'])),dims='time_sar')
        tmpds['ALTI-unit'] = xr.DataArray(np.tile(alti_name,len(tmpds['time_sar'])),dims='time_sar')
        concat.append(tmpds)
    ds_colocs = xr.concat(concat,dim='time_sar')
    # some stats on the colocs:
    for uu in np.unique(ds_colocs['ALTI-unit']):
        logging.info('count of matchups per alti %s : %s percentage %1.1f%%',uu,(ds_colocs['ALTI-unit'].values==uu).sum(),
                     100.*(ds_colocs['ALTI-unit'].values==uu).sum()/len(ds_colocs['ALTI-unit']))
    return ds_colocs





def from_npdt64_to_dt(dt64):
    # Convertir le numpy.datetime64 en timestamp (secondes depuis epoch)
    ref_date = np.datetime64("1970-01-01T00:00:00")
    ts = (dt64 - ref_date) / np.timedelta64(1, "s")

    # Créer un datetime "timezone-aware" en UTC (nouvelle méthode recommandée)
    dt = datetime.datetime.fromtimestamp(ts, datetime.UTC)

    return dt

def add_wv_measurement_fullpath(ds_colocs:xr.Dataset)->xr.Dataset:
    """
    
    add a new variable to ds_colocs with the fullpath to the WV measurement file from s1ifr database
    
    Arguments:
        ds_colocs: xr.Dataset, dataset with coloc data
    Returns:
        xr.Dataset: dataset with new variable 'wv_fullpath'
    """
    logging.info('step: find matching full path in Ifremer archive for WV SLC measurements')
    fullpaths = []
    fullpaths_slc = []
    cpt_not_found = 0
    cpt_total = 0
    for ii in tqdm(range(len(ds_colocs['time_sar']))):
        cpt_total += 1
        onedate = ds_colocs['time_sar'].values[ii]
        one_date_dt = from_npdt64_to_dt(onedate)
        one_unit = ds_colocs['SAR-unit'].values[ii]
        # print(onedate,one_unit)
        fullpath_measu_ocn = s1ifr.get_full_path_from_measurement.get_full_path_ocn_wv_from_approximate_date(datedt=one_date_dt,
                                                     sat=one_unit, level="L2", storage="datawork", nb_seconds_delta=3)
        fullpath_measu_slc = None
        # print(fullpath,os.path.exists(fullpath))
        if fullpath_measu_ocn is not None:
            if os.path.exists(fullpath_measu_ocn):
                fullpath_measu_slc = s1ifr.match_measu_l1_l2.getTiffcorresponding2NetCDF(fullpath_measu_ocn)
                fullpaths.append(fullpath_measu_ocn)
                # add the name of the alti:
            else:
                cpt_not_found += 1
                fullpaths.append(np.nan)
        else:
            cpt_not_found += 1
            fullpaths.append(np.nan)
            logging.debug('not foun case: %s %s',one_date_dt,one_unit)
        if fullpath_measu_slc is not None:
            if os.path.exists(fullpath_measu_slc):
                fullpaths_slc.append(fullpath_measu_slc)
                # add the name of the alti:
            else:
                cpt_not_found += 1
                fullpaths_slc.append(np.nan)
        else:
            cpt_not_found += 1
            fullpaths_slc.append(np.nan)
            logging.debug('not foun case: %s %s',one_date_dt,one_unit)
    logging.info('find matching full path in Ifremer archive done: not found %i/%i',cpt_not_found,cpt_total)
    ds_colocs['wv-id-ocn'] = xr.DataArray(fullpaths,dims=['time_sar'],
                    attrs={'description':'full path of Sentinel-1 WV OCN measurement (.nc) in Ifremer archive'})
    ds_colocs['wv-id-slc'] = xr.DataArray(fullpaths_slc,dims=['time_sar'],
                    attrs={'description':'full path of Sentinel-1 WV SLC measurement (.tiff) in Ifremer archive'})
    return ds_colocs

def write_coaligned_dataset(ds_colocs:xr.Dataset,output_filepath:str):
    """
    write the co-aligned dataset to a netcdf file

    Arguments:
        ds_colocs: xr.Dataset, dataset with coloc data
        output_filepath: str, path to output netcdf file
    """
    
    logging.info('step: write co-aligned dataset to CSV file %s',output_filepath)
    maskok_slc = (~ds_colocs['wv-id-slc'].isnull()) & (ds_colocs['wv-id-slc'].astype(str) != "nan") & (ds_colocs['wv-id-slc'].astype(str) != "")
    maskok_ocn = (~ds_colocs['wv-id-ocn'].isnull()) & (ds_colocs['wv-id-ocn'].astype(str) != "nan") & (ds_colocs['wv-id-ocn'].astype(str) != "")
    # print(maskok,maskok.values.sum())
    maskok = maskok_slc & maskok_ocn
    logging.info('mask OK %i/%i',maskok.sum(),len(maskok))
    subset_id_ok = ds_colocs.where(maskok,drop=True)
    df = subset_id_ok.to_dataframe()
    # rename
    df.rename(columns={"swh": "swh_WV_Quach_CCIseasate_Ifr_v2_polSpec_firstIPFintegration"}, inplace=True)
    df.rename(columns={"swh_uncertainty": "swh_uncertainty_WV_Quach_CCIseasate_Ifr_v2_polSpec_firstIPFintegration"}, inplace=True)
    df.rename(columns={"swh_quality": "swh_quality_WV_Quach_CCIseasate_Ifr_v2_polSpec_firstIPFintegration"}, inplace=True)
    df.rename(columns={"swh_rejection_flags": "swh_rejection_flags_WV_Quach_CCIseasate_Ifr_v2_polSpec_firstIPFintegration"}, inplace=True)
    df.to_csv(output_filepath)
    logging.info('output co-aligned dataset WV -alti %s',output_filepath)
    try:
        os.chmod(output_filepath,0o0664)
    except PermissionError:
        logging.warning('cannot change permission of file %s',output_filepath)


def entrypoint():
    # remove handlers added by imported modules to avoid duplicate logs
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    # parsing of arguments
    parser = argparse.ArgumentParser(description='Build co-aligned dataset between Sentinel-1 WV SWH predictions and CMEMS altimeter L3 data.')
    parser.add_argument('--start_date', type=str, required=True,
                        help='Start date in format YYYY-MM-DD')
    parser.add_argument('--stop_date', type=str, required=True,
                        help='Stop date in format YYYY-MM-DD')
    parser.add_argument('--output_directory', type=str, required=True,help='Output directory to save the co-aligned dataset CSV file')
    parser.add_argument('--log_level', type=str, default='INFO',
                        help='Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)')
    args = parser.parse_args()

    # configure logging
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.log_level)
    logging.basicConfig(level=numeric_level,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info('Welcome to build co-aligned dataset between Sentinel-1 WV SWH predictions and CMEMS altimeter L3 data.')
    logging.info('context: workbench creation for SAR Sentinel-1 ocean algorithms validation.')
    start_time = time.time()
    start_date = datetime.datetime.strptime(args.start_date, '%Y-%m-%d').date()
    stop_date = datetime.datetime.strptime(args.stop_date, '%Y-%m-%d').date()
    output_directory = args.output_directory
    if not os.path.exists(output_directory):
        os.makedirs(output_directory,exist_ok=True)
    list_nc_files = list_nc_coloc_files_wv_alti(start_date,stop_date)
    ds_colocs = read_and_concat_coloc_files(list_nc_files)
    ds_colocs = add_wv_measurement_fullpath(ds_colocs)
    output_filepath = os.path.join(output_directory,
            'coaligned_S1_WV_SWH_predictions_versus_CMEMS_ALTI_L3_DeltaT_3hours_DeltaG_2deg_%s_%s.csv'%(
                start_date.strftime('%Y%m%d'),stop_date.strftime('%Y%m%d')))
    if os.path.exists(output_filepath):
        logging.warning('output file %s already exists and will be overwritten',output_filepath)
    write_coaligned_dataset(ds_colocs,output_filepath)
    # estimate peak memory usage
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        peak_memory_mb = mem_info.rss / (1024 * 1024)  # Convert bytes to MB
        logging.info('Peak memory usage: %.2f MB', peak_memory_mb)
    except ImportError:
        logging.warning('psutil module not found, cannot estimate peak memory usage')   
    logging.info('elapsed time: %.2f seconds',time.time()-start_time)
    logging.info('script finished successfully')


if __name__ == '__main__':
    entrypoint()