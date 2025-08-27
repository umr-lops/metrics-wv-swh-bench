import xarray as xr
import numpy as np
from tqdm import tqdm

def colocate_cmems_with_s1wv(cmems_path, s1wv_path, time_tolerance_hours=3):
    """
    Colocate Sentinel-1 WV observations with CMEMS model data.
    For each S1WV observation, find the closest CMEMS time and then the nearest grid point in space.

    Args:
        cmems_path (str): Path to CMEMS NetCDF file.
        s1wv_path (str): Path to Sentinel-1 WV NetCDF file.
        time_tolerance_hours (float): Maximum allowed time difference for colocation.

    Returns:
        xr.Dataset: S1WV dataset with added colocated CMEMS model values.
    """
    # Load datasets
    ds_cmems = xr.open_dataset(cmems_path)
    ds_s1wv = xr.open_dataset(s1wv_path)

    # Convert CMEMS time to datetime64
    cmems_time_units = ds_cmems['time'].attrs.get('units', 'hours since 1950-01-01')
    cmems_time = xr.decode_cf(ds_cmems).time.values

    # Prepare arrays
    cmems_lats = ds_cmems['latitude'].values
    cmems_lons = ds_cmems['longitude'].values

    # Prepare output arrays
    colocated_vhm0 = []
    colocated_time = []

    # Loop over S1WV observations
    # for i in range(ds_s1wv.dims['time']):
    for i in tqdm(range(ds_s1wv.dims['time']), desc="Colocating S1WV with CMEMS"):
        obs_time = ds_s1wv['time'].values[i]
        obs_lat = ds_s1wv['lat'].values[i]
        obs_lon = ds_s1wv['lon'].values[i]

        # Find closest CMEMS time
        time_diffs = np.abs(cmems_time - obs_time)
        min_time_idx = np.argmin(time_diffs)
        if time_diffs[min_time_idx] > np.timedelta64(time_tolerance_hours, 'h'):
            colocated_vhm0.append(np.nan)
            colocated_time.append(np.datetime64('NaT'))
            continue

        # Find closest CMEMS grid point
        lat_idx = np.argmin(np.abs(cmems_lats - obs_lat))
        lon_idx = np.argmin(np.abs(cmems_lons - obs_lon))

        # Get colocated model value
        vhm0_val = ds_cmems['VHM0'][min_time_idx, lat_idx, lon_idx].values
        # Apply scale factor and offset if present
        scale = ds_cmems['VHM0'].attrs.get('scale_factor', 1.0)
        offset = ds_cmems['VHM0'].attrs.get('add_offset', 0.0)
        fill_value = ds_cmems['VHM0'].attrs.get('_FillValue', -32767)
        if vhm0_val == fill_value:
            vhm0_val = np.nan
        else:
            vhm0_val = vhm0_val * scale + offset

        colocated_vhm0.append(vhm0_val)
        colocated_time.append(cmems_time[min_time_idx])

    # Add results to S1WV dataset
    ds_s1wv['cmems_vhm0'] = (('time',), np.array(colocated_vhm0))
    ds_s1wv['cmems_time'] = (('time',), np.array(colocated_time))

    return ds_s1wv