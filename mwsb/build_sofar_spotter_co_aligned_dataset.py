import pandas as pd
import xarray as xr
import numpy as np
import os
import glob
from tqdm import tqdm
import logging
from collections import defaultdict
import functools # Used for LRU cache if you want to limit cache size

# Assuming sofards is already loaded as an xarray.Dataset
# Example placeholder if not provided:
# sofards = xr.open_dataset('your_sofar_dataset.nc')

path_l2daily_ifr = '/home/datawork-cersat-public/cache/project/mpc-sentinel1/analysis/s1_data_analysis/L2_daily/0.12/'
def sofar_wv_colocation(sofards,dev=False,spatial_tolerance = 0.1,time_tolerance_hours=1):
    """
    
    Collocate SOFAR Spotter data with SAR-derived Wave (WV) data from Ifremer L2 daily files.
    Args:
        sofards (xarray.Dataset): SOFAR Spotter dataset containing 'longitude', 'latitude', 'time', 'significantWaveHeight', and 'trajectory' variables.
        dev (bool, optional): If True, process only a small subset of data for development/testing.
        spatial_tolerance (float, optional): Maximum distance in degrees for collocation between Spotter and SAR WV points.
    Returns:
        collocated_df (pandas.DataFrame): DataFrame containing collocated points with columns:
            - 'trajectory_id': Identifier for the Spotter trajectory
            - 'HsSAR': Significant wave height from SAR data
            - 'lonSAR': Longitude of the SAR point
            - 'latSAR': Latitude of the SAR point           
            - 'HsSpotter': Significant wave height from Spotter data
            - 'lonSpotter': Longitude of the Spotter point
            - 'latSpotter': Latitude of the Spotter point
            - 'timeSpotter': Timestamp of the Spotter observation
            - 'timeSAR': Timestamp of the SAR observation
        collocation_outcome_counts (dict): Dictionary with counts of collocation outcomes:
            - 'SuccessfulCollocations': Total number of successful collocated points
            - 'NoWVFilesFound': Number of Spotter points with no WV files found for their date
            - 'NoCloseWVPoints': Number of Spotter points with WV files but no close SAR points within the spatial tolerance
            - 'NoCloseWVPointsIntime': Number of Spotter points with close SAR points but none within the time tolerance
            - 'PointsSOFARWithAtLeastOneSuccessfulCollocation': Number of Spotter points that had at least one successful collocation
    Note:
        - Ensure that the SAR WV data variable name for significant wave height is correctly set in the code (currently 'swh').
        - The function uses a simple Euclidean distance for spatial collocation; consider using geodesic distance for more accuracy over larger distances.
        - The function caches opened WV datasets to improve performance when multiple Spotter points fall on the same day.  
    """   
    
    # Create an empty list to store collocated data
    collocated_data = []


    # Initialize defaultdict to count specific outcomes globally (not per trajectory as in prev version)
    # Based on your modification, you removed the trajectory_id from the counter,
    # so this will now count overall totals for each category.
    collocation_outcome_counts = defaultdict(int)

    # --- IMPORTANT: Adjust 'swh' to the actual significant wave height variable name in your WV data ---
    # sar_swh_var_name = 'swh' # <--- CHANGE THIS IF YOUR WV DATA HAS A DIFFERENT VARIABLE NAME
    sar_swh_var_name = 's1_hs_emp_tot_v3p0'

    # --- CACHE FOR WV DAILY FILES ---
    # Option 1: Simple dictionary cache (can grow indefinitely)
    wv_file_cache = {}

    # Option 2: LRU (Least Recently Used) Cache for controlled memory usage
    # Uncomment the line below and comment out Option 1 if you prefer LRU cache
    # @functools.lru_cache(maxsize=128) # Cache up to 128 daily WV files
    # def open_wv_dataset(file_path):
    #     return xr.open_dataset(file_path)
    # --------------------------------
    # nb_points_to_process = 100 if dev else len(sofards.index)
    nb_points_to_process = 10000 if dev else len(sofards.index)
    logging.info(f"Processing {nb_points_to_process} Spotter points for collocation...")
    pbar = tqdm(range(nb_points_to_process))
    for ii in pbar:
        pbar.set_description(f"Nb collocated Spotter point {collocation_outcome_counts['SuccessfulCollocations']}/{nb_points_to_process}")
        sofaronept = sofards.isel(index=ii)
        
        # You had this commented out, but if you want to use trajectory_id for *anything* in future,
        # it's good to keep it handy. For the global counts you've set up, it's not strictly needed.
        current_trajectory_id = sofaronept['trajectory'].values # Still extracting for potential future use

        lonspotter = sofaronept['longitude'].values.item() 
        latspotter = sofaronept['latitude'].values.item()
        timespotter = sofaronept['time'].values 
        
        pd_timespotter = pd.Timestamp(timespotter)
        year_str = pd_timespotter.strftime('%Y')
        day_of_year = pd_timespotter.strftime('%j')

        pattern = os.path.join(path_l2daily_ifr, year_str, day_of_year, '*nc')
        lstwvfiles = glob.glob(pattern)
        
        # Flag to check if this specific Spotter point found ANY close SAR WV points
        found_any_close_sar_point_for_this_spotter = False 
        
        if not lstwvfiles:
            collocation_outcome_counts['NoWVFilesFound'] += 1
            continue 

        # If we are here, at least one WV file was found for this day
        current_spotter_point_had_collocation_in_any_wv_file = False

        for ffwv in lstwvfiles:
            try:
                # --- Use the cache here ---
                # If using simple dict cache:
                if ffwv not in wv_file_cache:
                    wv_file_cache[ffwv] = xr.open_dataset(ffwv)
                dswv = wv_file_cache[ffwv]
                
                # If using LRU cache (uncomment @functools.lru_cache and related lines above):
                # dswv = open_wv_dataset(ffwv)
                # ---------------------------

                if 'lon' not in dswv.data_vars or 'lat' not in dswv.data_vars:
                    # print(f"Skipping {ffwv}: 'lon' or 'lat' not found in data variables.")
                    continue
                
                if sar_swh_var_name not in dswv.data_vars:
                    # print(f"Skipping {ffwv}: '{sar_swh_var_name}' not found in data variables.")
                    continue
                
                sar_lons = dswv['lon'].values
                sar_lats = dswv['lat'].values
                sar_times = dswv['fdatedt'].values
                sar_paths = dswv['subpath'].values
                sar_swh_raw = dswv[sar_swh_var_name]

                distances = np.sqrt((sar_lons - lonspotter)**2 + (sar_lats - latspotter)**2)
                close_indices = np.where(distances < spatial_tolerance)[0]

                if len(close_indices) > 0:
                    current_spotter_point_had_collocation_in_any_wv_file = True
                    
                    for idx in close_indices:
                        sar_lon = sar_lons[idx]
                        sar_lat = sar_lats[idx]
                        sar_time = sar_times[idx]
                        sar_path = sar_paths[idx]
                        if 'wv1' in sar_path.lower():
                            incidence_angle = 24.0
                        else:
                            incidence_angle = 37.0

                        if np.abs((pd.Timestamp(sar_time) - pd_timespotter).total_seconds()) < time_tolerance_hours * 3600:
                        
                            if sar_swh_raw.ndim > 1:
                                sar_hs = sar_swh_raw.isel(fdatedt=idx, partition=0).item() 
                            else:
                                sar_hs = sar_swh_raw.isel(fdatedt=idx).item()

                            collocated_data.append({
                                # 'trajectory_id': current_trajectory_id, # Re-added for the collocated_df
                                'HsSAR': sar_hs,
                                'lonSAR': sar_lon,
                                'latSAR': sar_lat,
                                'pathSAR': sar_path,
                                'incidenceAngleSAR': incidence_angle,
                                'HsSpotter': sofaronept['significantWaveHeight'].values.item(),
                                'lonSpotter': lonspotter,
                                'latSpotter': latspotter,
                                'timeSpotter': pd_timespotter,
                                'timeSAR': dswv['fdatedt'].isel(fdatedt=idx).values
                            })
                            collocation_outcome_counts['SuccessfulCollocations'] += 1
                        else:
                            collocation_outcome_counts['NoCloseWVPointsIntime'] += 1
                    
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"Error processing file {ffwv} for Spotter point {ii}: {e}")
                continue

        # After checking all WV files for this Spotter point:
        if not current_spotter_point_had_collocation_in_any_wv_file and lstwvfiles:
            collocation_outcome_counts['NoCloseWVPoints'] += 1
        else:
            collocation_outcome_counts['PointsSOFARWithAtLeastOneSuccessfulCollocation'] += 1
        if dev is True and collocation_outcome_counts['SuccessfulCollocations'] >=500:
            logging.info("Dev mode: reached 500 successful collocations, stopping early.")
            break


    # Convert the list of dictionaries to a pandas DataFrame
    collocated_df = pd.DataFrame(collocated_data)
    print('nb WV files opened and cached:', len(wv_file_cache))
    print("\nCollocation complete. Here's a peek at the resulting DataFrame:")
    print(collocated_df.head())
    print(f"Total successful collocated points: {len(collocated_df)}")

    print("\nSummary of collocation outcomes (global counts):")
    # Convert defaultdict to a regular dict for printing
    print(dict(collocation_outcome_counts))
    print(f"\nTotal Spotter points processed: {nb_points_to_process}")
    # Note: The sum of 'SuccessfulCollocations', 'NoWVFilesFound', and 'NoCloseWVPoints'
    # should add up to the total number of Spotter *points* in your original dataset,
    # as each Spotter point (row in sofards) gets categorized into one of these.
    # 'SuccessfulCollocations' here counts individual SAR matches, so if one Spotter point
    # matches multiple SAR points, it contributes multiple times to this count.
    # The other two counts ('NoWVFilesFound', 'NoCloseWVPoints') correctly count Spotter points (singular events).
    # For a count of unique Spotter points that had a successful collocation, you'd need
    # to derive it from collocated_df.

    if not collocated_df.empty:
        unique_spotter_points_with_successful_collocation = collocated_df.groupby(['timeSpotter', 'lonSpotter', 'latSpotter']).ngroups
        # Example: Number of unique Spotter observations that resulted in at least one collocation
        unique_successful_spotter_points = collocated_df.groupby(['timeSpotter', 'lonSpotter', 'latSpotter']).ngroups
        print(f"Number of unique Spotter observations with at least one successful collocation: {unique_successful_spotter_points}")

    # To make the global counts truly represent Spotter points (not individual SAR matches for 'SuccessfulCollocations'),
    # we'd need to adjust the counting logic slightly for 'SuccessfulCollocations' or
    # create a separate count for "Spotter points that successfully collocated".
    # For now, 'SuccessfulCollocations' is the total number of individual (Spotter, SAR) pairs.
    return collocated_df, dict(collocation_outcome_counts)

if __name__ == '__main__':
    sofar_file = '/home/datawork-cersat-public/provider/sofar-ocean/drifter/spotter/opendata/spotter_data_bulk.nc'
    sofards = xr.open_dataset(sofar_file).load()
    spa = 0.5 # deg
    deltah = 1 # hours
    import logging
    logging.basicConfig(level=logging.INFO)
    logging.info('start')
    collocated_df,collocation_outcome_counts = sofar_wv_colocation(sofards,dev=False,spatial_tolerance=spa,time_tolerance_hours=deltah)
    fout = '/scale/project/lops-siam-sentinel1-workbench/data/ifremer/coaligned_datasets/predictions/waves/sofar_spotter_WV_coaligned_dataset_deltaT_%shour_deltaG_%sdeg.csv'%(deltah,spa)
    collocated_df.to_csv(fout)
    print(fout)
