"""
GAS
22 August 25
methods to associate a dataset WV with SWH with buoys
context: Round Robin CCI sea state + WV SWH benchmark
"""

import xarray as xr
import pandas as pd
from tqdm import tqdm
import numpy as np
from haversine import haversine_vector, Unit
from datetime import timedelta
import collections

import pandas as pd

def get_buoy_data():
    """
    Lit les données des bouées depuis un fichier CSV et prépare le DataFrame.
    """
    # 1. Lire les données des bouées
    buoy_df = pd.read_csv('/home/datawork-cersat-public/provider/cci_seastate/round-robin/sar/validation/insitu/colocations/RR_s1wv_2019_ndbc_buoy-collocated-orbits-under-50km.dat',
                        sep='\s+')

    # 2. Créez la colonne 'time'
    time_cols_map = {'year': buoy_df['YYYY'], 'month': buoy_df['MM'], 'day': buoy_df['DD'], 'hour': buoy_df['HH'], 'minute': buoy_df['MM.1']}
    buoy_df['time'] = pd.to_datetime(time_cols_map)

    # 3. Assurez-vous que les colonnes de latitude et longitude ont les bons noms, le script s'attend à 'lat' et 'lon'
    # Dans votre cas, elles s'appellent 'LAT' et 'LON', ce qui peut causer des erreurs. Renommons-les.
    buoy_df = buoy_df.rename(columns={'LAT': 'lat', 'LON': 'lon'})
    return buoy_df






def create_mock_satellite_data(file_path):
    """Crée un fichier NetCDF de données satellites fictives."""
    time = pd.to_datetime(pd.date_range(start='2023-01-01 00:00:00', end='2023-01-01 23:59:59', freq='1min'))
    # Création d'une trajectoire de satellite simple
    lon = np.linspace(0, 10, len(time)) + np.random.randn(len(time)) * 0.1
    lat = np.linspace(50, 55, len(time)) + np.random.randn(len(time)) * 0.1
    
    ds = xr.Dataset(
        {
            'lon': ('time', lon),
            'lat': ('time', lat),
            'sea_surface_temperature': ('time', 273.15 + 10 + np.random.randn(len(time)))
        },
        coords={'time': time}
    )
    ds.to_netcdf(file_path)
    return ds

def create_mock_buoy_data():
    """Crée un DataFrame de données de bouées fictives."""
    data = {
        'time': pd.to_datetime(['2023-01-01 08:05:00', '2023-01-01 12:30:00', '2023-01-01 18:45:00']),
        'lon': [2.5, 5.1, 8.2],
        'lat': [51.0, 52.5, 54.0],
        'buoy_id': ['bouee_A', 'bouee_B', 'bouee_C']
    }
    return pd.DataFrame(data)

def colocate_satellite_buoy(satellite_ds, buoy_df, time_threshold_minutes=60, distance_threshold_km=50):
    """
    Co-localise les données satellites avec les données des bouées.

    Args:
        satellite_ds (xr.Dataset): Dataset Xarray contenant les données satellites.
        buoy_df (pd.DataFrame): DataFrame Pandas contenant les données des bouées.
        time_threshold_minutes (int): Seuil de temps en minutes.
        distance_threshold_km (int): Seuil de distance en kilomètres.

    Returns:
        pd.DataFrame: Un DataFrame contenant les paires de données co-localisées.
    """
    
    matchups = []

    # S'assurer que les colonnes de temps sont au format datetime
    satellite_ds['time'] = pd.to_datetime(satellite_ds['time'].values)
    buoy_df['time'] = pd.to_datetime(buoy_df['time'])
    cpt = collections.defaultdict(int)
    # Itérer sur chaque mesure de bouée
    for index in tqdm(range(len(buoy_df))):
    # for index, buoy_row in buoy_df.iterrows():
        buoy_row = buoy_df.iloc[index]
        buoy_time = buoy_row['time']
        buoy_coords = (buoy_row['lat'], buoy_row['lon'])

        # 1. Filtrage temporel
        # Convertissez-le en np.datetime64 avant de l'utiliser pour le slicing
        buoy_time_np = np.datetime64(buoy_time)
    
        # 1. Filtrage temporel en utilisant la version numpy
        time_min = buoy_time_np - np.timedelta64(int(time_threshold_minutes / 2), 'm')
        time_max = buoy_time_np + np.timedelta64(int(time_threshold_minutes / 2), 'm')
        # print('time_min',time_min,type(time_min))
        # time_min = buoy_time - timedelta(minutes=time_threshold_minutes / 2)
        # time_max = buoy_time + timedelta(minutes=time_threshold_minutes / 2)
        ids = np.where((satellite_ds.time.values>=time_min) & (satellite_ds.time.values<=time_max))[0]
        satellite_subset = satellite_ds.isel(time=ids)
        # satellite_subset = satellite_ds.sel(time=slice(time_min, time_max))

        if not satellite_subset.time.size:
            continue

        # 2. Filtrage spatial
        # Prépare les coordonnées pour le calcul de distance vectorisé
        satellite_coords = list(zip(satellite_subset['lat'].values, satellite_subset['lon'].values))

        # Calcule la distance de la bouée à chaque point satellite du sous-ensemble
        distances = haversine_vector([buoy_coords] * len(satellite_coords), satellite_coords, unit=Unit.KILOMETERS)
        
        satellite_subset['distance_to_buoy'] = ('time', distances)

        # Sélectionne les points satellites dans le rayon de distance
        nearby_satellite_points = satellite_subset.where(satellite_subset.distance_to_buoy <= distance_threshold_km, drop=True)

        if not nearby_satellite_points.time.size:
            cpt['no_time_matchup'] += 1
            continue
            
        # 3. Trouve le point satellite le plus proche dans le sous-ensemble
        closest_satellite_point = nearby_satellite_points.isel(time=nearby_satellite_points.distance_to_buoy.argmin())

        # Ajoute les informations de la bouée au point satellite trouvé
        # print(closest_satellite_point)
        if closest_satellite_point['time'].values:
            if not isinstance(closest_satellite_point['time'].values,np.ndarray): #case one single value
                closest_satellite_point = closest_satellite_point.expand_dims('time')
            cpt['matchup'] += 1
            matchup = closest_satellite_point.to_dataframe().reset_index()
            for col in buoy_df.columns:
                matchup[f'buoy_{col}'] = buoy_row[col]
        else:
            cpt['no_geo_matchup'] += 1
        
        matchups.append(matchup)

    if not matchups:
        return pd.DataFrame()
    print('counter',cpt)
    return pd.concat(matchups, ignore_index=True)

# --- Programme Principal ---
if __name__ == '__main__':
    # Création des données de démonstration
    satellite_file = 'satellite_data.nc'
    create_mock_satellite_data(satellite_file)
    buoy_data = create_mock_buoy_data()

    # Chargement des données
    satellite_dataset = xr.open_dataset(satellite_file)
    
    print("Données Satellites (extrait):")
    print(satellite_dataset)
    print("\nDonnées Bouées:")
    print(buoy_data)

    # Exécution de la co-localisation
    colocated_data = colocate_satellite_buoy(satellite_dataset, buoy_data)

    print("\n--- Résultats de la Co-localisation ---")
    if colocated_data.empty:
        print("Aucune co-localisation trouvée.")
    else:
        print(colocated_data)