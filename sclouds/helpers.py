import numpy as np
import matplotlib.pyplot as plt


DATA_REPO = "/uio/lagringshotell/geofag/students/metos/hannasv/data_processed/"
FIGURE_REPO = "/uio/lagringshotell/geofag/students/metos/hannasv/figures/"
RAW_ERA_REPO = "/uio/lagringshotell/geofag/students/metos/hannasv/era_interim_data/"
RESULTS_REPO =  "/uio/lagringshotell/geofag/students/metos/hannasv/results/"
LAPTOP_REPO = '/home/hanna/Desktop/master_thesis/era/'
LAPTOP_RESULTS_REPO = '/home/hanna/Desktop/master_thesis/test_results/'

LAT = (30,67)
LON = (-15,42)

EXTENT = [LAT, LON]
# TODO : better name for this one

VARIABLES =  ["t2m", 'sp', 'q', 'r', 'tcc']
PRESSURE_LEVELS = [300, 400, 500, 700, 850, 1000]

# TODO add get_season function.
def get_season(data, season):
    """
    data : xarray dataset
    season : str

    Returns the xarray dataset containing only one season.
    """
    data = data.groupby('time.season')
    for group in data:
        key, dataset = group
        if key == season:
            return dataset
    return
