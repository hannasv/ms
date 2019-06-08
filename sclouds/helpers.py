import numpy as np
import matplotlib.pyplot as plt


DATA_REPO = "/uio/lagringshotell/geofag/students/metos/hannasv/data_processed/"
FIGURE_REPO = "/uio/lagringshotell/geofag/students/metos/hannasv/figures/"
RAW_ERA_REPO = "/uio/lagringshotell/geofag/students/metos/hannasv/era_interim_data/"
RESULTS_REPO =  "/uio/lagringshotell/geofag/students/metos/hannasv/results/"

LAT = (30,67)
LON = (-15,42)

VARIABLES =  ["t2m", 'sp', 'q', 'r', 'tcc']
PRESSURE_LEVELS = [300, 400, 500, 700, 850, 1000]
