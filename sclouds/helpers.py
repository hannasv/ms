import numpy as np
import matplotlib.pyplot as plt
import glob
import xarray as xr

LAT = (30,50)
LON = (-15,25)

EXTENT = [LAT, LON]

VARIABLES =  ["t2m", 'sp', 'q', 'r', 'tcc']

import os

import numpy as np
import xarray as xr

# directories currently in use
path_read_data       = '/home/hanna/lagrings/ERA5_monthly/'

path_ar_results       = '/home/hanna/lagrings/results/ar/'
path_convlstm_results = '/home/hanna/lagrings/results/convlstm/'
path_stats_results    = '/home/hanna/lagrings/results/stats/'



def generate_output_file_name_trained_ar_model():
    """ Generates output file name, contain all information about the training
    prosedure.

    """

    pass

def get_list_of_trained_ar_models():
    pass

def get_list_of_trained_conv_lstm_models():
    pass
