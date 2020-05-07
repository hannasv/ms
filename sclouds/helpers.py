import os
import glob

import xarray as xr
import numpy as np

LAT = (30,50)
LON = (-15,25)
SPATIAL_RESOLUTION = 0.25
TEMPORAL_RESOLUTION = 'h' # TODO: this need to be a proper dt format

EXTENT = [LAT, LON]
VARIABLES =  ["t2m", 'sp', 'q', 'r', 'tcc']

LONGNAME = {"t2m":"Temperature", 'q':"Specific Humidity",
            'sp':"Surface Pressure", 'r': "Relative Humidity",
            'tcc':"Cloud Fractional Cover"}

UNITS = {"t2m":"K", 'sp':"Pa", 'q':"kg kg^-1", 'r': "1", 'tcc':"1"}

STATISTICS = ['mean', 'min', 'max', 'std', 'median', 'mad']

FILTERS = ['coast', 'sea', 'land', 'artefact', 'all']

MONTHS  = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
            'August', 'September', 'October', 'November', 'December']


# The orginal TODO make dictionary of all partitions.
train_start = '2004-01-01'
train_stop  = '2013-12-31'

test_start  = '2014-01-01'
test_stop   = '2018-12-31'

# Directories currently in use
path_input            = '/home/hanna/lagrings/ERA5_monthly/'
path_ar_results       = '/home/hanna/lagrings/results/ar/'
path_convlstm_results = '/home/hanna/lagrings/results/convlstm/'
path_stats_results    = '/home/hanna/lagrings/results/stats/'

# duplicated, available in sclouds.plot.helpers
path_store_plots = '/home/hanna/MS-thesis/python_figs/'
path_filter     = '/home/hanna/MS-suppl/filters/'

def get_lat_array():
    """ Returns array of latitude values. Range [30, 50] and
    0.25 degree resolution.

    Returns
    -----------
     _ : array-like
        Numpy array of valid latitude values.
    """
    return np.arange(LAT[0], LAT[1]+SPATIAL_RESOLUTION,
                        step = SPATIAL_RESOLUTION)

def get_lon_array():
    """Returns array of longitude values. Range [-15, 25] and
    0.25 degree resolution.

    Returns
    -----------
     _ : array-like
        Numpy array of valid longitude values.
    """
    return np.arange(LON[0], LON[1]+SPATIAL_RESOLUTION,
                        step = SPATIAL_RESOLUTION)

def merge(files):
    """ Merging a list of filenames into a dataset.open_mfdataset

    Parameteres
    -----------
    files : List[str]
        List of abolute paths to files.

    Returns
    ------------
     _ : xr.dataset
        Merged files into one dataset.
    """
    #assert len(files) == 5
    #datasets = [xr.open_dataset(fil) for fil in files]
    #return xr.merge(datasets)
    return xr.open_mfdataset(files, compat='no_conflicts', join='outer')

def get_list_of_variables_in_ds(ds):
    """ Returs list of variables in dataset

    Parameteres
    ------------
    ds : xr.Dataset
        dataset

    Returns
    --------
    List[str] of vaiables in dataset

    """
    return  [k for k in ds.variables.keys()]

def get_pixel_from_files(files, lat, lon):
    """Select pixel in dataset based on coordinates.

    Parameters
    ------------
    files : List[str]
        List of abolute paths to files.

    lat :  float
        Requested latitude value.

    lon : float
        Requested longitude value.

    Returns
    ----------------
    ds : xr.Dataset
        Subset of ds, the selecting the requested coordinate.
    """

    data = merge(files)
    return data.sel(latitude = lat, longitude = lon)

def get_pixel_from_ds(ds, lat, lon):
    """Select pixel in dataset based on coordinates.

    Parameters
    ------------
    ds : xr.Dataset
        dataset

    lat :  float
        Requested latitude value.
    lon : float
        Requested longitude value.

    Returns
    ----------------
    ds : xr.Dataset
        Subset of ds, the selecting the requested coordinate.
    """
    return ds.sel(latitude = lat, longitude = lon)

def generate_output_file_name_trained_ar_model():
    """ Generates output file name, contain all information about the training
    prosedure.

    Raises
    ------------
    NotImplementedError
    """
    raise NotImplementedError('Comming soon ...')

def get_list_of_trained_ar_models():
    """ Returns list of trained convolutional lstm modelself.
    Raises
    ------------
    NotImplementedError
    """
    raise NotImplementedError('Comming soon ...')

def get_list_of_trained_conv_lstm_models():
    """ Returns list of trained convolutional lstm modelself.
    Raises
    ------------
    NotImplementedError
    """
    raise NotImplementedError('Comming soon ...')
