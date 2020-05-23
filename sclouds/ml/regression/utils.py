import os, sys

import numpy as np
import xarray as xr

def get_list_of_files_era5(start = '2012-01-01', stop = '2012-01-31', include_start = True, include_stop = True):
    """ Returns list of files containing data for the requested period.

    Parameteres
    ----------------------
    start : str
        Start of period. First day included. (default '2012-01-01')

    stop : str
        end of period. Last day included. (default '2012-01-31')

    Returns
    -----------------------
    subset : List[str]
        List of strings containing all the absolute paths of files containing
        data in the requested period.
    """
    # Remove date.
    parts = start.split('-')
    start_search_str = '{}_{:02d}'.format(parts[0], int(parts[1]))

    if stop is not None:
        parts = stop.split('-')
        stop_search_str = '{}_{:02d}'.format(parts[0], int(parts[1]))
    else:
        stop_search_str = ''
    path_input = '/uio/lagringshotell/geofag/students/metos/hannasv/ERA5_tcc/'
    if (start_search_str == stop_search_str) or (stop is None):
        subset = glob.glob(os.path.join( path_input, '{}*tcc*.nc'.format(start_search_str)))
    else:
        # get all files
        files = glob.glob(os.path.join( path_input, '*.nc' ))
        files = np.sort(files) # sorting then for no particular reson
        #print(files)
        min_fil = os.path.join(path_input, start_search_str + '_tcc_era.nc')
        max_fil = os.path.join(path_input, stop_search_str + '_tcc_era.nc')
        #print(min_fil)
        #print(max_fil)
        if include_start and include_stop:
            smaller = files[files <= max_fil]
            subset  = smaller[smaller >= min_fil] # results in all the files

        elif include_start and not include_stop:
            smaller = files[files < max_fil]
            subset  = smaller[smaller >= min_fil] # results in all the files

        elif not include_start and include_stop:
            smaller = files[files <= max_fil]
            subset  = smaller[smaller > min_fil] # results in all the files
        else:
            raise ValueError('Something wierd happend. ')
    return subset

def get_list_of_files_traditional_model(start = '2012-01-01', stop = '2012-01-31', include_start = True, include_stop = True):
    """ Returns list of files containing data for the requested period.

    Parameteres
    ----------------------
    start : str
        Start of period. First day included. (default '2012-01-01')

    stop : str
        end of period. Last day included. (default '2012-01-31')

    Returns
    -----------------------
    subset : List[str]
        List of strings containing all the absolute paths of files containing
        data in the requested period.
    """
    # Remove date.
    parts = start.split('-')
    start_search_str = '{}_{:02d}'.format(parts[0], int(parts[1]))

    if stop is not None:
        parts = stop.split('-')
        stop_search_str = '{}_{:02d}'.format(parts[0], int(parts[1]))
    else:
        stop_search_str = ''

    if (start_search_str == stop_search_str) or (stop is None):
        subset = glob.glob(os.path.join( path_input, '{}*tcc*.nc'.format(start_search_str)))
    else:
        # get all files
        files = glob.glob(os.path.join( path_input, '*.nc' ))
        files = np.sort(files) # sorting then for no particular reson

        min_fil = os.path.join(path_input, start_search_str + '_tcc.nc')
        max_fil = os.path.join(path_input, stop_search_str + '_tcc.nc')

        if include_start and include_stop:
            smaller = files[files <= max_fil]
            subset  = smaller[smaller >= min_fil] # results in all the files

        elif include_start and not include_stop:
            smaller = files[files < max_fil]
            subset  = smaller[smaller >= min_fil] # results in all the files

        elif not include_start and include_stop:
            smaller = files[files <= max_fil]
            subset  = smaller[smaller > min_fil] # results in all the files
        else:
            raise ValueError('Something wierd happend. ')
    return subset


def get_list_of_files_excluding_period_traditional_model(start = '2012-01-01', stop = '2012-01-31'):

    first_period = get_list_of_files_traditional_model(start = '2004-04-01', stop = start,
                                include_start = True, include_stop = False)
    last_period = get_list_of_files_traditional_model(start = stop, stop = '2018-12-31',
                        include_start = False, include_stop = True)
    entire_period = list(first_period) + list(last_period)
    return entire_period

def get_list_of_files_excluding_period(start = '2012-01-01', stop = '2012-01-31'):

    first_period = get_list_of_files(start = '2004-04-01', stop = start,
                                include_start = True, include_stop = False)
    last_period = get_list_of_files(start = stop, stop = '2018-12-31',
                        include_start = False, include_stop = True)
    entire_period = list(first_period) + list(last_period)
    return entire_period

def get_list_of_files(start = '2012-01-01', stop = '2012-01-31', include_start = True, include_stop = True):
    """ Returns list of files containing data for the requested period.

    Parameteres
    ----------------------
    start : str
        Start of period. First day included. (default '2012-01-01')

    stop : str
        end of period. Last day included. (default '2012-01-31')

    Returns
    -----------------------
    subset : List[str]
        List of strings containing all the absolute paths of files containing
        data in the requested period.
    """

    parts = start.split('-')
    start_search_str = '{}_{:02d}'.format(parts[0], int(parts[1]))

    if stop is not None:
        parts = stop.split('-')
        stop_search_str = '{}_{:02d}'.format(parts[0], int(parts[1]))
    else:
        stop_search_str = ''

    if (start_search_str == stop_search_str) or (stop is None):
        subset = glob.glob(os.path.join( path_input, '{}*.nc'.format(start_search_str)))
    else:
        # get all files
        files = glob.glob(os.path.join( path_input, '*.nc' ))
        files = np.sort(files) # sorting then for no particular reson

        if include_start and include_stop:
            min_fil = os.path.join(path_input, start_search_str + '_q.nc')
            max_fil = os.path.join(path_input, stop_search_str + '_tcc.nc')

            smaller = files[files <= max_fil]
            subset  = smaller[smaller >= min_fil] # results in all the files

        elif include_start and not include_stop:
            min_fil = os.path.join(path_input, start_search_str + '_q.nc')
            max_fil = os.path.join(path_input, stop_search_str + '_q.nc')

            smaller = files[files < max_fil]
            subset  = smaller[smaller >= min_fil] # results in all the files

        elif not include_start and include_stop:
            min_fil = os.path.join(path_input, start_search_str + '_tcc.nc')
            max_fil = os.path.join(path_input, stop_search_str + '_tcc.nc')

            smaller = files[files <= max_fil]
            subset  = smaller[smaller > min_fil] # results in all the files
        else:
            raise ValueError('Something wierd happend. ')

    #assert len(subset)%5==0, "Not five of each files, missing variables in file list!"
    #assert len(subset)!=0, "No files found, check if you have mounted lagringshotellet."

    return subset


def sigmoid(x):
    """ Computed the sigmoid transformation. Truncates real axis to  values in
    the range 0 and 1.

    Expression : np.exp(x)/(1 + np.exp(x)).

    Parameteres
    -------------------
    x : array-like
        Vector containing the values.

    Returnes
    --------------------
    s : array-like
        The sigmoid transform of x
    """
    s = np.exp(x)/(1 + np.exp(x))
    return s

def inverse_sigmoid(x):
    """Also known as the logit function. Expression np.log(x/(1-x).
    Use to transform the response to be in the range (-inf, +inf).

    Parameteres
    -------------------
    x : array-like
        Vector containing the values.

    Returnes
    --------------------
    _ : array-like
        The inverse sigmoid transform of x
    """
    return np.log(x/(1-x + 0.0000001))

def mean_squared_error(y_true, y_pred):
    """Computes the Mean Squared Error score metric.

    Parameteres
    ------------------
    y_true : array-like
        Actual vales of y.
    y_pred : array-like
        Predicted values of y.

    Returns
    -------------------
    mse : float
        mean squared error
    """
    mse = np.nanmean(np.square(np.subtract(y_true, y_pred)), axis = 0)
    return mse


def accumulated_squared_error(y_true, y_pred):
    """Computes the Mean Squared Error score metric.

    Parameteres
    ----------------
    y_true : array-like
        Actual vales of y.
    y_pred : array-like
        Predicted values of y.

    Returns
    ----------------
    ase : float
        Accumulated squared error between y_true and y_pred.
    """
    ase = np.nansum(np.square(np.subtract(y_true, y_pred)), axis = 0)
    return ase


def r2_score(y_true, y_pred):
    """ Computes the R2 score score metric.

    Parameteres
    ---------------------------
    y_true : array-like
        Actual vales of y.
    y_pred : array-like
        Predicted values of y.

    Returns
    ----------------------------
    r2 : float
         Coefficient of determination.

    Notes
    -----------
    Describes variation of data captured by the model.
    """
    numerator   = np.nansum(np.square(np.subtract(y_true, y_pred)), axis=0)
    denominator = np.nansum(np.square(np.subtract(y_true, np.nanmean(y_true))), axis = 0)
    val = numerator/denominator
    return 1 - val


def fit_pixel(X, y):
    """Traines one pixel of the grid.

    Parameteres
    -----------------

    Returns
    -------------
    coeffs :
    """
    from scipy.linalg import inv
    #coeffs = np.dot(pinv(np.dot(X.T, X)), np.dot(X.T, y))
    coeffs = np.dot(inv(np.dot(X.T, X)), np.dot(X.T, y))
    return coeffs

def predict_pixel(X, coeffs):
    """Make prediction of one pixel. Return"""
    return np.dot(X, coeffs)

""" Utilities for reading routines.
"""

import os
import glob

import numpy as np
import xarray as xr

#sys.path.insert(0,'/uio/hume/student-u89/hannasv/MS/sclouds/')
from sclouds.helpers import merge, path_input

def get_list_of_files_traditional_model(start = '2012-01-01', stop = '2012-01-31', include_start = True, include_stop = True, var = 'tcc'):
    """ Returns list of files containing data for the requested period.

    Parameteres
    ----------------------
    start : str
        Start of period. First day included. (default '2012-01-01')

    stop : str
        end of period. Last day included. (default '2012-01-31')

    Returns
    -----------------------
    subset : List[str]
        List of strings containing all the absolute paths of files containing
        data in the requested period.
    """
    # Remove date.
    parts = start.split('-')
    start_search_str = '{}_{:02d}'.format(parts[0], int(parts[1]))

    if stop is not None:
        parts = stop.split('-')
        stop_search_str = '{}_{:02d}'.format(parts[0], int(parts[1]))
    else:
        stop_search_str = ''

    if (start_search_str == stop_search_str) or (stop is None):
        subset = glob.glob(os.path.join( path_input, '{}*{}*.nc'.format(start_search_str, var)))
    else:
        # get all files
        files = glob.glob(os.path.join( path_input, '*.nc' ))
        files = np.sort(files) # sorting then for no particular reson

        min_fil = os.path.join(path_input, start_search_str + '_{}.nc'.format(var))
        max_fil = os.path.join(path_input, stop_search_str + '_{}.nc'.format(var))

        if include_start and include_stop:
            smaller = files[files <= max_fil]
            subset  = smaller[smaller >= min_fil] # results in all the files

        elif include_start and not include_stop:
            smaller = files[files < max_fil]
            subset  = smaller[smaller >= min_fil] # results in all the files

        elif not include_start and include_stop:
            smaller = files[files <= max_fil]
            subset  = smaller[smaller > min_fil] # results in all the files
        else:
            raise ValueError('Something wierd happend. ')
    return subset



def get_xarray_dataset_for_period(start = '2012-01-01', stop = '2012-01-31'):
    """ Reads data from the requested period into a xarray dataset.
    I stop is not provided it defaults to one month of data.

    Parameteres
    ----------------------
    start : str
        Start of period. First day included. (default '2012-01-01')
    stop : str, optional
        end of period. Last day included. (default '2012-01-31')

    Returns
    -----------------------
    data : xr.Dataset
        Dataset including all variables in the requested period.
    """
    #from utils import merge
    files = get_list_of_files(start = start, stop = stop)
    #print("Num files {}".format(len(files)))
    data = merge(files)
    if stop is not None:
        data = data.sel(time = slice(start, stop))
    return data

def get_list_of_files_excluding_period(start = '2012-01-01', stop = '2012-01-31'):

    first_period = get_list_of_files(start = '2004-04-01', stop = start, include_start = True, include_stop = False)
    last_period = get_list_of_files(start = stop, stop = '2018-12-31', include_start = False, include_stop = True)
    #print(first_period)
    #print(last_period)
    entire_period = list(first_period) + list(last_period)
    return entire_period

def get_list_of_files(start = '2012-01-01', stop = '2012-01-31', include_start = True, include_stop = True):
    """ Returns list of files containing data for the requested period.

    Parameteres
    ----------------------
    start : str
        Start of period. First day included. (default '2012-01-01')

    stop : str
        end of period. Last day included. (default '2012-01-31')

    Returns
    -----------------------
    subset : List[str]
        List of strings containing all the absolute paths of files containing
        data in the requested period.
    """
    print('enters, searching in directory {}'.format(path_input))
    # Remove date.
    parts = start.split('-')
    start_search_str = '{}_{:02d}'.format(parts[0], int(parts[1]))

    if stop is not None:
        parts = stop.split('-')
        stop_search_str = '{}_{:02d}'.format(parts[0], int(parts[1]))
    else:
        stop_search_str = ''

    if (start_search_str == stop_search_str) or (stop is None):
        subset = glob.glob(os.path.join( path_input, '{}*.nc'.format(start_search_str)))
    else:
        # get all files
        files = glob.glob(os.path.join( path_input, '*.nc' ))
        files = np.sort(files) # sorting then for no particular reson

        if include_start and include_stop:
            min_fil = os.path.join(path_input, start_search_str + '_q.nc')
            max_fil = os.path.join(path_input, stop_search_str + '_tcc.nc')

            smaller = files[files <= max_fil]
            subset  = smaller[smaller >= min_fil] # results in all the files

        elif include_start and not include_stop:
            min_fil = os.path.join(path_input, start_search_str + '_q.nc')
            max_fil = os.path.join(path_input, stop_search_str + '_q.nc')

            smaller = files[files < max_fil]
            subset  = smaller[smaller >= min_fil] # results in all the files

        elif not include_start and include_stop:
            min_fil = os.path.join(path_input, start_search_str + '_tcc.nc')
            max_fil = os.path.join(path_input, stop_search_str + '_tcc.nc')

            smaller = files[files <= max_fil]
            subset  = smaller[smaller > min_fil] # results in all the files
        else:
            raise ValueError('Something wierd happend. ')
    return subset

def dataset_to_numpy_grid_keras(pixel, seq_length):
    """ Takes a xr.dataset and transforms it to a numpy matrix.

    Parameteres
    -------------
    pixel : xr.dataset
        Dataset containing (a pixel) timeseries of all variables.

    seq_length : int
        Sets length of training sequence.

    Returns
    ---------------------
    X : array-like
        Matrix containing the explanatory variables.
    y : array-like
        Responce variable.
    """
    n_time = len(pixel.time.values)
    n_lat  = len(pixel.latitude.values)
    n_lon  = len(pixel.longitude.values)
    num_vars = 4
    X = np.zeros((n_time, n_lat, n_lon, num_vars))

    q   = pixel.q.values
    t2m = pixel.t2m.values
    r   = pixel.r.values
    sp  = pixel.sp.values
    tcc = pixel.tcc.values

    X[:, :, :, 0] = q
    X[:, :, :, 1] = t2m
    X[:, :, :, 2] = r
    X[:, :, :, 3] = sp

    y = tcc[:, :, :, np.newaxis]
    num_samples, lat, ln, num_vars = X.shape

    # Reshapes data into sequence
    X = X.reshape((int(num_samples/seq_length), seq_length, n_lat, n_lon, num_vars))
    y = y.reshape((int(num_samples/seq_length), seq_length, n_lat, n_lon, 1))
    return X, y

def train_test_split_keras(dataset, seq_length, val_split = 0.2):
    """ Train test split used for keras.
    Usage input for hyperparam tuning.

    Parameteres
    -------------
    dataset : xr.dataset
        Dataset containing

    seq_length : int
        Length of training sequence applied for dataset.

    val_split : float
        Validation split. Default = 0.2

    Returns
    ---------------------
    X_train, y_train : array-like
        Data used to train model.

    X_test, y_test : array-like
        Part of data used to valibrate the data.
    """
    X, y = dataset_to_numpy_grid_keras(dataset, seq_length)

    num_samples, seq_length, _, _, num_vars = X.shape
    last_train_idx = int((1-val_split)*num_samples)

    X_train = X[:last_train_idx, :, :, :]
    y_train = y[:last_train_idx, :,]

    X_test = X[last_train_idx:, :, :, :]
    y_test = y[last_train_idx:, :,]
    return X_train, y_train, X_test, y_test

def dataset_to_numpy_grid(dataset, bias = True):
    """ Takes a xr.dataset and transforms it to a numpy matrix.

    Parameteres
    -------------
    dataset : xr.dataset
        Dataset containing

    val_split : float
        Validation split. Default = 0.2

    Returns
    ---------------------
    X : array-like
        Training data for statistical models.
    y : array-like

    """
    n_time = len(dataset.time.values)
    n_lat  = len(dataset.latitude.values)
    n_lon  = len(dataset.longitude.values)

    num_vars = 4
    # If bias is included the number of variables is increased.
    if bias:
        num_vars = 4+1
    X = np.zeros((n_time, n_lat, n_lon, num_vars))

    q   = dataset.q.values
    t2m = dataset.t2m.values
    r   = dataset.r.values
    sp  = dataset.sp.values
    tcc = dataset.tcc.values

    X[:, :, :, 0] = q
    X[:, :, :, 1] = t2m
    X[:, :, :, 2] = r
    X[:, :, :, 3] = sp

    if bias:
        X[:, :, :, 4] = 1

    return X, tcc[:, :, :, np.newaxis]

def dataset_to_numpy_order_traditional_ar_grid(dataset, order, bias = True):
    """ Tranforms a dataset to matrices.

    Parameters
    ----------------------------
    dataset : xr.Dataset
        Contains the data you want to make a prediction based.
    order : float
        The number of previos timesteps included as predictors.
    bias : bool
        Determines weather to include a bias column or not (default True)
    keep the order of xarray time, lat, lon

    Returns
    ---------------------
    X : array-like
        Matrix containing the explanatory variables.
    y : array-like
        Responce variable.

    Notes
    --------------------------
    Index description:

    5 (4) - tcc previos time step

    """

    if bias:
        var_index = 1
    else:
        var_index = 0

    times  = dataset.time.values
    n_time = len(dataset.time.values) - order
    n_lat  = len(dataset.latitude.values)
    n_lon  = len(dataset.longitude.values)


    X = np.zeros( (n_time, n_lat, n_lon, order + var_index) )
    y = np.zeros( (n_time, n_lat, n_lon) )

    tcc = dataset.tcc.values

    if bias:
        X[:, :, :, 0] = 1 # bias

    y[:, :, :] = tcc[:-order]

    # tcc1, tcc2, ..., tcc_n
    for temp_order in range(1, order+1):
        a = times[:-temp_order]
        b = times[temp_order:]
        bo = [element.astype(int) == temp_order for element in (b-a).astype('timedelta64[h]') ]

        remove_from_end = order - temp_order
        if remove_from_end != 0:
            X[:, :, :, var_index] = tcc[temp_order:, :, :][bo][:-remove_from_end, :, :]
        else:
            X[:, :, :, var_index] = tcc[temp_order:, :, :][bo]
        var_index+=1
    #print(X.shape)
    #print(y.shape)
    return X, y


def dataset_to_numpy_order_traditional_ar(dataset, order, bias = True):
    """ Tranforms a dataset to matrices.

    Parameters
    ----------------------------
    dataset : xr.Dataset
        Contains the data you want to make a prediction based.
    order : float
        The number of previos timesteps included as predictors.
    bias : bool
        Determines weather to include a bias column or not (default True)
    keep the order of xarray time, lat, lon

    Returns
    ---------------------
    X : array-like
        Matrix containing the explanatory variables.
    y : array-like
        Responce variable.

    Notes
    --------------------------
    Index description:

    5 (4) - tcc previos time step

    """

    if bias:
        var_index = 1
    else:
        var_index = 0

    times = dataset.time.values
    #print("Detected {} samples.".format(len(times)))
    X = np.zeros( (len(times)-order, order + var_index))
    y = np.zeros( (len(times)-order ))

    tcc = dataset.tcc.values
    #print('len tcc {}'.format(len(tcc)))

    if bias:
        X[:, 0] = 1 # bias

    y = tcc[:len(times)-order, np.newaxis]
    #print('len y should be tcc - order {}'.format(len(y)))

    # tcc1, tcc2, ..., tcc_n
    for temp_order in range(1, order+1):
        remove_from_end = len(tcc) - (order - temp_order)
        #print('expected length : len(times)-order {}'.format(len(times)-order))
        a = tcc[:len(times)-order]
        #print('len(a) {}'.format(len(a)))
        b = tcc[slice(temp_order, remove_from_end)]
        #print('len(b) {}'.format(len(b)))
        bo = [element.astype(int) == temp_order for element in (b-a).astype('timedelta64[h]') ]
        #print('len(bo) {}'.format(len(bo)))
        ins = b.copy()
        ins[np.array(bo)] = np.nan
        #print('X shape {}'.format(X[:, var_index].shape))
        X[:, var_index] = ins
        var_index+=1
    #print(X.shape)
    #print(y.shape)
    return X, y




def dataset_to_numpy_grid_order(dataset, order, bias = True):
    """ Tranforms a dataset to a grid matrices, based on information on bias
    and order of the ar model.

    Parameters
    ----------------------------
    dataset : xr.Dataset
        Contains the data you want to make a prediction based.
    order : float
        The number of previos timesteps included as predictors.
    bias : bool
        Determines weather to include a bias column or not (default True)
    keep the order of xarray time, lat, lon

    Returns
    ---------------------
    X : array-like
        Matrix containing the explanatory variables.
    y : array-like
        Responce variable.

    Notes
    --------------------------
    Index description:

    0 - q, specific humidity
    1 - t2m, two meter temperature
    2 - r, relative humidity
    3 - sp, specific humitidy
    (4 - bias/ intercept)
    5 (4) - tcc previos time step
    """
    times  = dataset.time.values
    n_time = len(dataset.time.values) - order
    n_lat  = len(dataset.latitude.values)
    n_lon  = len(dataset.longitude.values)

    if bias:
        var_index = 5
    else:
        var_index = 4

    X = np.zeros( (n_time, n_lat, n_lon, order + var_index) )
    y = np.zeros( (n_time, n_lat, n_lon) )

    q   = dataset.q.values
    t2m = dataset.t2m.values
    r   = dataset.r.values
    sp  = dataset.sp.values
    tcc = dataset.tcc.values

    X[:, :, :, 0] = q[:-order,:,:]
    X[:, :, :, 1] = t2m[:-order,:,:]
    X[:, :, :, 2] = r[:-order,:,:]
    X[:, :, :, 3] = sp[:-order,:,:]

    if bias:
        X[:, :, :, 4] = 1 # bias

    y[:, :, :] = tcc[:-order]

    # tcc1, tcc2, ..., tcc_n
    for temp_order in range(1, order+1):
        #print("Adding order ")
        a = times[:-temp_order]
        b = times[temp_order:]
        bo = [element.astype(int) == temp_order for element in (b-a).astype('timedelta64[h]') ]

        remove_from_end = order - temp_order
        if remove_from_end != 0:
            X[:, :, :, var_index] = tcc[temp_order:, :, :][bo][:-remove_from_end, :, :]
        else:
            X[:, :, :, var_index] = tcc[temp_order:, :, :][bo]
        var_index+=1
    return X, y


def dataset_to_numpy_order(dataset, order, bias = True):
    """ Tranforms a dataset to matrices.

    Parameters
    ----------------------------
    dataset : xr.Dataset
        Contains the data you want to make a prediction based.
    order : float
        The number of previos timesteps included as predictors.
    bias : bool
        Determines weather to include a bias column or not (default True)
    keep the order of xarray time, lat, lon

    Returns
    ---------------------
    X : array-like
        Matrix containing the explanatory variables.
    y : array-like
        Responce variable.

    Notes
    --------------------------
    Index description:

    0 - q, specific humidity
    1 - t2m, two meter temperature
    2 - r, relative humidity
    3 - sp, specific humitidy
    (4 - bias/ intercept)
    5 (4) - tcc previos time step

    """

    if bias:
        var_index = 5
    else:
        var_index = 4

    times = dataset.time.values
    #print("Detected {} samples.".format(len(times)))
    X = np.zeros( (len(times)-order, order + var_index))
    y = np.zeros( (len(times)-order ))

    q   = dataset.q.values
    t2m = dataset.t2m.values
    r   = dataset.r.values
    sp  = dataset.sp.values
    tcc = dataset.tcc.values

    X[:, 0] = q[:len(times)-order]
    X[:, 1] = t2m[:len(times)-order]
    X[:, 2] = r[:len(times)-order]
    X[:, 3] = sp[:len(times)-order]

    if bias:
        X[:, 4] = 1 # bias

    y = tcc[:len(times)-order, np.newaxis]

    # tcc1, tcc2, ..., tcc_n
    for temp_order in range(1, order+1):
        remove_from_end = len(tcc) - (order - temp_order)
        #print('expected length : len(times)-order {}'.format(len(times)-order))
        a = tcc[:len(times)-order]
        #print('len(a) {}'.format(len(a)))
        b = tcc[slice(temp_order, remove_from_end)]
        #print('len(b) {}'.format(len(b)))
        bo = [element.astype(int) == temp_order for element in (b-a).astype('timedelta64[h]') ]
        #print('len(bo) {}'.format(len(bo)))
        ins = b.copy()
        ins[np.array(bo)] = np.nan
        #print('X shape {}'.format(X[:, var_index].shape))
        X[:, var_index] = ins
        var_index+=1
    return X, y


def dataset_to_numpy(pixel, bias = True):
    """ Takes a xr.dataset and transforms it to a numpy matrix.

    Parameteres
    -------------
    pixel : xr.dataset
        Dataset containing (a pixel) timeseries of all variables.

    bias : bool
        Determines it the bias should be included. (default True)

    Returns
    ---------------------
    X : array-like
        Matrix containing the explanatory variables.
    y : array-like
        Responce variable.
    """
    n = len(pixel.time.values)

    num_vars = 4
    if bias:
        num_vars = 4+1

    X = np.zeros((n, num_vars))
    q   = pixel.q.values
    t2m = pixel.t2m.values
    r   = pixel.r.values
    sp  = pixel.sp.values
    tcc = pixel.tcc.values

    X[:, 0] = q
    X[:, 1] = t2m
    X[:, 2] = r
    X[:, 3] = sp

    if bias:
        X[:, 4] = 1
    return X, tcc[:, np.newaxis]
