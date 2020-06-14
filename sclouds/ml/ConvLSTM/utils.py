"""
- If `return_sequences`
       - If data_format='channels_first'
          5D tensor with shape:
          `(samples, time, filters, output_row, output_col)`
       - If data_format='channels_last'
          5D tensor with shape:
          `(samples, time, output_row, output_col, filters)`
- Else
  - If data_format ='channels_first'
      4D tensor with shape:
      `(samples, filters, output_row, output_col)`
  - If data_format='channels_last'
      4D tensor with shape:
      `(samples, output_row, output_col, filters)`
"""
import os
import glob

import xarray as xr
import numpy as np

path_input = '/home/hannasv/data/'

def get_list_of_files_excluding_period(start = '2012-01-01', stop = '2012-01-31'):
    first_period = get_list_of_files(start = '2004-04-01', stop = start,
                                include_start = True, include_stop = False)
    last_period = get_list_of_files(start = stop, stop = '2018-12-31',
                        include_start = False, include_stop = True)
    entire_period = list(first_period) + list(last_period)
    return entire_period

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
    assert len(files) != 0, 'No files to merge'
    #datasets = [xr.open_dataset(fil) for fil in files]
    #return xr.merge(datasets)
    return xr.open_mfdataset(files, compat='no_conflicts') # , join='outer'

def get_train_test(test_start, test_stop, model = 'ar'):
    """Loads train and test data to datasets ... """
    #logger.info('Retrives data')
    # if type == 'ar':
    files_train = get_list_of_files_excluding_period(test_start, test_stop)
    files_test = get_list_of_files(test_start, test_stop)

    #else:
        #files_train = get_list_of_files_excluding_period_traditional_model(test_start, test_stop)
        #files_test = get_list_of_files_traditional_model(test_start, test_stop)

    #logger.info('Detected the relevant files. ')
    train_dataset = merge(files_train)
    #logger.info('Merged training data for {} to {}'.format(test_start,
                                    #test_stop))
    test_dataset = merge(files_test)
    #logger.info('Merged test data for {} to {}'.format(test_start, test_stop))
    return train_dataset, test_dataset

# Custom R2-score metrics for keras backend
def r2_keras(y_true, y_pred):
    import tensorflow.keras.backend as kb
    SS_res =  kb.sum(kb.square(y_true - y_pred))
    SS_tot = kb.sum(kb.square(y_true - kb.mean(y_true)))
    return ( 1 - SS_res/(SS_tot + kb.epsilon()) )

def mae(y_actual, y_predict):
    """
    #Custum keras loss function, accumulated squared error.
    """
    import tensorflow.keras.backend as kb
    return kb.sum(kb.abs(kb.subtract(y_actual, y_predict)), axis = 0)

def dataset_to_numpy_grid_keras_dataformat_channel_last(pixel, seq_length):
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

    pixel = replace_nans_with_values(1.5, pixel)
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

    samples, lat, ln, num_vars = X.shape
    print('shape of input X {}'.format(X.shape))
    print(samples/(seq_length))
    # Reshapes data into sequence
    try:
        X = X.reshape( ( int(samples/(seq_length)), seq_length, n_lat, n_lon, num_vars))
        y = y.reshape( ( int(samples/(seq_length)), seq_length, n_lat, n_lon) )
    except ValueError:
        print('enters except')
        dim = samples%(seq_length*batch_size)
        print('old tot num samples {}, new tot num samples {}'.format(n_time, dim))
        X_cropped = X[dim:, :, :, :]
        y_cropped = y[dim:, :, :, :]
        print('shape X cropped {}'.format(X_cropped.shape))
        X = X_cropped.reshape(( int(samples/(seq_length)), seq_length, n_lat, n_lon, num_vars))
        y = y_cropped.reshape((int(samples/(seq_length)), seq_length, n_lat, n_lon))
    print('shape of output X {}'.format(X.shape))
    print('shape of output X {}'.format(X.shape))
    print('shape of output y {}'.format(y.shape))

    return X, y

def dataset_to_numpy_grid_keras_dataformat_channel_last_batch_size(pixel, seq_length, batch_size):
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

    pixel = replace_nans_with_values(1.5, pixel)
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
    samples, lat, ln, num_vars = X.shape
    print('shape of input X {}'.format(X.shape))
    print(samples/(seq_length*batch_size))
    # Reshapes data into sequence
    try:
        X = X.reshape( ( int(samples/(seq_length*batch_size)), batch_size, seq_length, n_lat, n_lon, num_vars))
        y = y.reshape( ( int(samples/(seq_length*batch_size)), batch_size, seq_length, n_lat, n_lon) )
    except ValueError:
        print('enters except')
        dim = samples%(seq_length*batch_size)
        print('old tot num samples {}, new tot num samples {}'.format(n_time, dim))
        X_cropped = X[dim:, :, :, :]
        y_cropped = y[dim:, :, :, :]
        print('shape X cropped {}'.format(X_cropped.shape))
        X = X_cropped.reshape(( int(samples/(seq_length*batch_size)), batch_size, seq_length, n_lat, n_lon, num_vars))
        y = y_cropped.reshape((int(samples/(seq_length*batch_size)),
                                batch_size, seq_length, n_lat, n_lon))
    print('shape of output X {}'.format(X.shape))
    print('shape of output y {}'.format(y.shape))

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

def replace_nans_with_values(val, data):
    """ Returns copy of dataset filled with values
    """
    return data.fillna(val).copy()


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

    print('\n searchers for files')
    # Remove date.
    parts = start.split('-')
    start_search_str = '{}_{:02d}'.format(parts[0], int(parts[1]))

    if stop is not None:
        parts = stop.split('-')
        stop_search_str = '{}_{:02d}'.format(parts[0], int(parts[1]))
    else:
        stop_search_str = ''

    print('\n Searching in directory {} \n'.format( path_input))

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
            #print('detected min fil {}'.format(min_fil))
            max_fil = os.path.join(path_input, stop_search_str + '_tcc.nc')

            smaller = files[files <= max_fil]
            subset  = smaller[smaller > min_fil] # results in all the files
        else:
            raise ValueError('Something wierd happend. ')

    #assert len(subset)%5==0, "Not five of each files, missing variables in file list!"
    #assert len(subset)!=0, "No files found, check if you have mounted lagringshotellet."
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

    print("Num files {}".format(len(files)))
    data = xr.open_mfdataset(files, compat='no_conflicts') # , join='outer'(files)
    if stop is not None:
        data = data.sel(time = slice(start, stop))
    return data

def get_data_keras(dataset, num_samples = None, seq_length = 24,  batch_size = None,
                        data_format='channels_last'):
    """ """
    if num_samples is None:
        print('reads inn all available samples removing the last non complete values.')

    # Read in temperature, pressure and humidites.
    if data_format=='channels_first':
        # `(samples, time, filters, output_row, output_col)`
        raise NotImplementedError('Coming soon .. Use not implemend error.')
    elif data_format=='channels_last':
        #  `(samples, time, output_row, output_col, filters)`
        #  `(samples, time, output_row, output_col, filters)`
        if batch_size is None:
            X_train, y_train = dataset_to_numpy_grid_keras_dataformat_channel_last(dataset, seq_length)
        else:
            X_train, y_train = dataset_to_numpy_grid_keras_dataformat_channel_last_batch_size(dataset, seq_length, batch_size)
    else:
        raise ValueError('Not valid data_format try {}, {}'.format('channels_first',
                                                        'channels_last'))
    return X_train, y_train


if __name__ == '__main__':
    #import tensorflow as tf
    #from sclouds.io.utils import get_xarray_dataset_for_period
    data = get_xarray_dataset_for_period(start = '2012-01-01', stop = '2012-01-31')
    print(data)
    X, y = get_data_keras(data, num_samples = None, seq_length = 24, batch_size = 10,
                    data_format='channels_last')
    print(X.shape)
    print(y.shape)
