import os

import numpy as np
import xarray as xr


read_path  = '/home/hanna/lagrings/ERA5_monthly/'
write_path = '/home/hanna/lagrings/results/ar/'

def get_xarray_dataset_for_period(start = '2012-01-01', stop = '2012-03-31'):
    """ Gets data from the requested period into a xarray dataset. """
    #from utils import merge
    files = get_list_of_files(start = '2012-01-01', stop = '2012-03-31')
    data = merge(files)
    data = data.sel(time = slice(start, stop))
    return data

def get_list_of_files(start = '2012-01-01', stop = '2012-01-31'):
    """ Returns list of files containing data for the requested period.
    """
    # Remove date.
    parts = start.split('-')
    start_search_str = '{}_{:02d}'.format(parts[0], int(parts[1]))

    parts = stop.split('-')
    stop_search_str = '{}_{:02d}'.format(parts[0], int(parts[1]))

    if start_search_str == stop_search_str:
        subset = glob.glob(os.path.join( read_path, '{}*.nc'.format(start_search_str)))
    else:
        min_fil = os.path.join(read_path, start_search_str + '_q.nc')
        max_fil = os.path.join(read_path, stop_search_str + '_tcc.nc')

        # get all files
        files = glob.glob(os.path.join( read_path, '*.nc' ))
        files = np.sort(files) # sorting then for no particular reson

        smaller = files[files <= max_fil]
        subset  = smaller[smaller >= min_fil] # results in all the files
    assert len(subset)%5==0, "Not five of each files, missing variables in file list!"
    return subset


def dataset_to_numpy_grid(pixel, bias = True):
    """ Dataset """
    n_time     = len(pixel.time.values)
    n_lat = len(pixel.latitude.values)
    n_lon = len(pixel.longitude.values)

    if bias:
        num_vars = 4+1
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
    X[:, :, :, 4] = 1
    return X, tcc[:, :, :, np.newaxis]

def dataset_to_numpy_order(dataset, order):
    """ Generates a dataset from the
    dataset : xr.Dataset
        Contains the data you want to make a prediction based.

    TODO add explination column???

    keep the order of xarray time, lat, lon
    """
    times = dataset.time.values
    X = np.zeros( (len(times)-order, n_lat, n_lon, order+5) )
    y = np.zeros( (len(times)-order, n_lat, n_lon) )
    # X.shape = (lat, lon, variables, times)

    q   = dataset.q.values
    t2m = dataset.t2m.values
    r   = dataset.r.values
    sp  = dataset.sp.values
    tcc = dataset.tcc.values

    X[:, :, :, 0] = q[:-order,:,:]
    X[:, :, :, 1] = t2m[:-order,:,:]
    X[:, :, :, 2] = r[:-order,:,:]
    X[:, :, :, 3] = sp[:-order,:,:]
    X[:, :, :, 4] = 1 # bias

    y[:, :, :] = tcc[:-order]

    index = 5

    # tcc1, tcc2, ..., tcc_n
    for temp_order in range(1, order+1):
        a = times[:-temp_order]
        b = times[temp_order:]
        bo = [element.astype(int) == temp_order for element in (b-a).astype('timedelta64[h]') ]

        remove_from_end = order - temp_order
        if remove_from_end != 0:
            #remove_from_end = 1
            # Which clouds to add at which column, remember that they shoudl start from t-1, t-2, t-3 ...
            X[:, :, :, index] = tcc[temp_order:, :, :][bo][:-remove_from_end, :, :]
        else:
            X[:, :, :, index] = tcc[temp_order:, :, :][bo]
        index+=1
    return X, y

def dataset_to_numpy(pixel, bias = True):
    """ Dataset. """
    print('Warning this does not include prior timesteps of tcc ... '+
            'Use dtaset_to_numpy_order() available the same directory. ')
    n = len(pixel.time.values)

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
    X[:, 4] = 1
    return X, tcc[:, np.newaxis]


def mean_squared_error(y_true, y_pred):
    """Computes the Mean Squared Error score metric.

    Parameteres
    ----------------
    y_true : array-like
    y_pred : array-like

    Returns
    ----------------
    mse : float

    """
    mse = np.square(np.subtract(y_true, y_pred)).mean(axis = 0)
    return mse


def accumulated_squared_error(y_true, y_pred):
    """Computes the Mean Squared Error score metric.

    Parameteres
    ----------------
    y_true : array-like
    y_pred : array-like

    Returns
    ----------------
    ase : float

    """
    ase = np.square(np.subtract(y_true, y_pred)).sum(axis = 0)
    return ase


def r2_score(y_true, y_pred):
    """ Computes the R2 score score metric.

    Parameteres
    ----------------
    y_true : array-like
    y_pred : array-like

    Returns
    ----------------
    r2 : float
    """
    numerator   = np.square(np.subtract(y_true, y_pred)).sum(axis=0)
    denominator = np.square(np.subtract(y_true, np.average(y_true))).sum(axis=0)
    val = numerator/denominator
    return 1 - val


def merge(files):
    """ Merging a list of filenames into a dataset.
    Parameteres
    -----------
    files : List[str]

    Returns
    ------------
        xr.dataset
    """
    #assert len(files) == 5
    datasets = [xr.open_dataset(fil) for fil in files]
    return xr.merge(datasets)

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
    data = merge(files)
    return data.sel(latitude = lat, longitude = lon)

def get_pixel_from_ds(ds, lat, lon):
    return ds.sel(latitude = lat, longitude = lon)

def fit_pixel(X, y):
    from scipy.linalg import inv
    coeffs = np.dot(inv(np.dot(X.T, X)), np.dot(X.T, y))
    return coeffs

def predict_pixel(X, coeffs):
    """Make prediction of one pixel """
    return np.dot(X, coeffs)
