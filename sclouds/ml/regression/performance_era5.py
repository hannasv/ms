read_era5 = '/home/hanna/lagrings/ERA5_tcc/'
read_tcc  = '/home/hanna/lagrings/ERA5_monthly/'

import os
import glob

import numpy as np
import xarray as xr

def get_list_of_files(start = '2012-01-01', stop = '2012-01-31', include_start = True, include_stop = True, var = 'tcc', 
                                        path_input = '/home/hanna/lagrings/ERA5_monthly/'):
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
        files = glob.glob(os.path.join( path_input, '*{}*.nc'.format(var) ))
        files = np.sort(files) # sorting then for no particular reson
        
        if path_input == '/home/hanna/lagrings/ERA5_tcc/':
            min_fil = os.path.join(path_input, start_search_str + '_{}_era.nc'.format(var))
            max_fil = os.path.join(path_input, stop_search_str + '_{}_era.nc'.format(var))
        else:
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
    return xr.open_mfdataset(files, compat='no_conflicts')

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



for start, stop in [('2004-04-01', '2008-12-31'),
                    ('2009-01-01', '2013-12-31'),
                    ('2014-01-01', '2018-12-31')]:

    # Load Data -- 
    files_tcc =  get_list_of_files(start  = start, stop = stop, include_start = True, include_stop = True, var = 'tcc', 
		                                path_input = '/uio/lagringshotell/geofag/students/metos/hannasv/ERA5_monthly/')

    files_era5 =  get_list_of_files(start  = start, stop = stop, include_start = True, include_stop = True, var = 'tcc', 
		                                path_input = '/uio/lagringshotell/geofag/students/metos/hannasv/ERA5_tcc/')

    files_r =  get_list_of_files(start = start, stop = stop, include_start = True, include_stop = True, var = 'r', 
                                        path_input = '/uio/lagringshotell/geofag/students/metos/hannasv/ERA5_monthly/')

    true_data = merge(np.concatenate([files_tcc, files_r ]))
    era5_data = merge(files_era5)
    data = xr.merge([era5_data.rename({'tcc':'era'}), true_data])
    latitude = true_data.latitude.values
    longitude = true_data.longitude.values

    mse_storage = np.zeros((len(latitude),
                             len(longitude)))

    r2_storage = np.zeros((len(latitude),
                             len(longitude)))

    ase_storage = np.zeros((len(latitude),
                             len(longitude)))

    for i, lat in enumerate(latitude):
        for j, lon in enumerate(longitude):

            y_test_true = data['tcc'].sel(latitude = lat, longitude = lon).values
            y_test_pred = data['era'].sel(latitude = lat, longitude = lon).values

            if len(y_test_pred) != len(y_test_true):
                print('era {}'.format(len(y_test_pred)))
                print('true {}'.format(len(y_test_true)))
            #assert len(y_test_true) != len(y_test_pred)

            mse  = mean_squared_error(y_test_true, y_test_pred)
            print('mse shape {}'.format(np.shape(mse)))
            ase  = accumulated_squared_error(y_test_true, y_test_pred)
            r2   = r2_score(y_test_true, y_test_pred)

            mse_storage[i, j] = mse
            r2_storage[i, j] = r2
            ase_storage[i, j] = ase

            print('Finished with pixel {}/{}'.format((i+1)*j, 81*161))

    path_ar_results = '/uio/lagringshotell/geofag/students/metos/hannasv/results/era5/'
    filename      = '/uio/lagringshotell/geofag/students/metos/hannasv/results/era5/ERA5_{}_{}.nc'.format(start, stop)

    print('Stores file {}'.format(filename))
    vars_dict = {'mse': (['latitude', 'longitude'], mse_storage),
                 'r2':  (['latitude', 'longitude'], r2_storage),
                 'ase': (['latitude', 'longitude'], ase_storage),

                 'global_mse': np.mean(mse_storage),
                 'global_r2':  np.mean(r2_storage),
                 'global_ase': np.mean(ase_storage),
                  }
    ds = xr.Dataset(vars_dict,
                     coords={'longitude': (['longitude'], longitude),
                             'latitude': (['latitude'], latitude),
                            })
    ds.to_netcdf(filename)
