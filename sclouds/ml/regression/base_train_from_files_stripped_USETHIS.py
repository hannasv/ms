import os
import glob

#os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import numpy as np
import xarray as xr

path = '/global/D1/homes/hannasv/ar_data3/'
path_ar_results = '/home/hannasv/results_ar_stripped/'

#data = xr.open_dataset('/global/D1/homes/hannasv/ar_data/all_vars_lat_lon_35.0_-15.0.nc')
#data = xr.open_dataset(list_files[0])
#print(data)

filnavn = '/global/D1/homes/hannasv/ar_data/all_vars_lat_lon_35.0_-15.0.nc'
#data = xr.open_dataset(filnavn)
#print(data)

from timeit import timeit

from utils import (dataset_to_numpy, dataset_to_numpy_order,
                                         dataset_to_numpy_order_traditional_ar)

from utils import (mean_squared_error, mean_absolute_error,
                                         r2_score, fit_pixel,
                                         predict_pixel,
                                         accumulated_squared_error,
                                         sigmoid, inverse_sigmoid)

ar_data_path = path #'/global/D1/homes/hannasv/ar_data/'
results_path = path_ar_results #'/home/hannasv/results_ar/'


#filnavn = '/global/D1/homes/hannasv/ar_data/all_vars_lat_lon_35.0_-15.0.nc'
#data = xr.open_dataset(filnavn)
#print(data)

def drop_nans(X, y):
    a = np.concatenate([X, y], axis = 1)
    B = a[~np.isnan(a).any(axis = 1)]
    X = B[:, :-1]
    y = B[:, -1, np.newaxis] # not tested
    return X, y

def generate_model_name(model_type, bias, transform, sigmoid, order):
    name = model_type
    if bias:
        name += '-B'
    if transform:
        name += '-T'
    if sigmoid:
        name += '-S'
    name += '-{}'.format(order)
    return name

def get_config_from_model_name(name):
    """returns dict with config"""
    splits = name.split('-')

    bias = False
    transform = False
    sigmoid = False

    if 'B' in splits:
        bias = True
    if 'T' in splits:
        transform = True
    if 'S' in splits:
        sigmoid = True

    order      = int(splits[-1])
    model_type = splits[0]
    temp_dict  = {'transform'  : transform,
                'sigmoid'    : sigmoid,
                'order'      : order,
                'start'      : '2004',
                'stop'       : '2013',
                'test_start' : '2014',
                'test_stop'  : '2018',
                'bias'       :  bias,
                'type'       :  model_type}

    return temp_dict

def get_list_of_files_to_regridd(model_name, path):
    print('searching in {} for model {}'.format(path, model_name))
    finished_files = glob.glob(os.path.join(path, '*performance*{}*L{}*'.format(model_name, 1)))
    print('Found {} files in {}.'.format(len(finished_files), path))
    lh_files = glob.glob('/global/D1/homes/hannasv/ar_data3/*.nc')
    #print(lh_files)
    # Get search string from trained models.
    crop_files_finished = []
    for fil in finished_files:
        splits = fil.split('_')
        lat = splits[-2]
        lon = splits[-1][:-3]
        crop_files_finished.append('{}_{}'.format(lon, lat))

    list_to_regridd = []
    # Crop_files = []
    for fil in lh_files:
        splits = fil.split('_')
        lat = splits[-2]
        lon = splits[-1][:-3]
        search_key = '{}_{}'.format(lat, lon)

        if not search_key in crop_files_finished:
            list_to_regridd.append(fil)
    return list_to_regridd

def transform_X(X, lat, lon, data, order):
    X_train = np.zeros(X.shape)
    variables = ['q', 't2m', 'r', 'sp']

    lat = float(lat)
    lon = float(lon)

    if len(variables) > 0:
        for j, var in enumerate(variables):
            m = data[var].sel(latitude = lat, longitude = lon)['mean'].values
            s = data[var].sel(latitude = lat, longitude = lon)['std'].values
            X_train[:, j] = (X[:, j]- m)/s
    else:
        j = 0
    if order > 0:
        var = 'tcc'
        for k in range(order):
            m = data[var].sel(latitude = lat, longitude = lon)['mean'].values
            s = data[var].sel(latitude = lat, longitude = lon)['std'].values
            # Something wierd with the rotation of cloud cover values
            X_train[:, k+j+1] = (X[:, j+k+1]- m)/s
    return X_train


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
    return np.log(x/(1 - x + 0.1))


def train_ar_model(transform=False, bias=False, sig=False, order=0, overwrite_results = True):
    # path_transform = '/home/hanna/lagrings/results/stats/2014-01-01_2018-12-31/'
    # path = '/home/hanna/lagrings/ar_data/'
    # print(bias)
    # print(transform)
    if transform and bias:
        print('Not valid model....')
        raise OSError('Not valid model config')

    # path_transform = '/home/hanna/lagrings/results/stats/2014-01-01_2018-12-31/'
    # path = '/home/hanna/lagrings/ar_data/'

    #lagr_path = '/uio/lagringshotell/geofag/students/metos/hannasv/'
    #path_transform = '{}results/stats/2014-01-01_2018-12-31'.format(lagr_path)

    #path = '/global/D1/homes/hannasv/ar_data/'
    #path_ar_results = '/home/hannasv/results_ar/'
    #list_files = ['/global/D1/homes/hannasv/ar_data/all_vars_lat_lon_35.0_-15.0.nc', '/global/D1/homes/hannasv/ar_data/all_vars_lat_lon_35.0_-14.75.nc']
    list_files = glob.glob('/global/D1/homes/hannasv/ar_data3/*.nc')

    import xarray as xr
    path = '/global/D1/homes/hannasv/ar_data3/'
    path_ar_results = '/home/hannasv/results_ar_stripped/'

    #latitude  = 30.0
    #longitude = 5.25
    SPATIAL_RESOLUTION = 0.25

    latitudes =  np.arange(30.0, 50.0 +SPATIAL_RESOLUTION, step = SPATIAL_RESOLUTION)
    longitudes = np.arange(-15, 25+SPATIAL_RESOLUTION, step = SPATIAL_RESOLUTION)
    base = '/home/hannasv/results/stats/2014-01-01_2018-12-31/'

    explaination = ['q', 't2m', 'r', 'sp', 'bias','L1']
    tr_e = []
    tr_index = 4

    full_name = generate_model_name('AR', bias, transform, sig, order)
    config = get_config_from_model_name(full_name)

    full_name_tr = generate_model_name('TR', bias, transform, sig, order)
    tr_config = get_config_from_model_name(full_name_tr)
    path_ar_results = os.path.join(path_ar_results, full_name)

    if not os.path.isdir(path_ar_results):
        os.mkdir(path_ar_results)
        print('Created directory {}'.format(path_ar_results))
    print(full_name)
    list_to_regridd = get_list_of_files_to_regridd(model_name=full_name, path=path_ar_results)
    #list_to_regridd = glob.glob('/global/D1/homes/hannasv/ar_data/*.nc')
    list_files = list_to_regridd
    print('Detected {}/13041 num files to regridd.'.format(len( list_to_regridd )))
    corrupt_files = []
    for i, fil in enumerate(list_files): #list_to_regridd:
        #print(xr.open_dataset(fil))
        #print(fil==filnavn)
        print('Working on fil {}/{}'.format(i, len(list_files)))
        print(fil)
        splits = fil.split('_')
        latitude = splits[-2]
        longitude = splits[-1][:-3]

        #try:
        data = xr.open_dataset(fil, engine = 'netcdf4')
        #print(data)
        #print(data.sp)
        #except Exception as e:
        #    print('FATAL ERROR !!! Unable to load {} got error {}'.format(fil, e))

        #print(data)
        explain = explaination.copy()
        print(explain)
        tr_explain = tr_e.copy()

        #for o in range(0, order +1):
        o=1
        name    = full_name+'-L{}'.format(o)
        tr_name = full_name_tr+'-L{}'.format(o)

        w_filename        = '{}weights_{}_{}_{}.nc'.format(path_ar_results, name, longitude, latitude)
        p_filename        = '{}performance_{}_{}_{}.nc'.format(path_ar_results, name, longitude, latitude)
        #data = xr.open_dataset(path+fil)
        start_time = timeit()
        try:
            X_train, y_train = dataset_to_numpy_order(dataset = data.sel(time = slice('2004', '2013')),
                                                      order = order,  bias = bias)
            X_test, y_test   = dataset_to_numpy_order(dataset = data.sel(time = slice('2014', '2018')),
                                                      order = order,  bias = bias)
        except RuntimeError:
            X_train = []
            y_train = []
            X_test = []
            y_test=[]
            print('Failed to read file')
        #print('transform {}'.format(transform))
        #print(bias)
        if transform: # and not bias):# or (not transform and bias):
            X_train = transform_X(X_train, lat = latitude, lon=longitude, data=stats_data, order=o)
            X_test = transform_X(X_test, lat = latitude, lon=longitude, data=stats_data, order=o)

        if sig:
            y_train = inverse_sigmoid(y_train)
            y_test  = inverse_sigmoid(y_test)

        name    = full_name+'-L{}'.format(o)
        tr_name = full_name_tr+'-L{}'.format(o)

        eval_dict    = {}
        eval_tr_dict = {}
        weights_dict = {}
        weights_tr_dict = {}

        Xtr, ytr =  drop_nans(X_train[:, :int(tr_index+o+bias)], y_train)
        Xte, yte =  drop_nans(X_test[:, :int(tr_index+o+bias)], y_test)
        #print(Xtr.shape)
        if sig:
            yte = sigmoid(yte)
            ytr = sigmoid(ytr)

        if np.isnan(yte).any():
            print('Warning nans detected in training data')

        if np.isnan(ytr).any():
            print('Warning nans detected in test data')

            ############# Fitting
        try:
            coeffs      = fit_pixel(Xtr, ytr)

            y_test_pred  = predict_pixel(Xte, coeffs)
            y_train_pred = predict_pixel(Xtr, coeffs)

            ################ Evaluation
            mse_test  = mean_squared_error(y_test_pred, yte)
            mse_train = mean_squared_error(y_train_pred, ytr)

            mae_test  = mean_squared_error(y_test_pred, yte)
            mae_train = mean_squared_error(y_train_pred, ytr)

        except np.linalg.LinAlgError:
            coeffs = np.ones(len(explain))
            mse_test  = [np.nan]
            mse_train = [np.nan] #mean_squared_error(y_train_pred, ytr)

            mae_test  = [np.nan] #mean_squared_error(y_test_pred, yte)
            mae_train = [np.nan] #mean_squared_error(y_train_pred, ytr)

        ##################### Adding the autoregressive model
        #print(coeffs)
        #print(explaination)
        weights_dict['coeffs'] = (['weights'], coeffs.flatten())  # 'latitude', 'longitude',

        eval_dict['mse_test']  = mse_test[0]   #(['latitude', 'longitude'],)
        eval_dict['mse_train'] = mse_train[0]

        eval_dict['mae_test']  = mae_test[0]   #(['latitude', 'longitude'], )
        eval_dict['mae_train'] = mae_train[0]  #(['latitude', 'longitude'], )

        num_test_samples  = len(yte)
        num_train_samples = len(ytr)

        eval_dict['num_test_samples']  = num_test_samples  # (['latitude', 'longitude'], )
        eval_dict['num_train_samples'] = num_train_samples # (['latitude', 'longitude'], )

        eval_dict.update(config)
        weights_dict.update(config)

        stop_time = timeit()
        #print(stop_time - start_time)
        eval_dict['time_elapsed_seconds'] = (stop_time - start_time) #(['latitude', 'longitude'], )

        w_filename        = '{}/weights_{}_{}_{}.nc'.format(path_ar_results, name, longitude, latitude)
        p_filename        = '{}/performance_{}_{}_{}.nc'.format(path_ar_results, name, longitude, latitude)
        print(w_filename)
        ds = xr.Dataset(weights_dict, coords={'latitude':  (['latitude'],  [latitude]),
                                    'longitude': (['longitude'], [longitude]),
                                    'weights':   (['weights'],   explain )
                                    })
        ds.to_netcdf(w_filename, engine = 'h5netcdf')

        ds = xr.Dataset(eval_dict, coords={'latitude': (['latitude'],   [latitude]),
                                 'longitude': (['longitude'], [longitude])
                                 })
        ds.to_netcdf(p_filename, engine = 'h5netcdf')
        print('finished calibrating bias {}, sigmoid {}, Transform {}, order/Lag {} - ({}, {})'.format(bias, sig, transform, o, longitude, latitude))
        #else:
            #print('Model config already calibrated bias {}, sigmoid {}, Transform {}, order/Lag {} - ({}, {})'.format(bias, sig, transform, o, longitude, latitude))
        #except Exception as e:
        #     print('It happens when trying to acess the data')
        #     corrupt_files.append(fil)

    print('Corrupt files')
    print(corrupt_files)


if __name__ == "__main__":
    train_ar_model(transform=False, bias=True, sig=False, order=5, overwrite_results = False)
    train_ar_model(transform=False, bias=False, sig=False, order=0, overwrite_results = False)

