import os
import glob

import numpy as np
import xarray as xr

from timeit import timeit

from utils import (dataset_to_numpy, dataset_to_numpy_order,
                                         dataset_to_numpy_order_traditional_ar)

from utils import (mean_squared_error, mean_absolute_error,
                                         r2_score, fit_pixel,
                                         predict_pixel,
                                         accumulated_squared_error,
                                         sigmoid, inverse_sigmoid)

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

def transform_X(X, lat, lon, data, order):
    X_train = np.zeros(X.shape)
    variables = ['q', 't2m', 'r', 'sp']

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

path_transform = '/home/hanna/lagrings/results/stats/2014-01-01_2018-12-31/'
path = '/home/hanna/lagrings/ar_data/'


# path_transform = '/home/hanna/lagrings/results/stats/2014-01-01_2018-12-31/'
# path = '/home/hanna/lagrings/ar_data/'

lagr_path = '/uio/lagringshotell/geofag/students/metos/hannasv/'
path_transform = '{}results/stats/2014-01-01_2018-12-31'.format(lagr_path)

path = '{}ar_data/'.format(lagr_path)
path_ar_results = '{}/results/ar/'.format(lagr_path)

latitude  = 30.0
longitude = 5.25
SPATIAL_RESOLUTION = 0.25

latitudes =  np.arange(30.0, 50.0 +SPATIAL_RESOLUTION, step = SPATIAL_RESOLUTION)
longitudes = np.arange(-15, 25+SPATIAL_RESOLUTION, step = SPATIAL_RESOLUTION)
base = '{}/results/stats/2014-01-01_2018-12-31/'.format(lagr_path)

transform = False
if transform:

    ds_tcc = xr.open_dataset(base + 'stats_pixel_tcc_all.nc')
    ds_r = xr.open_dataset(base +'stats_pixel_r_all.nc')
    ds_q = xr.open_dataset(base+'stats_pixel_q_all.nc')
    ds_t2m = xr.open_dataset(base+'stats_pixel_t2m_all.nc')
    ds_sp = xr.open_dataset(base+'stats_pixel_sp_all.nc')

    stats_data = {'q':ds_q, 't2m': ds_t2m, 'r': ds_r, 'sp': ds_sp, 'tcc': ds_tcc}

sig = True
bias = False
order =5
model_type = 'AR'

explaination = ['q', 't2m', 'r', 'sp']
tr_index = 3

if bias:
    explaination.append('bias')
    tr_index += 1

full_name = generate_model_name(model_type, bias, transform, sig, order)
config = get_config_from_model_name(full_name)

full_name_tr = generate_model_name('TR', bias, transform, sig, order)
tr_config = get_config_from_model_name(full_name_tr)

for latitude in latitudes:#[30.0, 30.25]: # , 30.5, 30.75, 31.0
    for longitude in longitudes:#[0.0, 0.25, 0.5, 0.75, 1.0]:
    #for latitude in latitudes:

       explain = explaination.copy()
       tr_explain = []

       for o in range(0, order +1):
            name    = full_name+'-o{}'.format(o)
            tr_name = full_name_tr+'-o{}'.format(o)

            w_filename        = '{}weights_{}_{}_{}.nc'.format(path_ar_results, name, longitude, latitude)
            p_filename        = '{}performance_{}_{}_{}.nc'.format(path_ar_results, name, longitude, latitude)

            fil = 'all_vars_lat_lon_{}_{}.nc'.format(latitude, longitude)
            data = xr.open_dataset(path+fil)
            #if not os.path.exists(w_filename) and os.path.exists(p_filename):
            if o > 0:
               explain.append('O{}'.format(o))
               tr_explain.append('O{}'.format(o))
            start_time = timeit()
            X_train, y_train = dataset_to_numpy_order(dataset = data.sel(time = slice('2004', '2013')),
                                      order = order,  bias = bias)

            X_test, y_test   = dataset_to_numpy_order(dataset = data.sel(time = slice('2014', '2018')),
                                      order = order,  bias = bias)
            if transform:
                X_train = transform_X(X_train, lat = latitude, lon=longitude, data=stats_data, order=o)
                X_test = transform_X(X_test, lat = latitude, lon=longitude, data=stats_data, order=o)

            if sig:
                y_train = inverse_sigmoid(y_train)
                y_test  = inverse_sigmoid(y_test)

            name    = full_name+'-o{}'.format(o)
            tr_name = full_name_tr+'-o{}'.format(o)

            eval_dict    = {}
            eval_tr_dict = {}
            weights_dict = {}
            weights_tr_dict = {}

            Xtr, ytr =  drop_nans(X_train[:, :int(4+o)], y_train)
            Xte, yte =  drop_nans(X_test[:, :int(4+o)], y_test)

            if sig:
                yte = sigmoid(yte)
                ytr = sigmoid(ytr)

            if np.isnan(yte).any():
                print('Warning nans detected in training data')

            if np.isnan(ytr).any():
                print('Warning nans detected in test data')

            if o > 0:
                # updatig predictors
                Tr_Xtr =  Xtr[:, 4:]
                Tr_Xte =  Xte[:, 4:]

                coeffs_tr   = fit_pixel(Tr_Xtr, ytr)

                y_test_pred_tr  = predict_pixel(Tr_Xte, coeffs_tr)
                y_train_pred_tr = predict_pixel(Tr_Xtr, coeffs_tr)

                mse_test_tr  = mean_squared_error(y_test_pred_tr, yte)
                mse_train_tr = mean_squared_error(y_train_pred_tr, ytr)

                mae_test_tr  = mean_absolute_error(y_test_pred_tr, yte)
                mae_train_tr = mean_absolute_error(y_train_pred_tr, ytr)

            ############# Fitting
            coeffs      = fit_pixel(Xtr, ytr)

            y_test_pred  = predict_pixel(Xte, coeffs)
            y_train_pred = predict_pixel(Xtr, coeffs)

            ################ Evaluation
            mse_test  = mean_squared_error(y_test_pred, yte)
            mse_train = mean_squared_error(y_train_pred, ytr)

            mae_test  = mean_squared_error(y_test_pred, yte)
            mae_train = mean_squared_error(y_train_pred, ytr)

            ##################### Adding the autoregressive model
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

            ###################### Adding traditional model
            if o > 0:
                weights_tr_dict['coeffs'] = (['weights'], coeffs_tr.flatten())  # 'latitude', 'longitude',

                eval_tr_dict['mse_test']  = mse_test_tr[0]   #(['latitude', 'longitude'],)
                eval_tr_dict['mse_train'] = mse_train_tr[0]

                eval_tr_dict['mae_test']  = mae_test_tr[0]   #(['latitude', 'longitude'], )
                eval_tr_dict['mae_train'] = mae_train_tr[0]  #(['latitude', 'longitude'], )

                num_test_samples  = len(yte)
                num_train_samples = len(ytr)

                eval_tr_dict['num_test_samples']  = num_test_samples  # (['latitude', 'longitude'], )
                eval_tr_dict['num_train_samples'] = num_train_samples # (['latitude', 'longitude'], )

                eval_tr_dict.update(tr_config)
                weights_tr_dict.update(tr_config)

            stop_time = timeit()
            #print(stop_time - start_time)
            eval_dict['time_elapsed_seconds'] = (stop_time - start_time) #(['latitude', 'longitude'], )

            w_filename        = '{}weights_{}_{}_{}.nc'.format(path_ar_results, name, longitude, latitude)
            p_filename        = '{}performance_{}_{}_{}.nc'.format(path_ar_results, name, longitude, latitude)
            ds = xr.Dataset(weights_dict, coords={'latitude':  (['latitude'],  [latitude]),
                                              'longitude': (['longitude'], [longitude]),
                                              'weights':   (['weights'],   explain )
                                              })
            ds.to_netcdf(w_filename)

            ds = xr.Dataset(eval_dict, coords={'latitude': (['latitude'],   [latitude]),
                                           'longitude': (['longitude'], [longitude])
                                           })
            ds.to_netcdf(p_filename)
            ################# Storing traditional model
            if o > 0:
                w_tr_filename        = '{}/weights_{}_{}_{}.nc'.format(path_ar_results, tr_name, longitude, latitude)
                p_tr_filename        = '{}/performance_{}_{}_{}.nc'.format(path_ar_results, tr_name, longitude, latitude)

                ds = xr.Dataset(weights_tr_dict, coords={'latitude':  (['latitude'],  [latitude]),
                                                  'longitude': (['longitude'], [longitude]),
                                                  'weights':   (['weights'],   tr_explain )
                                                  })
                ds.to_netcdf(w_tr_filename)

                ds = xr.Dataset(eval_tr_dict, coords={'latitude': (['latitude'],   [latitude]),
                                               'longitude': (['longitude'], [longitude])
                                               })
                ds.to_netcdf(p_tr_filename)
            print('finished false, false, false({}, {})'.format(longitude, latitude))
            #else:
            #    print('file {} and {} exist'.format(p_filename, w_filename) )
