""" Explenation of the content of this file.
"""

import os
import glob

import numpy as np
import xarray as xr

from timeit import default_timer as timer

from sclouds.ml.regression.utils import (mean_squared_error, r2_score, fit_pixel, predict_pixel,
                     accumulated_squared_error,
                     sigmoid, inverse_sigmoid)


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
    print('enters dataset_to_numpy_order_traditional_ar')
    if bias:
        var_index = 1
    else:
        var_index = 0

    times = dataset.time.values
    #print("Detected {} samples.".format(len(times)))
    X = np.zeros( (len(times)-order, order + var_index))
    y = np.zeros( (len(times)-order ))
    print('generated empty X and y')
    tcc = dataset.tcc.values
    #print('len tcc {}'.format(len(tcc)))
    print(tcc)
    if bias:
        X[:, 0] = 1 # bias
    print('before y')
    y = tcc[:len(times)-order, np.newaxis]
    #print('len y should be tcc - order {}'.format(len(y)))
    print('finished y')
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
    print('finished X')
    #print(X.shape)
    #print(y.shape)
    return X, y



from utils import (dataset_to_numpy, dataset_to_numpy_order,
                    #dataset_to_numpy_order_traditional_ar,

#from sclouds.ml.regression.utils import (dataset_to_numpy, dataset_to_numpy_order,
#                    dataset_to_numpy_order_traditional_ar,

                              dataset_to_numpy_grid_order,
                              dataset_to_numpy_grid,
                              get_xarray_dataset_for_period,
                              get_list_of_files_excluding_period,
                              get_list_of_files,
                              get_list_of_files_excluding_period_traditional_model,
                              get_list_of_files_traditional_model)

#sys.path.insert(0,'/uio/hume/student-u89/hannasv/MS/sclouds/')
from sclouds.helpers import (merge, get_list_of_variables_in_ds,
                             get_pixel_from_ds, path_input, path_ar_results)

base = '/uio/lagringshotell/geofag/students/metos/hannasv/results/stats/2014-01-01_2018-12-31/' #'2014-01-01_2018-12-31/

class Model:

    def __init__(self, start = None, stop = None,
                    test_start = None, test_stop = None,
                    train_dataset = None, test_dataset = None,
                    order = 1, transform = False, sigmoid = False,
                    latitude = None, longitude = None, type = 'ar',
                    bias = True):
        """
        Parameters
        ----------
        start : str
            Start date of training/ fitting the model. Format: year-month-date
        stop  : str
            Stop date of training/ fitting the model. Format: year-month-date
        test_start : str, optional
            Start date of test data, used in evaluation of the model.
        test_stop  : str, optional
            Stop date of test data, used in evaluation of the model.
        order : int
            Number of previos time steps included as predictors
        transformer : bool, default = True
            Whether to standardize the data or not.
        sigmoid : bool, default = False
            Desides if you should siogmoid transform the response.
        """
        if stop is not None:
            assert start < stop, "Start {} need to be prior to stop {}".format(start, stop)

        self.start = start
        self.stop  = stop

        self.test_start = test_start
        self.test_stop  = test_stop
        self.timer_start = timer()

        self.type = type

        if train_dataset is not None:
            print('Sets training data .... ')
            self.dataset = train_dataset
        else:
            if((start is None and stop is None) and
                    (test_start is not None and test_stop is not None) ):

                if type == 'ar':
                    files = get_list_of_files_excluding_period(test_start, test_stop)
                else:
                    files = get_list_of_files_excluding_period_traditional_model(test_start, test_stop)

                #print('Detected {} files .. Merging might take a while ... '.format(len(files)))
                self.dataset = merge(files)

            elif((start is None and stop is None) and
                    (test_start is None and test_stop is None) ):
                    raise ValueError('Something is wrong with')
            else:
                # Based on start and stop descide which files it gets.
                if type == 'ar':
                    files = get_list_of_files(test_start, test_stop)
                else:
                    files = get_list_of_files_traditional_model(test_start, test_stop)
                #print('Detected {} files .. Merging might take a while ... '.format(len(files)))
                self.dataset = merge(files)

        if test_dataset is not None:
            print('Sets test data')
            self.test_dataset = test_dataset
        else:
            if self.test_start is not None and self.test_stop is not None:
                # Load test data
                print('Loads test data -- this should happen')
                if self.type == 'ar':
                    files = get_list_of_files(start = self.test_start, stop = self.test_stop,
                                include_start = True, include_stop = True)
                else:
                    files = get_list_of_files_traditional_model(start = self.test_start, stop = self.test_stop,
                                include_start = True, include_stop = True)
                self.test_dataset = merge(files)


        print('Finished loaded the dataset')
        self.order = order

        if latitude is None:
            self.latitude  = self.dataset.latitude.values
            print('sets latitude values to be {}'.format(self.latitude))
        else:
            self.latitude = np.array(latitude)
            print('sets latitude values to be {}'.format(self.latitude))

        if longitude is None:
            self.longitude = self.dataset.longitude.values
            print('sets longitude values to be {}'.format(self.longitude))
        else:
            self.longitude = np.array(longitude)
            print('sets longitude values to be {}'.format(self.longitude))

        if type == 'ar':
            self.variables = ['t2m', 'q', 'r', 'sp'] #get_list_of_variables_in_ds(self.dataset)
        else:
            # traditional model
            self.variables = []
        print('Sets enviornmental variables to {}'.format(self.variables))

        self.coeff_matrix = None
        self.evaluate_ds  = None

        self.mse = None
        self.r2 = None
        self.ase = None

        self.mse_train = None
        self.r2_train = None
        self.ase_train = None

        self.num_test_samples = None
        self.num_train_samples = None

        self.transform   = transform
        self.sigmoid     = sigmoid

        # Initialize containers if data should be transformed
        if self.transform:
            """ Read transformation from the correct folder in lagringshotellet """
            self.bias = False
            print("Transform is true, forces bias to be false .. ")
        else:
            self.bias = bias

        self.X_train = None
        self.y_train = None
        return

    def transform_data(self, X, y, lat, lon):
        """ Standardisation X by equation x_new = (x-mean(x))/std(x)

        Parameteres
        ---------------------
        lat : float
            Latitude coordinate.
        lon : float
            Longitude coordinate.

        Returns
        ---------------------
        mean, std : float
            Values used in transformation
        """
        """ Normalizes the distribution. It is centered around the mean with std of 1.

        Subtract the mean divide by the standard deviation. """

        # Removes nan's
        a = np.concatenate([X, y], axis = 1)
        a = a[~np.isnan(a).any(axis = 1)]

        X = a[:, :-1]

        #if self.sigmoid:
        #    y = inverse_sigmoid(a[:-1, np.newaxis]) # not tested
        #else:
        y_removed_nans = a[:, -1, np.newaxis]

        order = self.order
        #n_times, n_lat, n_lon, n_vars = X.shape[0]
        VARIABLES = ['t2m', 'q', 'r', 'sp']
        transformed = np.zeros(X.shape)

        for j, var in enumerate(self.variables):

            m = xr.open_dataset(base + 'stats_pixel_{}_all.nc'.format(var))['mean'].sel(latitude = lat, longitude = lon).values
            s = xr.open_dataset(base + 'stats_pixel_{}_all.nc'.format(var))['std'].sel(latitude = lat, longitude = lon).values

            transformed[:, j] = (X[:, j]- m)/s
            #for i in range(n_times):
            #    transformed[i, :, :, j] =  (X[i, :, :, j]  - m)/s
        if order > 0:
            var = 'tcc'
            for k in range(order):
                m = xr.open_dataset(base + 'stats_pixel_{}_all.nc'.format(var))['mean'].sel(latitude = lat, longitude = lon).values
                s = xr.open_dataset(base + 'stats_pixel_{}_all.nc'.format(var))['std'].sel(latitude = lat, longitude = lon).values
                # Something wierd with the rotation of cloud cover values
                transformed[:, k+j+1] = (X[:, j+k+1]- m)/s

        return transformed, y_removed_nans

    def fit(self):
        """ New fit function
        """
        print('enters fitting ')
        num_vars = self.bias + len(self.variables) + self.order

        coeff_matrix = np.zeros((len(self.latitude),
                                 len(self.longitude),
                                 num_vars))

        mse_storage = np.zeros((len(self.latitude),
                                 len(self.longitude)))

        r2_storage = np.zeros((len(self.latitude),
                                 len(self.longitude)))

        ase_storage = np.zeros((len(self.latitude),
                                 len(self.longitude)))

        mse_storage_train = np.zeros((len(self.latitude),
                                 len(self.longitude)))

        r2_storage_train = np.zeros((len(self.latitude),
                                 len(self.longitude)))

        ase_storage_train = np.zeros((len(self.latitude),
                                 len(self.longitude)))

        num_train_samples = np.zeros((len(self.latitude),
                                 len(self.longitude)))

        num_test_samples = np.zeros((len(self.latitude),
                                 len(self.longitude)))

        for i, lat in enumerate(self.latitude): # 81
            for j, lon in enumerate(self.longitude): # 161
                # Loads
                print('Starts with pixel {}/{}'.format((i+1)*j, 81*len(self.longitude)))
                coeff, mse, ase, r2, num_test, num_train, mse_tr, ase_tr, r2_tr = self.load_transform_fit(lat, lon)

                coeff_matrix[i, j, :] = coeff
                mse_storage[i, j] = mse
                r2_storage[i, j] = r2
                ase_storage[i, j] = ase

                mse_storage_train[i, j] = mse_tr
                r2_storage_train[i, j]  = r2_tr
                ase_storage_train[i, j] = ase_tr

                num_train_samples[i,j] = num_test
                num_test_samples[i,j] = num_train

                print('Finished with pixel {}/{}'.format((i+1)*j, 81*len(self.longitude)))

        self.coeff_matrix = coeff_matrix
        self.mse = mse_storage
        self.ase = ase_storage
        self.r2 = r2_storage

        self.mse_train = mse_storage_train
        self.ase_train = ase_storage_train
        self.r2_train = r2_storage_train

        self.num_test_samples = num_test_samples
        self.num_train_samples = num_train_samples
        return self

    def predict(self, lat, lon):
        """ Used by model loader.
        """
        # TODO loop over dataset ...
        ds     = get_pixel_from_ds(self.test_dataset, lat, lon)
        if self.type == 'ar':
            if self.order > 0:
                #print('Dataset has order {}'.format(order))
                X_test, y_test_true = dataset_to_numpy_order(ds, self.order, bias = self.bias)
            else:
                #print('Dataset has order {} -- should be zero.'.format(order))
                X_test, y_test_true  = dataset_to_numpy(ds, bias = self.bias)
        else:
            X_test, y_test_true   = dataset_to_numpy_order_traditional_ar(ds,
                                        order = self.order, bias = self.bias)
        #VARIABLES = ['t2m', 'q', 'r', 'sp']
        if self.transform:
            transformed_test = np.zeros((n_times, n_vars ))

            for j, var in enumerate(self.variables):
                t_data = xr.open_dataset(base + 'stats_pixel_{}_all.nc'.format(var))
                m = t_data['mean'].sel(latitude = lat, longitude = lon).values
                s = t_data['std'].sel(latitude = lat, longitude = lon).values

                transformed_test[:, j] = (X_test[:, j]- m)/s

            if order > 0:
                j = len(self.variables)
                var = 'tcc'
                t_data = xr.open_dataset(base + 'stats_pixel_{}_all.nc'.format(var))
                m = t_data['mean'].sel(latitude = lat, longitude = lon).values
                s = t_data['std'].sel(latitude = lat, longitude = lon).values

                for k in range(order):
                    # Something wierd with the rotation of cloud cover values
                    transformed_test[:, k+j] = (X_test[:, k+j]- m)/s
            X_test = transformed_test
            print('Finished transforming test data in load_transform_fit after {} seconds'.format(timer() - self.timer_start))

        i = (lat-30.0)/0.25
        j = (lon-(-15.0))/0.25

        coeffs = self.coeff_matrix[int(i), int(j), :][:, np.newaxis]
        y_test_pred = predict_pixel(X_test, coeffs)
        print('Finished predicting test pixel data in load_transform_fit after {} seconds'.format(timer() - self.timer_start))

        if self.sigmoid:
            y_test_pred = inverse_sigmoid(y_test_pred)

        if len(y_test_true) == 4:
            y_test_true = y_test_true[:, :, :, 0]


        if len(y_test_pred) == 4:
            y_test_pred = y_test_pred[:, :, :, 0]

        # Move most of content in store performance to evaluate
        mse  = mean_squared_error(y_test_true, y_test_pred)[0]
        print('mse shape {}'.format(np.shape(mse)))
        ase  = accumulated_squared_error(y_test_true, y_test_pred)[0]
        r2   = r2_score(y_test_true, y_test_pred)[0]
        #print(mse, ase, r2)
        print('Finished computing mse, ase, r2 data in load_transform_fit after {} seconds'.format(timer() - self.timer_start))
        print('mse {}, ase {}, r2 {}'.format(mse, ase, r2))
        return mse, ase, r2

    def load_transform_fit(self, lat, lon):
        """ Standardisation X by equation x_new = (x-mean(x))/std(x)

        Parameteres
        ---------------------
        lat : float
            Latitude coordinate.
        lon : float
            Longitude coordinate.

        Returns
        ---------------------
        mean, std : float
            Values used in transformation
        """
        """ Normalizes the distribution. It is centered around the mean with std of 1.

        Subtract the mean divide by the standard deviation. """
        # Move some of this to the dataloader part?
        local = timer()
        print('Enters load_transform_fit after {} seconds'.format(local - self.timer_start))
        ds     = get_pixel_from_ds(self.dataset, lat, lon)
        if self.type == 'ar':
            if self.order > 0:
                X, y   = dataset_to_numpy_order(ds, order = self.order, bias = self.bias)
            else:
                X, y   = dataset_to_numpy(ds, bias = self.bias)
        else:
            print('finds traditional model')
            X, y   = dataset_to_numpy_order_traditional_ar(ds,
                                        order = self.order, bias = self.bias)
        local = timer()
        print('Finished reading in pixel in load_transform_fit after {} seconds'.format(local - self.timer_start))
        # Removes nan's
        a = np.concatenate([X, y], axis = 1)
        a = a[~np.isnan(a).any(axis = 1)]

        X = a[:, :-1]
        #print(X.shape)
        if self.sigmoid:
            y = inverse_sigmoid(a[:, -1, np.newaxis]) # not tested
        else:
            y = a[:, -1, np.newaxis]
        #print(y.shape)

        order = self.order
        n_times, n_vars = X.shape
        #VARIABLES = ['t2m', 'q', 'r', 'sp']
        if self.transform:
            transformed_train = np.zeros(X.shape)
            for j, var in enumerate(self.variables):

                m = xr.open_dataset(base + 'stats_pixel_{}_all.nc'.format(var))['mean'].sel(latitude = lat, longitude = lon).values
                s = xr.open_dataset(base + 'stats_pixel_{}_all.nc'.format(var))['std'].sel(latitude = lat, longitude = lon).values

                transformed_train[:, j] = (X[:, j]- m)/s
                #for i in range(n_times):
                #    transformed[i, :, :, j] =  (X[i, :, :, j]  - m)/s
            if order > 0:
                j = len(self.variables)

                var = 'tcc'
                for k in range(order):
                    m = xr.open_dataset(base + 'stats_pixel_{}_all.nc'.format(var))['mean'].sel(latitude = lat, longitude = lon).values
                    s = xr.open_dataset(base + 'stats_pixel_{}_all.nc'.format(var))['std'].sel(latitude = lat, longitude = lon).values
                    # Something wierd with the rotation of cloud cover values
                    transformed_train[:, k+j] = (X[:, k+j]- m)/s

            X_train = transformed_train
        else:
            X_train = X
        local = timer()
        print('Finished transforming pixel in load_transform_fit after {} seconds'.format(timer() - self.timer_start))
        #if self.test_dataset is not None:
        #if self.test_start is not None and self.test_stop is not None:
            # Based on start and stop descide which files it gets.

        ds     = get_pixel_from_ds(self.test_dataset, lat, lon)
        #print(ds)

        if self.type == 'ar':
            if self.order > 0:
                #print('Dataset has order {}'.format(order))
                X_test, y_test_true = dataset_to_numpy_order(ds, self.order, bias = self.bias)
            else:
                #print('Dataset has order {} -- should be zero.'.format(order))
                X_test, y_test_true  = dataset_to_numpy(ds, bias = self.bias)
        else:
            X_test, y_test_true   = dataset_to_numpy_order_traditional_ar(ds,
                                        order = self.order, bias = self.bias)
        n_times, n_vars = X_test.shape
        print('Finished reading in test data pixel in load_transform_fit after {} seconds'.format(timer() - self.timer_start))

        #VARIABLES = ['t2m', 'q', 'r', 'sp']
        if self.transform:
            transformed_test = np.zeros((n_times, n_vars ))

            for j, var in enumerate(self.variables):
                t_data = xr.open_dataset(base + 'stats_pixel_{}_all.nc'.format(var))
                m = t_data['mean'].sel(latitude = lat, longitude = lon).values
                s = t_data['std'].sel(latitude = lat, longitude = lon).values

                transformed_test[:, j] = (X_test[:, j]- m)/s

            if order > 0:
                j = len(self.variables)
                var = 'tcc'
                t_data = xr.open_dataset(base + 'stats_pixel_{}_all.nc'.format(var))
                m = t_data['mean'].sel(latitude = lat, longitude = lon).values
                s = t_data['std'].sel(latitude = lat, longitude = lon).values

                for k in range(order):
                    # Something wierd with the rotation of cloud cover values
                    transformed_test[:, k+j] = (X_test[:, k+j]- m)/s
            X_test = transformed_test
            print('Finished transforming test data in load_transform_fit after {} seconds'.format(timer() - self.timer_start))

        num_test = (~np.isnan(X_test)).sum(axis=0)[0]
        num_train = (~np.isnan(X_train)).sum(axis=0)[0]
        #print('Xtrain shape {} y train.shape {}'.format(X_train.shape, y.shape))
        coeffs = fit_pixel(X_train, y)
        #print('coeff {}'.format(coeffs))
        print('Finished fitting pixel test data in load_transform_fit after {} seconds'.format(timer() - self.timer_start))
        y_test_pred = predict_pixel(X_test, coeffs)
        y_train_pred = predict_pixel(X_train, coeffs)
        print('Finished predicting test pixel data in load_transform_fit after {} seconds'.format(timer() - self.timer_start))

        if self.sigmoid:
            y_test_pred = inverse_sigmoid(y_test_pred)

        if len(y_test_true) == 4:
            y_test_true = y_test_true[:, :, :, 0]


        if len(y_test_pred) == 4:
            y_test_pred = y_test_pred[:, :, :, 0]

        # Move most of content in store performance to evaluate
        mse  = mean_squared_error(y_test_true, y_test_pred)[0]
        print('mse shape {}'.format(np.shape(mse)))
        ase  = accumulated_squared_error(y_test_true, y_test_pred)[0]
        r2   = r2_score(y_test_true, y_test_pred)[0]
        mse_tr = mean_squared_error(y, y_train_pred)[0]
        ase_tr = accumulated_squared_error(y, y_train_pred)[0]
        r2_tr  = r2_score(y, y_train_pred)[0]

        #print(mse, ase, r2)
        print('Finished computing mse, ase, r2 data in load_transform_fit after {} seconds'.format(timer() - self.timer_start))
        print('mse {}, ase {}, r2 {}'.format(mse, ase, r2))
        return coeffs.flatten(), mse, ase, r2, num_test, num_train, mse_tr, ase_tr, r2_tr

    def fit_evaluate(self):
        raise NotImplementedError('Coming soon ... ')


    def set_configuration(self, config_dict):
        """ Set loaded configuration.

        Parameters
        --------------------
        config_dict : dictionary
            Dictionary containing the config.
        """
        self.transform  = config_dict['transform']
        self.sigmoid    = config_dict['sigmoid']
        self.order      = config_dict['order']
        self.start      = config_dict['start']
        self.stop       = config_dict['stop']
        self.test_start = config_dict['test_start']
        self.test_stop  = config_dict['test_stop']
        self.bias       = config_dict['bias']
        return

    def set_weights_from_loaded_model(self, W):
        """ Sets weights from loaded model.

        Parameters
        --------------
        W : array-like
            Matrix containing weights from loaded model.
        """
        self.coeff_matrix = W
        return

    def get_evaluation(self):
        """Evaluation"""
        vars_dict = {'mse': (['latitude', 'longitude'], self.mse),
                     'r2':  (['latitude', 'longitude'], self.r2),
                     'ase': (['latitude', 'longitude'], self.ase),

                     'num_train_samples': (['latitude', 'longitude'],
                                    self.num_train_samples),
                     'num_test_samples': (['latitude', 'longitude'],
                                    self.num_test_samples),

                     'mse_train': (['latitude', 'longitude'], self.mse_train),
                     'r2_train':  (['latitude', 'longitude'], self.r2_train),
                     'ase_train': (['latitude', 'longitude'], self.ase_train),

                     'global_mse': np.mean(self.mse),
                     'global_r2':  np.mean(self.r2),
                     'global_ase': np.mean(self.ase),
                      }

        return vars_dict

    def get_configuration(self):
        """Returns dictionary of configuration used to initialize this model.
        """
        temp_dict = {'transform'  : self.transform,
                     'sigmoid'    : self.sigmoid,
                     'order'      : self.order,
                     'start'      : self.start,
                     'stop'       : self.stop,
                     'test_start' : self.test_start,
                     'test_stop'  : self.test_stop,
                     'bias'       : self.bias,
                     'type'       :self.type}
        return temp_dict

    def get_weights(self):
        """Returns dictionary of weigths fitted using this model.
        """
        temp_dict = {}

        if len(self.variables) > 0 :

            vars_dict = {'Wt2m':(['latitude', 'longitude'], self.coeff_matrix[:, :, 1]),
                         'Wr': (['latitude', 'longitude'], self.coeff_matrix[:, :, 2]),
                         'Wq':(['latitude', 'longitude'], self.coeff_matrix[:, :, 0]),
                         'Wsp': (['latitude', 'longitude'], self.coeff_matrix[:, :, 3]),
                          }
            ar_index =4
        else:
            ar_index = 0
            vars_dict = {}

        if self.bias:
            temp_dict['b'] = (['latitude', 'longitude'], self.coeff_matrix[:, :, ar_index])
            ar_index += 1


        if self.order > 0:
            for i in range(self.order):
                var = 'W{}'.format(i+1)
                temp_dict[var] = (['latitude', 'longitude'], self.coeff_matrix[:, :, ar_index])
                ar_index+=1

        temp_dict.update(vars_dict) # meges dicts together
        return temp_dict

    def save(self):
        """ Saves model configuration, evaluation, transformation into a file
        named by the current time. Repo : /home/hanna/lagrings/results/ar/
        """
        path_ar_results = '/uio/lagringshotell/geofag/students/metos/hannasv/results/ar/'
        filename      = '/uio/lagringshotell/geofag/students/metos/hannasv/results/ar/MODEL_{}.nc'.format(np.datetime64('now'))
        #os.path.join(path_ar_results, )
        print('Stores file {}'.format(filename))
        config_dict   = self.get_configuration()
        weights_dict  = self.get_weights()

        #if self.transform:
        #    tranformation = self.get_tranformation_properties()
        #    config_dict.update(tranformation)

        eval_dict     = self.get_evaluation()

        # Merges dictionaries together
        config_dict.update(weights_dict)
        config_dict.update(eval_dict)

        ds = xr.Dataset(config_dict,
                         coords={'longitude': (['longitude'], self.longitude),
                                 'latitude': (['latitude'], self.latitude),
                                })
        ds.to_netcdf(filename)
        return

def get_train_test(test_start, test_stop, model = 'ar'):
    print('gets train test data for ar model')
    if type == 'ar':
        files_train = get_list_of_files_excluding_period(test_start, test_stop)
        files_test = get_list_of_files(test_start, test_stop)

    else:
        files_train = get_list_of_files_excluding_period_traditional_model(test_start, test_stop)
        files_test = get_list_of_files_traditional_model(test_start, test_stop)

    print('Detected {} train files and {} test files. Merging might take a while .... '.format(len(files_train), len(files_test)))
    train_dataset = merge(files_train)
    print('finished merging train')
    test_dataset = merge(files_test)
    return train_dataset, test_dataset

if __name__ == '__main__':
    start = None
    stop  = None

    test_start = '2014-01-01'
    test_stop  = '2018-12-31'
    #test_start = None
    #test_stop  = None

    sig = False
    trans = True
    bias = True

    timer_start = timer()

    test_start = '2014-01-01'
    test_stop  = '2018-12-31'

    type = 'traditional'
    train_dataset, test_dataset = get_train_test(test_start, test_stop,
                                                  model = type)

    sig = False
    trans = True
    bias = True

    m = Model(start = start, stop = stop,
                 test_start = test_start, test_stop = test_stop,
                 train_dataset = train_dataset, test_dataset = test_dataset,
                 order = 1,                 transform = trans,
                 sigmoid = sig, latitude = None, longitude = None,
                 type = type)
    coeff = m.fit()
    m.save()
    print(m.get_configuration())
    """
    m = Model(start = start, stop = stop,
                 test_start = test_start, test_stop = test_stop,
                 train_dataset = train_dataset, test_dataset = test_dataset,
                 order = 1,                 transform = trans,
                 sigmoid = sig, latitude = None, longitude = None,
                 type = type)

    coeff = m.fit()
    m.save()
    print(m.get_configuration())
    print('Finished transforming pixel in load_transform_fit after {} seconds'.format(timer() - timer_start))

    print('Finish model 6 ... ')
    m = Model(start = start, stop = stop,
             test_start = test_start, test_stop = test_stop,
             train_dataset = train_dataset, test_dataset = test_dataset,
             order = 1,                 transform = trans,
             sigmoid = sig, latitude = [30], longitude = [0],
             type = type)
    coeff = m.fit()
    m.save()
    print(m.get_configuration())
    print('Finish model 7 ...')



    type = 'ar'
    train_dataset, test_dataset = get_train_test(test_start, test_stop, model = type)
    print('Loaded train')
    print(train_dataset)
    print('Loaded test')
    print(test_dataset)
    print('Finished merging')
    #start = '2012-01-01'
    #stop  = '2012-03-01'
    start = None
    stop  = None

    test_start = '2014-01-01'
    test_stop  = '2018-12-31'
    #test_start = None
    #test_stop  = None

    sig = False
    trans = True
    bias = True

    m = Model(start = start, stop = stop,
                 test_start = test_start, test_stop = test_stop,
                 train_dataset = train_dataset, test_dataset = test_dataset,
                 order = 1,                 transform = trans,
                 sigmoid = sig, latitude = None, longitude = None,
                 type = type)
    coeff = m.fit()
    m.save()
    print(m.get_configuration())
    print('pass model 1')
    m = Model(start = start, stop = stop,
                 test_start = test_start, test_stop = test_stop,
                  train_dataset = train_dataset, test_dataset = test_dataset,
                 order = 1,                 transform = trans,
                 sigmoid = sig, latitude = [30], longitude = [0],
                 type = type)
    coeff = m.fit()
    m.save()
    print(m.get_configuration())
    print('pass model 2')

    sig = False
    trans = True

    m = Model(start = start, stop = stop,
              test_start = test_start, test_stop = test_stop,
               train_dataset = train_dataset, test_dataset = test_dataset,

              order = 2,                 transform = trans,
              sigmoid = sig, latitude = [30], longitude = [0], type = type)
    coeff = m.fit()
    m.save()
    print(m.get_configuration())

    print('pass model 3')
    sig = True
    trans = True

    m = Model(start = start, stop = stop,
                 test_start = test_start, test_stop = test_stop,
              train_dataset = train_dataset, test_dataset = test_dataset,
                 order = 2,                 transform = trans,
                 sigmoid = sig, latitude = [30, 30.25], longitude = [0], type = type)
    coeff = m.fit()
    m.save()
    print(m.get_configuration())

    print('pass model 4')
    sig = True
    trans = False

    m = Model(start = start, stop = stop,
                 test_start = test_start, test_stop = test_stop,
                  train_dataset = train_dataset, test_dataset = test_dataset,

                 order = 2,                 transform = trans,
                 sigmoid = sig, latitude = [30, 30.25], longitude = [0], type = type)
    coeff = m.fit()
    m.save()
    print(m.get_configuration())

    sig = False
    trans = False
    print('pass model 5')
    m = Model(start = start, stop = stop,
                 test_start = test_start, test_stop = test_stop,
                 train_dataset = train_dataset, test_dataset = test_dataset,
                 order = 2,                 transform = trans,
                 sigmoid = sig, latitude = [30], longitude = [0], type = type)
    coeff = m.fit()
    m.save()
    print(m.get_configuration())
    print('about to load data for traditional model')
    """
