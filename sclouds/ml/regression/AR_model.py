""" Explenation of the content of this file.
"""

import os
import glob

import numpy as np
import xarray as xr

from sclouds.helpers import (merge, get_list_of_variables_in_ds,
                             get_pixel_from_ds, path_input, path_ar_results)

from sclouds.io.utils import (dataset_to_numpy, dataset_to_numpy_grid_order,
                              dataset_to_numpy_grid,
                              get_xarray_dataset_for_period)

from sclouds.ml.regression.utils import (mean_squared_error, r2_score,
                                         fit_pixel, predict_pixel,
                                         accumulated_squared_error,
                                         sigmoid, inverse_sigmoid)

class AR_model:
    """ Autoregressive models used in this thesis.

    Attributes
    -------------------
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
    longitude : array-like
        Longitude values.
    latitude  : array-like
        Latitude values.
    coeff_matrix : array-like
        Matrix

    Methods
    -------------------
    load(lat, lon)

    load_transform(lat, lon)

    fit()

    fit_evaluate()

    set_transformer_from_loaded_model(transformer, mean, std)

    set_weights_from_loaded_model(W)

    predict(X)

    get_evaluation()

    get_configuration()

    get_weights()

    get_tranformation_properties()

    save()
    """

    def __init__(self, start = '2012-01-01', stop = '2012-01-31',
                    test_start = None, test_stop = None,
                    order = 1, transform = False, sigmoid = False):
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

        # Based on start and stop descide which files it gets.
        self.dataset = get_xarray_dataset_for_period(start = self.start,
                                                     stop = self.stop)
        self.order = order

        self.longitude = self.dataset.longitude.values
        self.latitude  = self.dataset.latitude.values
        self.variables = ['t2m', 'q', 'r', 'sp'] #get_list_of_variables_in_ds(self.dataset)

        self.coeff_matrix = None
        self.evaluate_ds  = None

        self.transform   = transformer
        self.sigmoid     = sigmoid

        # Initialize containers if data should be transformed
        if self.transform:
            self.mean = np.zeros((len(self.latitude),
                                  len(self.longitude),
                                  len(self.variables)))

            self.std  = np.zeros((len(self.latitude),
                                  len(self.longitude),
                                  len(self.variables)))
            self.bias = False
        else:
            self.bias = True

        self.X_train = None
        self.y_train = None
        return

    def load(self, lat, lon):

        # Move some of this to the dataloader part?
        ds     = get_pixel_from_ds(self.dataset, lat, lon)

        # TODO : This should make a proper choice of loader function.

        X, y   = dataset_to_numpy(ds, bias = self.bias)
        # print('Number of samples prior to removal of nans {}.'.format(len(y)))
        # Removes nan's
        a = np.concatenate([X, y], axis = 1)
        B = a[~np.isnan(a).any(axis = 1)]
        X = B[:, :-1]
        y = B[:, -1, np.newaxis] # not tested
        return X, y

    def load_transform(self, lat, lon):
        """ Standardisation X by equation x_new = (x-mean(x))/std(x)

        Parameteres
        ---------------------
        X : array-like

        Returns
        ---------------------
        mean, std : float
            Values used in transformation
        """
        from sklearn.preprocessing import StandardScaler
        # Move some of this to the dataloader part?
        ds     = get_pixel_from_ds(self.dataset, lat, lon)
        X, y   = dataset_to_numpy(ds, bias = self.bias)
        #print('Number of samples prior to removal of nans {}.'.format(len(y)))
        # Removes nan's
        a = np.concatenate([X, y], axis = 1)
        a = a[~np.isnan(a).any(axis = 1)]

        X = a[:, :-1]

        if self.sigmoid:
            y = inverse_sigmoid(a[:-1, np.newaxis]) # not tested
        else:
            y = a[:, -1, np.newaxis]

        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        return X, y, scaler.mean_, np.sqrt(scaler.var_)

    def fit_evaluate(self, ):
        raise NotImplementedError('Coming soon ... ')

    def fit(self):
        """ Fits the data retrieved in the constructor, entire grid.
        """
        coeff_matrix = np.zeros((len(self.latitude),
                                 len(self.longitude),
                                 len(self.variables)))

        _X = np.zeros((len(self.dataset.time.values),
                      len(self.latitude),
                      len(self.longitude),
                      len(self.variables)))

        _y = np.zeros((len(self.dataset.time.values),
                      len(self.latitude),
                      len(self.longitude), 1))

        for i, lat in enumerate(self.latitude): # 81
            for j, lon in enumerate(self.longitude): # 161
                # TODO update this to be a dataloader
                if self.transform:
                    X, y, mean, std = self.load_transform(lat, lon)
                    self.mean[i, j, :]  =  mean.flatten()
                    self.std[i, j, :] =  std.flatten()
                else:
                    X, y = self.load(lat, lon)
                _X[:, i, j, :] = X
                _y[:, i, j, :] = y
                #print('Number of samples after removal of nans {}.'.format(len(y)))
                coeffs = fit_pixel(X, y)
                coeff_matrix[i, j, :] =  coeffs.flatten()

        self.X_train = _X
        self.y_train = _y
        self.coeff_matrix = coeff_matrix
        return coeff_matrix

    def set_transformer_from_loaded_model(self, mean, std):
        """ Set old tranformation """
        self.transform = True
        self.mean = mean
        self.std = std
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

    def predict(self, X):
        """ Make prediction for the entire grid/ Domain.
        Keeps nan's.

        Parameters
        --------------
        X : array-like
            Matrix containing input data.
        """
        n_time = len(self.dataset.time.values)
        n_lat  = len(self.latitude)
        n_lon  = len(self.longitude)
        Y      = np.zeros( (n_time, n_lat, n_lon, 1)  )
        print("X shape {}".format(X.shape))
        for i in range(n_lat):
            for j in range(n_lon):
                a = X[:, i, j, :]

                # Checks if data should be transformed and performs
                # transformation.
                if self.transform:
                    a = (a - self.mean[i, j, :])/self.std[i, j, :]

                _X = a[~np.isnan(a).any(axis=1)]
                _w = self.coeff_matrix[i, j, :, np.newaxis]

                y_pred = predict_pixel(_X, _w)

                if self.sigmoid:
                    y_pred = sigmoid(y_pred)

                Y[:, i, j, 0] = y_pred.flatten()
        return Y

    def get_evaluation(self):
        """ Get evaluation of data
        """
        # Checks if test_start and test_stop is provided.
        if self.test_start is not None and self.test_stop is not None:
            # Based on start and stop descide which files it gets.
            dataset = get_xarray_dataset_for_period(start = self.test_start,
                                                    stop = self.test_stop)
            if self.order > 0:
                X, y_true = dataset_to_numpy_grid_order(dataset, self.order, bias = self.bias)
            else:
                X, y_true = dataset_to_numpy_grid(dataset, bias = self.bias)
            y_pred = self.predict(X)
        else:
            #raise NotImplementedError('Coming soon ... get_evaluation()')
            print("X shape {}, y shape {}".format(self.X_train.shape, self.y_train.shape))
            y_pred = self.predict(self.X_train)
            y_true = self.y_train
        # Move most of content in store performance to evaluate
        mse  = mean_squared_error(y_true, y_pred)
        ase  = accumulated_squared_error(y_true, y_pred)
        r2   = r2_score(y_true, y_pred)

        vars_dict = {'mse': (['latitude', 'longitude'], mse[:, :, 0]),
                     'r2':  (['latitude', 'longitude'], r2[:, :, 0]),
                     'ase': (['latitude', 'longitude'], ase[:, :, 0]),
                     'global_mse': np.mean(mse),
                     'global_r2':  np.mean(r2),
                     'global_ase': np.mean(ase)
                      }
        return vars_dict

    def get_configuration(self):
        """Returns dictionary of configuration used to initialize this model.
        """
        temp_dict = {'transform' : self.transform,
                     'sigmoid' : self.sigmoid,
                     'order' : self.order,
                     'start' : self.start,
                     'stop'  : self.stop,
                     'test_start' : self.test_start,
                     'test_stop'  : self.test_stop,
                     'bias':self.bias}
        return temp_dict

    def get_weights(self):
        """Returns dictionary of weigths fitted using this model.
        """
        temp_dict = {}

        if self.bias:
            temp_dict['b'] = (['latitude', 'longitude'], self.coeff_matrix[:, :, 4])
            ar_index = 5
        else:
            ar_index = 4

        if self.order > 0:
            for i in range(self.order):
                var = 'W{}'.format(i+1)
                temp_dict[var] = (['latitude', 'longitude'], self.coeff_matrix[:, :, ar_index])
                ar_index+=1

        vars_dict = {'Wt2m':(['latitude', 'longitude'], self.coeff_matrix[:, :, 1]),
                     'Wr': (['latitude', 'longitude'], self.coeff_matrix[:, :, 2]),
                     'Wq':(['latitude', 'longitude'], self.coeff_matrix[:, :, 0]),
                     'Wsp': (['latitude', 'longitude'], self.coeff_matrix[:, :, 3]),
                      }
        temp_dict.update(vars_dict) # meges dicts together
        return temp_dict

    def get_tranformation_properties(self):
        """ Returns dictionary of the properties used in the transformations,
        pixelwise mean and std.
        """
        vars_dict = {'mean_t2m':(['latitude', 'longitude'], self.mean[:, :, 1]),
                     'std_t2m':(['latitude', 'longitude'], self.std[:, :, 1]),
                     'mean_r':(['latitude', 'longitude'], self.mean[:, :, 2]),
                     'std_r':(['latitude', 'longitude'], self.std[:, :, 2]),
                     'mean_q':(['latitude', 'longitude'], self.mean[:, :, 0]),
                     'std_q':(['latitude', 'longitude'], self.std[:, :, 0]),
                     'mean_sp':(['latitude', 'longitude'], self.mean[:, :, 3]),
                     'std_sp':(['latitude', 'longitude'], self.std[:, :, 3]),
                      }
        return vars_dict

    def save(self):
        """ Saves model configuration, evaluation, transformation into a file
        named by the current time. Repo : /home/hanna/lagrings/results/ar/
        """
        filename      = os.path.join(path_ar_results, 'AR_{}.nc'.format(np.datetime64('now')))

        config_dict   = self.get_configuration()
        weights_dict  = self.get_weights()
        tranformation = self.get_tranformation_properties()
        eval_dict     = self.get_evaluation()

        # Merges dictionaries together
        config_dict.update(weights_dict)
        config_dict.update(tranformation)
        config_dict.update(eval_dict)

        ds = xr.Dataset(config_dict,
                         coords={'longitude': (['longitude'], self.longitude),
                                 'latitude': (['latitude'], self.latitude),
                                })
        ds.to_netcdf(filename)
        return
