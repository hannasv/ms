import os
import glob

import numpy as np
import xarray as xr

from sclouds.helpers import (merge, get_list_of_variables_in_ds,
                             get_pixel_from_ds, path_read_data, path_ar_results)
from sclouds.io.utils import dataset_to_numpy
from sclouds.ml.regression.utils import (mean_squared_error, r2_score,
                                         fit_pixel, predict_pixel,
                                         accumulated_squared_error,
                                         sigmoid, inverse_sigmoid)

class AR_model:
    """ Autoregressive models used in this thesis.

    TODO update with start stop

    """

    def __init__(self, order = 1, transformer = False, sigmoid = False):
        # Based on start and stop descide which files it gets.
        files        = self.get_test_files()
        self.dataset = merge(files)
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
        return

    def get_test_files(self):
        return ['/home/hanna/lagrings/ERA5_monthly/2012_01_q.nc',
                '/home/hanna/lagrings/ERA5_monthly/2012_01_r.nc',
                '/home/hanna/lagrings/ERA5_monthly/2012_01_t2m.nc',
                '/home/hanna/lagrings/ERA5_monthly/2012_01_sp.nc',
                '/home/hanna/lagrings/ERA5_monthly/2012_01_tcc.nc']

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

    def fit(self):
        """ Fits the data retrieved in the constructor, entire grid.
        """
        coeff_matrix = np.zeros((len(self.latitude),
                                 len(self.longitude),
                                 len(self.variables)))

        for i, lat in enumerate(self.latitude): # 81
            for j, lon in enumerate(self.longitude): # 161
                # TODO update this to be a dataloader
                if self.transform:
                    X, y, mean, std = self.load_transform(lat, lon)
                    self.mean[i, j, :]  =  mean.flatten()
                    self.std[i, j, :] =  std.flatten()
                else:
                    X, y = self.load(lat, lon)
                #print('Number of samples after removal of nans {}.'.format(len(y)))
                coeffs = fit_pixel(X, y)
                coeff_matrix[i, j, :] =  coeffs.flatten()

        self.coeff_matrix = coeff_matrix
        return coeff_matrix

    def set_transformer_from_loaded_model(self, transformer, mean, std):
        """ Set old tranformation """
        self.transform = True
        self.mean = mean
        self.std = std
        return

    def set_weights_from_loaded_model(self, W):
        """ Sets weights from loaded model """
        self.coeff_matrix = W
        return

    def predict(self, X):
        """ Make prediction for the entire grid/ Domain.
        Keeps nan's.
        """
        n_time = len(self.dataset.time.values)
        n_lat  = len(self.latitude)
        n_lon  = len(self.longitude)
        Y      = np.zeros( (n_time, n_lat, n_lon, 1)  )

        for i in range(n_lat):
            for j in range(n_lon):
                a = X[:, i, j, :]

                if self.transform:
                    a = (a - self.mean[i, j, :])/self.std[i, j, :]

                _X = a[~np.isnan(a).any(axis=1)]
                _w = self.coeff_matrix[i, j, :, np.newaxis]

                y_pred = predict_pixel(_X, _w)

                if self.sigmoid:
                    y_pred = sigmoid(y_pred)

                Y[:, i, j, 0] = y_pred.flatten()
        return Y

    def evaluate(self, y_true, y_pred):
        # MOve most of content in store performance to evaluate
        mse  = mean_squared_error(y_true, y_pred)
        ase  = accumulated_squared_error(y_true, y_pred)
        r2   = r2_score(y_true, y_pred)
        vars_dict = {'mse': (['latitude', 'longitude'], mse[:, :, 0]),
                     'r2':  (['latitude', 'longitude'], r2[:, :, 0]),
                     'ase': (['latitude', 'longitude'], ase[:, :, 0]),
                      }

        ds = xr.Dataset(vars_dict,
                         coords={'longitude': (['longitude'], self.longitude),
                                 'latitude': (['latitude'], self.latitude),
                                })
        self.evaluate_ds = ds
        return ds


    def save_performance(self, y_true, y_pred):
        """ Evaluate the performace in each grid
        To make the code work simply use the same training and test data.
        """
        from utils import write_path
        end_location = os.path.join(write_path, 'test_performace.nc')

        if not self.evaluate_ds:
            self.evaluate(y_true, y_pred)

        self.evaluate_ds.to_netcdf(end_location)
        return

    def save_transform(self):
        """ Save model to repo : /home/hanna/lagrings/results/ar/
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

        ds = xr.Dataset(vars_dict,
                         coords={'longitude': (['longitude'], self.longitude),
                                 'latitude': (['latitude'], self.latitude),
                                })

        end_location = os.path.join(path_ar_results, 'test2_transformer.nc')
        ds.to_netcdf(end_location)
        return


    def save_model(self):
        """ Save model to repo : /home/hanna/lagrings/results/ar/

        TODO : Save model should include necessary information to restore
        transformation if model is trained have transformation

        TODO : This should also know if there is a bias.

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
        ds = xr.Dataset(temp_dict,
                         coords={'longitude': (['longitude'], self.longitude),
                                 'latitude': (['latitude'], self.latitude),
                                })
        end_location = os.path.join(path_ar_results, 'test2.nc')
        ds.to_netcdf(end_location)
        return
