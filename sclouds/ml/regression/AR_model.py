import os
import glob

import numpy as np
import xarray as xr

from sclouds.ml.regression.utils import (mean_squared_error, r2_score, merge,
                                         get_list_of_variables_in_ds, fit_pixel, predict_pixel,
                                         accumulated_squared_error)

class AR_model:
    """ Autoregressive models used in this thesis. Inspired by sklearn
    """

    def __init__(self):
        # Based on start and stop descide which files it gets.
        files = self.get_test_files()
        self.dataset = merge(files)

        self.longitude = self.dataset.longitude.values
        self.latitude = self.dataset.latitude.values
        self.variables = get_list_of_variables_in_ds(self.dataset)

        self.coeff_matrix = None
        self.evaluate_ds = None
        return

    def get_test_files(self):
        return ['/home/hanna/lagrings/ERA5_monthly/2012_01_q.nc',
                '/home/hanna/lagrings/ERA5_monthly/2012_01_r.nc',
                '/home/hanna/lagrings/ERA5_monthly/2012_01_t2m.nc',
                '/home/hanna/lagrings/ERA5_monthly/2012_01_sp.nc',
                '/home/hanna/lagrings/ERA5_monthly/2012_01_tcc.nc']


    def fit(self):
        """ Fits the data retrieved in the constructor, entire grid.
        """
        coeff_matrix = np.zeros( (len(self.latitude),
                                 len(self.longitude),
                                 len(self.variables) ))

        for i, lat in enumerate(self.latitude): # 81
            for j, lon in enumerate(self.longitude): # 161

                # Move some of this to the dataloader part?
                ds     = get_pixel_from_ds(self.dataset, lat, lon)
                X, y   = dataset_to_numpy(ds, bias = True)
                print('Number of samples prior to removal of nans {}.'.format(len(y)))

                # Removes nan's
                Xy = np.concatenate([X, y], axis = 1)
                Xy = Xy[~np.isnan(Xy).any(axis = 1)]

                print('Number of samples after removal of nans {}.'.format(len(y)))
                coeffs = fit_pixel(X, y)
                coeff_matrix[i, j, :] =  coeffs.flatten()

        self.coeff_matrix = coeff_matrix
        return coeff_matrix

    def set_weights_from_loaded_model(self, W):
        self.coeff_matrix = W
        return

    def predict(self, X):
        """ Make prediction for the entire grid/ Domain."""
        n_time = len(self.dataset.time.values)
        n_lat  = len(self.latitude)
        n_lon  = len(self.longitude)
        Y      = np.zeros( (n_time, n_lat, n_lon, 1)  )

        for i in range(n_lat):
            for j in range(n_lon):
                a = X[:, i, j, :]
                _X = a[~np.isnan(a).any(axis=1)]
                # TODO make sure this doesn't contain nan's
                _w = self.coeff_matrix[i, j, :, np.newaxis]

                y_pred = predict_pixel(_X, _w)
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

    def store_performance(self, y_true, y_pred):
        """ Evaluate the performace in each grid
        To make the code work simply use the same training and test data.
        """
        from utils import write_path
        end_location = os.path.join(write_path, 'test_performace.nc')

        if not self.evaluate_ds:
            self.evaluate(y_true, y_pred)

        self.evaluate_ds.to_netcdf(end_location)
        return

    def save_model(self):
        """ Save model to repo : /home/hanna/lagrings/results/ar/
        """
        temp_dict = {}

        for i in range(int(len(self.variables) - 5)):
            var = 'W{}'.format(i+1)
            temp_dict[var] = (['latitude', 'longitude'],
                                np.zeros((len(self.latitude),
                                          len(self.longtude))))

        vars_dict = {'b': (['latitude', 'longitude'], self.weights[:, :, 4]),
                     'Wt2m':(['latitude', 'longitude'], self.weights[:, :, 1]),
                     'Wr': (['latitude', 'longitude'], self.weights[:, :, 2]),
                     'Wq':(['latitude', 'longitude'], self.weights[:, :, 0]),
                     'Wsp': (['latitude', 'longitude'], self.weights[:, :, 3]),
                      }
        temp_dict.update(vars_dict) # meges dicts together

        ds = xr.Dataset(temp_dict,
                         coords={'longitude': (['longitude'], self.longitude),
                                 'latitude': (['latitude'], self.latitude),
                                })
        ds.to_netcdf(end_location)
        return
