""" Paralisering av regresjon
"""
import os
import glob

import numpy as np
import xarray as xr

from timeit import default_timer as timer

from sclouds.ml.regression.utils import (mean_squared_error, r2_score, fit_pixel,
                                         predict_pixel,
                                         accumulated_squared_error,
                                         sigmoid, inverse_sigmoid)

from sclouds.ml.regression.utils import (dataset_to_numpy, dataset_to_numpy_order,
                                         dataset_to_numpy_order_traditional_ar,
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
base = '/home/hanna/lagrings/results/stats/2014-01-01_2018-12-31/'

#path_ar_results = '/uio/lagringshotell/geofag/students/metos/hannasv/results/ar/'
#filename = '/uio/lagringshotell/geofag/students/metos/hannasv/results/ar/MODEL_{}.nc'.format(np.datetime64('now'))

def get_train_test(test_start, test_stop, model = 'ar'):
    """Loads train and test data to datasets ... """
    #logger.info('Retrives data')
    if type == 'ar':
        files_train = get_list_of_files_excluding_period(test_start, test_stop)
        files_test = get_list_of_files(test_start, test_stop)

    else:
        files_train = get_list_of_files_excluding_period_traditional_model(test_start, test_stop)
        files_test = get_list_of_files_traditional_model(test_start, test_stop)

    #logger.info('Detected the relevant files. ')
    train_dataset = merge(files_train)
    #logger.info('Merged training data for {} to {}'.format(test_start,
                                    #test_stop))
    test_dataset = merge(files_test)
    #logger.info('Merged test data for {} to {}'.format(test_start, test_stop))
    return train_dataset, test_dataset


class Transformer:
    """ Object used to transforma all pixels timer.
    """

    def __init__(self, start=None, stop=None, test_start = None, test_stop = None,
                train_dataset = None, test_dataset = None,
                 order = 1, transform = False, sigmoid = False,
                 latitude = None, longitude = None, type = 'ar',
                 bias = True):

                self.start = start
                self.stop = stop
                self.test_start = test_start
                self.test_stop = test_stop

                self.train_dataset = train_dataset
                self.test_dataset = test_dataset

                self.transform = transform
                self.sigmoid   = sigmoid
                self.bias = bias

                self.order = order
                self.type  = type

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

                if self.transform:
                    self.mean, self.std = self.load_transformation()

    def load_transformation(self):
        """ Load transformation ..
        Returns xr.Datasets.
        """
        if len(self.variables) == 0:
            mean = xr.open_dataset(base + 'stats_pixel_{}_all.nc'.format('tcc'))['mean']
            std  = xr.open_dataset(base + 'stats_pixel_{}_all.nc'.format('tcc'))['std']
            print(mean)
            print(std)
        else:
            raise ValueError('Not implemented yet')
            var = self.variables[0]
            mean = xr.open_dataset(base + 'stats_pixel_{}_all.nc'.format(var))['mean']
            std  = xr.open_dataset(base + 'stats_pixel_{}_all.nc'.format(var))['std']
            for var in self.variables[1:]:
                mean  = xr.merge([mean, xr.open_dataset(base + 'stats_pixel_{}_all.nc'.format(var))['mean'] ])
                std = xr.merge([mean, xr.open_dataset(base + 'stats_pixel_{}_all.nc'.format(var))['std'] ])
        return mean, std

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

    def get_X_and_y(self, dataset):
        """Returns X and y based on a dataset. """
        if self.type == 'ar':
            if self.order > 0:
                #print('Dataset has order {}'.format(order))
                X, y = dataset_to_numpy_order(dataset,
                                            self.order, bias = self.bias)
            else:
                #print('Dataset has order {} -- should be zero.'.format(order))
                X, y  = dataset_to_numpy(dataset, bias = self.bias)
        else:
            X, y = dataset_to_numpy_order_traditional_ar(
                                    dataset, order = self.order, bias = self.bias)
        return X, y

    def transform_X(self, X, lat, lon):
        X_train = np.zeros(X.shape)
        if len(self.variables) > 0:
            for j, var in enumerate(self.variables):
                m = self.mean[var].sel(latitude = lat, longitude = lon).values
                s = self.std[var].sel(latitude = lat, longitude = lon).values
                X_train[:, j] = (X[:, j]- m)/s
        else:
            j = 0
        if order > 0:
            var = 'tcc'
            for k in range(order):
                m = self.mean.sel(latitude = lat, longitude = lon).values
                s = self.std.sel(latitude = lat, longitude = lon).values
                # Something wierd with the rotation of cloud cover values
                X_train[:, k+j+1] = (X[:, j+k+1]- m)/s
        return X_train

    def get_transformed_pixel(self, lat, lon):
        """ Update to take X and y as input and then ..
        """
        X_train, y_train = self.get_X_and_y(self.train_dataset.sel(latitude = lat, longitude = lon))
        # Initialize containers
        if self.transform:
            Xtrain = self.transform_X(X_train, lat, lon)

        if self.sigmoid:
            y_train = sigmoid(y)

        if self.test_dataset is not None:
            X_test, y_test = self.get_X_and_y(self.test_dataset.sel(latitude = lat, longitude = lon))
            if self.transform:
                X_test = self.transform_X(X_test, lat, lon)
            return X_train, y_train, X_test, y_test
        else:
            return X_train, y_train


class ModelPixel:
    """ En tråd må startes med all dataen den skal ha.
    """
    def __init__(self, config:dict, lat: float, lon: float):
        self.config = config
        self.config['lat'] = lat
        self.config['lon'] = lon
        self.timer_start = timer()

        # for storing of file
        self.score = dict()
        self.coeff = None

        self.longitude = lon
        self.latitude  = lat

    def get_weights(self, coeffs):
        """Returns dictionary of weigths fitted using this model.
        """
        temp_dict = {}

        if len(self.variables) > 0 :
            vars_dict = {'Wt2m':coeffs[1],
                         'Wr': coeffs[2],
                         'Wq': coeffs[0],
                         'Wsp': coeffs[3],
                          }
            ar_index =4
        else:
            ar_index = 0
        vars_dict = {}

        if self.bias:
            temp_dict['b'] = coeffs[ar_index]
            ar_index += 1

        if self.order > 0:
            for i in range(self.order):
                var = 'W{}'.format(i+1)
                temp_dict[var] = (['latitude', 'longitude'], self.coeff_matrix[:, :, ar_index])
                ar_index+=1

        temp_dict.update(vars_dict) # meges dicts together
        return temp_dict

    def fit_evaluate(self, X_train, y_train, X_test, y_test):
        # Do the regression -- change this if it to slow.
        coeffs = fit_pixel(X_train, y_train)
        print('Finished fitting pixel test data in load_transform_fit after {} seconds'.format(timer() - self.timer_start))

        y_test_pred = predict_pixel(X_test, coeffs)
        y_train_pred = predict_pixel(X_train, coeffs)

        if self.config['sigmoid']:
            y_test_pred  = inverse_sigmoid(y_test_pred)
            y_train_pred = inverse_sigmoid(y_train_pred)

        self.score['mse_test'] = mean_squared_error(y_test, y_test_pred)[0]
        self.score['ase_test'] = accumulated_squared_error(y_test, y_test_pred)[0]
        self.score['r2_test'] = r2_score(y_test, y_test_pred)[0]

        self.score['mse_train'] = mean_squared_error(y_train, y_train_pred)[0]
        self.score['ase_train'] = accumulated_squared_error(y_train, y_train_pred)[0]
        self.score['r2_train']  = r2_score(y_train, y_train_pred)[0]

        print('Finished computing mse, ase, r2 data in load_transform_fit after {} seconds'.format(timer() - self.timer_start))
        return coeffs,

    def save(self, X_train, y_train, X_test, y_test):
        """ Saves model configuration, evaluation, transformation into a file
        named by the current time. Repo : /home/hanna/lagrings/results/ar/
        """
        coeffs = self.fit_evaluate(X_train, y_train, X_test, y_test)

        filename        = '/uio/lagringshotell/geofag/students/metos/hannasv/results/ar/TEST_MODEL_{}_{}.nc'.format(
                                self.longitude, self.latitude)
        path_ar_results = '/uio/lagringshotell/geofag/students/metos/hannasv/results/ar/'

        path_ar_results = '/home/hanna/lagrings/results/ar/'
        filename        = '{}/TEST_MODEL_{}_{}.nc'.format(path_ar_results,
                            self.longitude, self.latitude)

        #os.path.join(path_ar_results, )
        print('Stores file {}'.format(filename))
        config_dict   = self.get_configuration()
        weights_dict  = self.get_weights(coeffs)

        # Merges dictionaries together
        config_dict.update(weights_dict)
        config_dict.update(self.score)

        ds = xr.Dataset(config_dict,
                         coords={'longitude': (['longitude'], self.longitude),
                                 'latitude': (['latitude'], self.latitude),
                                })
        ds.to_netcdf(filename)
        return

from multiprocessing import Process

def process_for_thread(lat, lon, X_train, y_train, X_test, y_test):

    model = ModelPixel(transformer.get_configuration().copy(), lat, lon)
    model.save(X_train, y_train, X_test, y_test)
    return

if __name__ == '__main__':

    start = None
    stop  = None

    test_start = '2014-01-01'
    test_stop  = '2018-12-31'
    #test_start = None
    #test_stop  = None

    sig = False
    trans = False
    bias = True
    order = 1
    timer_start = timer()

    test_start = '2014-01-01'
    test_stop  = '2018-12-31'

    type = 'traditional'

    lat = 30.0
    lon = 0.0

    test_start = '2012-01-01'
    test_stop  = '2012-01-31'

    test_files = glob.glob('/home/hanna/lagrings/ERA5_monthly/*2012*01*.nc')
    data = xr.open_mfdataset(test_files, compat='no_conflicts')

    train_dataset = data.copy()
    test_dataset  = data.copy()

    transformer = Transformer(start=test_start, stop=test_start,
                              test_start = None, test_stop = None,
                              train_dataset = train_dataset,
                              test_dataset = test_dataset,
                              order = order, transform = trans, sigmoid = sig,
                              latitude = lat, longitude = lon, type = 'traditional',
                              bias = bias)

    longitudes = [0.0, 0.25, 0.50]
    latitudes  = [30.0, 30.25, 30.50]

    process  = []

    for lat in latitudes:
        for lon in longitudes:
            X_train, y_train, X_test, y_test = transformer.get_transformed_pixel(lat, lon)
            print('Retrived data for {}, {}'.format(lon, lat))
            #model = ModelPixel(transformer.get_configuration().copy(), lat, lon)
            #model.save(X_train, y_train, X_test, y_test)
            p = Process(target = process_for_thread,
                        args = (lat, lon, X_train, y_train, X_test, y_test))
            p.start()
            print('Started thread  {}, {}'.format(lon, lat))

    for ps in process:
        ps.join()
