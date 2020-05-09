""" Explenation of the content of this file.
"""

import os
import glob

import numpy as np
import xarray as xr

from utils import (mean_squared_error, r2_score,
                     fit_pixel, predict_pixel,
                     accumulated_squared_error,
                     sigmoid, inverse_sigmoid)

from sclouds.io.utils import (dataset_to_numpy, dataset_to_numpy_order,
                              dataset_to_numpy_grid_order,
                              dataset_to_numpy_grid,
                              get_xarray_dataset_for_period,
                              dataset_to_numpy_order_traditional_ar,
                              dataset_to_numpy_order_traditional_ar_grid)

import os,sys,inspect
#currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#¤parentdir = os.path.dirname(currentdir)

sys.path.insert(0,'/uio/hume/student-u89/hannasv/MS/sclouds/')
from helpers import (merge, get_list_of_variables_in_ds,
                             get_pixel_from_ds, path_input, path_ar_results)

#sys.path.insert(0,'/uio/hume/student-u89/hannasv/MS/sclouds/io/')

def get_list_of_files_traditional_model(start = '2012-01-01', stop = '2012-01-31', include_start = True, include_stop = True):
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
        subset = glob.glob(os.path.join( path_input, '{}*tcc*.nc'.format(start_search_str)))
    else:
        # get all files
        files = glob.glob(os.path.join( path_input, '*.nc' ))
        files = np.sort(files) # sorting then for no particular reson

        min_fil = os.path.join(path_input, start_search_str + '_tcc.nc')
        max_fil = os.path.join(path_input, stop_search_str + '_tcc.nc')

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


def get_list_of_files_excluding_period(start = '2012-01-01', stop = '2012-01-31'):

    first_period = get_list_of_files(start = '2004-04-01', stop = start,
                                include_start = True, include_stop = False)
    last_period = get_list_of_files(start = stop, stop = '2018-12-31',
                        include_start = False, include_stop = True)
    entire_period = list(first_period) + list(last_period)
    return entire_period

class TRADITIONAL_AR_model:
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
    transform : bool, default = True
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
        Filename becomes timestamp in utctime.
    """

    def __init__(self, start = None, stop = None,
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


        if((start is None and stop is None) and
                (test_start is not None and test_stop is not None) ):

            files = get_list_of_files_excluding_period(test_start, test_stop)
            self.dataset = merge(files)

        elif((start is None and stop is None) and
                (test_start is None and test_stop is None) ):
                raise ValueError('Something is wrong with')

        else:
            # Based on start and stop descide which files it gets.
            self.dataset = get_xarray_dataset_for_period(start = self.start,
                                                         stop = self.stop)


        print('Finished loaded the dataset')
        self.order = order

        self.longitude = self.dataset.longitude.values
        self.latitude  = self.dataset.latitude.values
        self.variables = [] #get_list_of_variables_in_ds(self.dataset)

        self.coeff_matrix = None
        self.evaluate_ds  = None

        self.transform   = transform
        self.sigmoid     = sigmoid

        # Initialize containers if data should be transformed
        if self.transform:
            """ Read transformation from the correct folder in lagringshotellet """

            self.mean = np.zeros((len(self.latitude),
                                  len(self.longitude),
                                  len(self.variables)+self.order))

            self.std  = np.zeros((len(self.latitude),
                                  len(self.longitude),
                                  len(self.variables)+self.order))
            self.bias = False
        else:
            self.bias = True

        self.X_train = None
        self.y_train = None
        return

    def load(self, lat, lon):

        # Move some of this to the dataloader part?
        ds     = get_pixel_from_ds(self.dataset, lat, lon)
        #print(ds)
        if self.order > 0:
            X, y   = dataset_to_numpy_order_traditional_ar(ds, order = self.order, bias = self.bias)
        else:
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
        lat : float
            Latitude coordinate.
        lon : float
            Longitude coordinate.

        Returns
        ---------------------
        mean, std : float
            Values used in transformation
        """
        from sklearn.preprocessing import StandardScaler
        # Move some of this to the dataloader part?
        ds     = get_pixel_from_ds(self.dataset, lat, lon)

        if self.order > 0:
            X, y   = dataset_to_numpy_order_traditional_ar(ds, order = self.order, bias = self.bias)
        else:
            X, y   = dataset_to_numpy(ds, bias = self.bias)

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

    def fit_evaluate(self):
        raise NotImplementedError('Coming soon ... ')

    def fit(self):
        """ Fits the data retrieved in the constructor, entire grid.
        """
        print('Traines model ....')
        num_vars = self.bias + len(self.variables) + self.order

        coeff_matrix = np.zeros((len(self.latitude),
                                 len(self.longitude),
                                 num_vars))

        _X = np.zeros((len(self.dataset.time.values)-self.order,
                       len(self.latitude),
                       len(self.longitude),
                       num_vars))

        _y = np.zeros((len(self.dataset.time.values)-self.order,
                       len(self.latitude),
                       len(self.longitude),
                       1))

        for i, lat in enumerate(self.latitude): # 81
            for j, lon in enumerate(self.longitude): # 161

                if self.transform:
                    X, y, mean, std = self.load_transform(lat, lon)
                    self.mean[i, j, :]  =  mean.flatten()
                    self.std[i, j, :] =  std.flatten()
                else:
                    X, y = self.load(lat, lon)
                if self.order > 0:
                    _X[:, i, j, :] = X
                    _y[:, i, j, :] = y
                else:
                    _X[:, i, j, :] = X
                    _y[:, i, j, :] = y
                #print('Number of samples after removal of nans {}.'.format(len(y)))
                coeffs = fit_pixel(X, y)
                coeff_matrix[i, j, :] =  coeffs.flatten()
            print('Finished with pixel {}/{}'.format((i+1)*j, 81*161))
        self.X_train = _X
        self.y_train = _y
        self.coeff_matrix = coeff_matrix
        return coeff_matrix

    def set_transformer_from_loaded_model(self, mean, std):
        """ Set loaded tranformation

        Parameters
        ---------------------
        mean : array-like
            Matrix containing the means used in transformation of different
            pixels.
        std : array-like
            Matrix containing the standarddeviation used in transformation
            of different pixels.
        """
        self.transform = True
        self.mean = mean
        self.std = std
        return

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
        Y      = np.zeros( (n_time-self.order, n_lat, n_lon, 1)  )

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
        print('Evaluation .... ')
        # Checks if test_start and test_stop is provided.
        if self.test_start is not None and self.test_stop is not None:
            # Based on start and stop descide which files it gets.
            files = get_list_of_files(start = self.test_start, stop = self.test_stop,
                            include_start = True, include_stop = True)
            dataset = merge(files)

            if self.order > 0:
                X, y_true = dataset_to_numpy_order_traditional_ar_grid(dataset, self.order, bias = self.bias)
                print('Detects shap Xtest {} and ytest {}'.format( np.shape(X), np.shape(y_true)  ))
            else:
                X, y_true = dataset_to_numpy_grid(dataset, bias = self.bias)
            y_pred = self.predict(X)
        else:
            #raise NotImplementedError('Coming soon ... get_evaluation()')
            print("X shape {}, y shape {}".format(self.X_train.shape, self.y_train.shape))
            y_pred = self.predict(X) # prediction based on testset and
            y_true = self.y_train

        print('before shape pred {}'.format(np.shape(y_pred)))
        y_pred = y_pred[:,:,:,0]
        print('after shape pred {}'.format(np.shape(y_pred)))

        # Move most of content in store performance to evaluate
        mse  = mean_squared_error(y_true, y_pred)
        print('mse shape {}'.format(np.shape(mse)))
        ase  = accumulated_squared_error(y_true, y_pred)
        r2   = r2_score(y_true, y_pred)

        print('(~np.isnan(X)).sum(axis=0) {}'.format(np.shape(
                                                (~np.isnan(X)).sum(axis=0))))
        print('(~np.isnan(self. Xtrain)).sum(axis=0) {}'.format(np.shape(
                                    (~np.isnan(self.X_train)).sum(axis=0))))


        vars_dict = {'mse': (['latitude', 'longitude'], mse),
                     'r2':  (['latitude', 'longitude'], r2),
                     'ase': (['latitude', 'longitude'], ase),

                     'num_train_samples': (['latitude', 'longitude'],
                                    (~np.isnan(self.X_train)).sum(axis=0)[:,:,0]),
                     'num_test_samples': (['latitude', 'longitude'],
                                    (~np.isnan(X)).sum(axis=0)[:,:,0]),

                     'global_mse': np.mean(mse),
                     'global_r2':  np.mean(r2),
                     'global_ase': np.mean(ase),

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
                     'bias'       : self.bias}
        return temp_dict

    def get_weights(self):
        """Returns dictionary of weigths fitted using this model.
        """
        temp_dict = {}

        if self.bias:
            temp_dict['b'] = (['latitude', 'longitude'], self.coeff_matrix[:, :, 0])
            ar_index = 1
        else:
            ar_index = 0

        if self.order > 0:
            for i in range(self.order):
                var = 'W{}'.format(i+1)
                temp_dict[var] = (['latitude', 'longitude'], self.coeff_matrix[:, :, ar_index])
                ar_index+=1
        """
        vars_dict = {'Wt2m':(['latitude', 'longitude'], self.coeff_matrix[:, :, 1]),
                     'Wr': (['latitude', 'longitude'], self.coeff_matrix[:, :, 2]),
                     'Wq':(['latitude', 'longitude'], self.coeff_matrix[:, :, 0]),
                     'Wsp': (['latitude', 'longitude'], self.coeff_matrix[:, :, 3]),
                      }
        temp_dict.update(vars_dict) # meges dicts together
        """
        return temp_dict

    def get_tranformation_properties(self):
        """ Returns dictionary of the properties used in the transformations,
        pixelwise mean and std.
        """
        # TODO update this based on

        temp_dict = {}

        if self.bias:
            temp_dict['b'] = (['latitude', 'longitude'], self.coeff_matrix[:, :, 0])
            ar_index = 1
        else:
            ar_index = 0

        if self.order > 0:
            for i in range(self.order):
                var = 'W{}'.format(i+1)
                temp_dict[var] = (['latitude', 'longitude'], self.coeff_matrix[:, :, ar_index])
                ar_index+=1
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
        """
        return temp_dict

    def save(self):
        """ Saves model configuration, evaluation, transformation into a file
        named by the current time. Repo : /home/hanna/lagrings/results/ar/
        """
        filename      = os.path.join(path_ar_results, 'AR_traditional{}.nc'.format(np.datetime64('now')))
        print('Stores file {}'.format(filename))
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

if __name__ == '__main__':

    m = TRADITIONAL_AR_model(start = '2012-01-01',      stop = '2012-01-03',
                 test_start = '2012-03-01', test_stop = '2012-03-03',
                 order = 1,                 transform = True,
                 sigmoid = False)
    coeff = m.fit()
    m.save()

    print(m.get_configuration())
