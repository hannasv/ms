""" Write summary of the auto regressive model loader.
"""

import os
import glob

import numpy as np
import xarray as xr

from sclouds.helpers import  path_ar_results
from sclouds.ml.regression.model import Model
#from sclouds.ml.regression.utils import write_path, read_path

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

def get_test_data(test_start, test_stop, model = 'ar'):
    print('gets train test data for ar model')
    if type == 'ar':
        #files_train = get_list_of_files_excluding_period(test_start, test_stop)
        files_test = get_list_of_files(test_start, test_stop)

    else:
        #files_train = get_list_of_files_excluding_period_traditional_model(test_start, test_stop)
        files_test = get_list_of_files_traditional_model(test_start, test_stop)

    print('Detected {} train files and {} test files. Merging might take a while .... '.format(len(files_train), len(files_test)))
    #train_dataset = merge(files_train)
    print('finished merging train')
    test_dataset = merge(files_test)
    return test_dataset


class AR_model_loader:
    """ Load trained model

    Attributes
    ---------------------
    weights_ds : xr.Dataset
        Dataset of trained weights.
    W_numpy : array-like
        Grid of trained models.

    Methods
    ---------------------

    get_model

    load_model_to_xarray

    load_transformation

    load_model_to_numpy

    """
    def __init__(self):
        """
        Parameters
        -----------------


        """
        # 1. Period
        # 2. Vars
        # Order ar models, number of timesteps.
        self.dataset = None
        self.W_numpy = None
        self.config  = {}

    def get_test_model(self):
        """ Picks a random model.
        Returns
        ------------
        _ : str
            Path to random model.
        """
        # gets the filename of the requested model
        return os.path.join(path_ar_results, '*.nc')[0]

    def load_model_to_xarray(self, file):
        """ Loades the model into a xarray dataset.

        Parameters
        --------------
        file : str
            Requested model to load.

        Returns
        -------------
        datset : xr.Dataset
            Loads trained model into a dataset.
        """
        self.dataset = xr.open_dataset(file)
        try:
            type = self.dataset.type.values
        except KeyError:
            type = 'ar'
            print('Sets type to ar because the test case is ar.')
        self.config = {'transform': self.dataset.transform.values,
                       'sigmoid': self.dataset.sigmoid.values,
                       'order': self.dataset.order.values,
                       'start': self.dataset.start.values,
                       'stop': self.dataset.stop.values,
                       'test_start': self.dataset.test_start.values,
                       'test_stop': self.dataset.test_stop.values,
                       'bias': self.dataset.bias.values,
                       'type':self.dataset.type.values}
        return self.dataset

    def get_configuration(self):
        """ Get configuration from constructor.
        """
        return self.config

    def load_model_from_hyperparameters(self):
        """ Load model from HyperParameters.

        Returnes
        -----------------
        model : sclouds.ml.regression.AR_model
            Loads a trained model based on a hyper parametersettings.

        Issue many models might have the same settings?
        """
        raise NotImplementedError('Comming soon ....')

    def load_model_from_file(self, file):
        """ Load AR model from filename.

        Parameters
        -----------------------------
        file : str
            Filename of requested model.

        Returnes
        -----------------
        model : sclouds.ml.regression.AR_model
            Loads a trained model based on filename.
        """
        ds_model = load_model_to_xarray(file)
        test_dataset = get_test_data(test_start, test_stop, model = 'ar')
        # duplicate test as train data to aviod that data is read into model.
        model = Model(start = self.config.start, stop = self.config.stop,
                        test_start = self.config.test_start, test_stop = self.config.test_stop,
                        train_dataset = test_dataset, test_dataset = test_dataset,
                        order = 1, transform = self.config.transform, sigmoid = self.config.sigmoid,
                        latitude = None, longitude = None, type = self.dataset.type.values,
                        bias = self.config.bias)

        weights = self.load_weights(file = file)
        model.set_weights_from_loaded_model(W = weights)
        #model.set_configuration(self.config)
        return model

    def load_weights(self, file):
        """
         weights: xr.Dataset
            contains the weights and biases

         Returns
         --------------

            X : weights in matrix for

            explenation : List[str]
                Exlpaines the content in dimension 2.

            TODO:
                Make sure they come in the correct oder

            OBS kan brukes til å se fort hva modellen inneholder
        """
        if not self.dataset:
            # Havn't defined the data
            self.load_model_to_xarray(file = file)

        latitudes  = self.dataset.latitude.values
        longitudes = self.dataset.longitude.values

        n_lat = len(latitudes)
        n_lon = len(longitudes)

        # Number of variables is sum of meteorological, bias
        # (bool 0 or 1), and the order of the ar model
        order = self.dataset.order.values
        n_vars = len(self.variables) + order + self.dataset.bias.values
        weights = np.zeros((n_lat, n_lon,  n_vars))

        if len(self.variables) > 0 :
            weights[:, :, 0] = self.dataset.Wq.values
            weights[:, :, 1] = self.dataset.Wt2m.values
            weights[:, :, 2] = self.dataset.Wr.values
            weights[:, :, 3] = self.dataset.Wsp.values
            ar_index = 4
        else:
            ar_index = 0

        if self.dataset.bias.values:
            weights[:, :, ar_index] = self.dataset.b.values
            var_idx += 1

        if order > 0:
            for o in range(order):
                var = 'W{}'.format(o+1)
                weights[:, :, var_idx] = self.dataset[var].values
                var_idx += 1
        return weights

    def get_list_of_trained_AR_models(self):
        """Returns list of trained models."""
        return glob.glob(os.path.join(path_ar_results, '*.nc'))

    def get_num_trained_AR_models(self):
        """Returns number of trained models."""
        return len(glob.glob(os.path.join(path_ar_results, '*.nc')))

    def get_best_models(self, num = 1, metric = 'mse'):
        """ Find the best models

        Parameters
        ---------------------
        num : int
            Number of models you
        metric : str
            Metric desciding the criterion to pick the model.
        Returns
        ----------------
        name_best_model : str
            The name of the best model.
            List of filenames with the top ``num'' performace.
        """

        if num > 1:
            raise NotImplementedError('Comming soon ...')
        all_models = self.get_list_of_trained_AR_models()
        # Init empty container with room for num objects.
        name_best_model = self.get_best_model_from_list_of_files(all_models,
                                                                 num = 1,
                                                                 metric = 'mse')
        return name_best_model

    def get_best_hyperparameters(self, num = 1, metric = 'mse'):
        """ Find the hyperparameter settings of the corresponding to the

        Parameters
        ---------------------
        num : int
            Number of best models to return (default : 1)
        metric : str
            Metric to base the criterion on (default : 'mse')

        Returns
        ----------------
        temp_dict : dict or list of dictionaries
            Returns a dictionary of the best hyperparamsettings or
        """
        if num > 1:
            raise NotImplementedError('Comming soon ...')

        valid_metrics =  ['mse', 'ase', 'r2'] # TODO move this to somewhere appropriate.

        if not metric in valid_metrics:
            raise ValueError('Metric {} is not valid. Please try {}'.format(metric, valid_metrics))

        best_model = self.get_best_models(num = 1, metric = metric)
        data = xr.open_dataset(best_model)

        temp_dict = {'transform'  : data.transform.values,
                     'sigmoid'    : data.sigmoid.values,
                     'order'      : data.order.values,
                     'start'      : data.start.values,
                     'stop'       : data.stop.values,
                     'test_start' : data.start.values,
                     'test_stop'  : data.stop.values,
                     'bias'       : data.bias.values,
                    }
        return temp_dict

    def get_best_model_from_list_of_files(self, files, num = 1, metric = 'mse'):
        """ Returns best models from a list of files.

        Parameters
        ---------------
        files : List[str]
            List of path to models of interest.
        metric : str, optional
            Defaults to mean squared error
        """
        if num > 1:
            raise NotImplementedError('Coming soon ... ')

        # First model sets the bar
        name_best_model = files[0]
        temp_data = xr.open_dataset(name_best_model)
        performance_best_model = temp_data['global_{}'.format(metric)].values

        for model in files[1:]:
            temp_data = xr.open_dataset(model)
            # Idea store them all and then sort them by axis 1 and take the top num.
            performance = temp_data['global_{}'.format(metric)].values
            # TODO for mse we want the smallest one and for r2 we want the largest one-
            if metric == 'r2':
                if performance > performance_best_model:
                    # Updates best model.
                    name_best_model = model
                    performance_best_model = performance
            else:
                if performance < performance_best_model:
                    # Updates best model.
                    name_best_model = model
                    performance_best_model = performance
        return name_best_model

    def get_model_based_on_hyperparameters(self, start = None,
                                           stop = None,
                                           test_start = None,
                                           test_stop = None,
                                           order = None,
                                           transform = None,
                                           sigmoid = None,
                                           bias = None):
        """ Returns list of models based on relevant hyperaparameter criterions.

        Parameters
        -------------------------------------
        start : str, optional
            year-month-day
        stop : str, optional
            year-month-day
        test_start : str, optional
            year-month-day
        test_stop : str, optional
            year-month-day
        order : int, optional
            The number of previos timestep
        transform : bool, optional
            Applied transformation of data.
        sigmoid : bool, optional
            Applied sigmoid transformation on response.
        bias : bool, optional
            Including bias/ intecept.

        Returns
        -----------------------
        files : List[str]
            List of files names corresponding to the provided description.
        """

        files = [] # stores the filest where all
        all_files = self.get_list_of_trained_AR_models()

        for fil in all_files:
            # Determine which if values is not none and test that all these are true.
            model = xr.open_dataset(fil)
            # Listing the eight conditions.
            c1 = (start is None or model.start.values == start)
            c2 = (stop is None or model.stop.values == stop)
            c3 = (test_start is None or model.test_start.values == test_start)
            c4 = (test_stop is None or model.test_stop.values == test_stop)
            c5 = (order is None or model.order.values == order)
            c6 = (transform is None or model.transform.values == transform)
            c7 = (sigmoid is None or model.sigmoid.values == sigmoid)
            c8 = (bias is None or model.bias.values == bias)

            if c1 and c2 and c3 and c4 and c5 and c6 and c7 and c8:
                files.append(fil)
        return files

    def get_last_trained_model(self):
        """ Returns the name of the last trained model.

        Returns
        ----------
        last : str
            Name of most resently trained model.
        """
        all_models = loader.get_list_of_trained_AR_models()
        last = max(all_models)
        #raise NotImplementedError('Comming soon .... ')
        return last
