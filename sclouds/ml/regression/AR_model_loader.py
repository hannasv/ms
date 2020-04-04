""" Write summary of the auto regressive model loader.
"""

import os
import glob

import numpy as np
import xarray as xr

from sclouds.helpers import  path_ar_results
#from sclouds.ml.regression.utils import write_path, read_path

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
        self.weights_ds = None
        self.W_numpy = None

    def get_model(self):
        """ Gets string models """
        # gets the filename of the requested model
        return os.path.join(path_ar_results, 'test.nc')

    def load_model_to_xarray(self):
        """ Loades the model into a xarray dataset.
        """
        self.weights_ds = xr.open_dataset(self.get_model())
        return self.weights_ds

    def load_transformation(self):
        """Returns trained tranformation
        Develop this code after the
        """
        return

    def load_model_to_numpy(self):
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
        if not self.weights_ds:
            # Havn't defined the data
            self.load_model_to_xarray()

        variables = self.weights_ds.variables.keys()

        keys = [k for k in self.weights_ds.variables.keys()]
        n_vars = int(len(keys) - 9)
        latitudes = self.weights_ds.latitude.values
        longitudes = self.weights_ds.longitude.values

        n_lat = len(latitudes)
        n_lon = len(longitudes)

        n_vars = 5
        W = np.zeros((n_lat, n_lon,  n_vars))

        W_q   = self.weights_ds.Wq.values
        W_t2m = self.weights_ds.Wt2m.values
        W_r   = self.weights_ds.Wr.values
        W_sp  = self.weights_ds.Wsp.values
        b     = self.weights_ds.b.values

        W[:, :, 0] = W_q[:,:]
        W[:, :, 1] = W_t2m[:,:]
        W[:, :, 2] = W_r[:,:]
        W[:, :, 3] = W_sp[:,:]
        W[:, :, 4] = b

        # TODO : Add timesteps

        # if len(variables) > 5+2: # 2 since it contains the latitude, longitude information.
        #    for i in range(1, len(variables)-2-5+1):
        #        var = 'W{}'.format(i)
        #          W[:, :, i+4] = self.weights_ds[var].values
        return W

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
        name_best_model = self.get_best_model_from_list_of_files(files, num = 1, metric = 'mse')
        return name_best_model


    def get_best_hyperparameters(self, num = 1, metric = 'mse'):
        """ Find the hyperparameter settings of the corresponding to the

        Parameters
        ---------------------
        num : int
            Number of models you

        Returns
        ----------------
        something : dict or list of dictionaries
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
                     'test_stop'  : data.stop.values
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
            performace = temp_data['global_{}'.format(metric)].values
            if performace > performance_best_model:
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
                                           sigmoid = None):
        """ Returns list of models based on relevant hyperaparameter criterions.

        start : str, optional
            year-month-day
        stop : str, optional
            year-month-day
        test_start : str, optional
            year-month-day
        test_stop : str, optional
            year-month-day
        order : int
            The number of previos timestep
        transform : bool
            Applied transformation of data.
        sigmoid : bool
            Applied sigmoid transformation on response.
        """

        files = [] # stores the filest where all
        all_files = get_list_of_trained_AR_models()

        for fil in all_files:
            # Determine which if values is not none and test that all these are true.
            model = xr.open_dataset(fil)
            c1 = (start is None or model.start.values == start)
            c2 = (stop is None or model.stop.values == stop)
            c3 = (test_start is None or model.test_start.values == test_start)
            c4 = (test_stop is None or model.test_stop.values == test_stop)
            c5 = (order is None or model.order.values == order)
            c6 = (transform is None or model.transform.values == transform)
            c7 = (sigmoid is None or model.sigmoid.values == sigmoid)

            if c1 and c2 and c3 and c4 and c5 and c6 and c7:
                files.append(fil)
        return files
