import os
import glob

import numpy as np
import xarray as xr

from sclouds.ml.regression.utils import write_path, read_path

class AR_model_loader:
    """ Load trained model
    """
    def __init__(self):
        # 1. Period
        # 2. Vars
        # Order ar models, number of timesteps.
        self.weights_ds = None
        self.W_numpy = None

    def get_model(self,):
        """ Gets string models """
        # gets the filename of the requested model
        return os.path.join(write_path, 'test.nc')

    def load_model_to_xarray(self):
        """ Loades the model into a xarray dataset.
        """
        self.weights_ds = xr.open_dataset(self.get_model())
        return self.weights_ds

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
