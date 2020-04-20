import os
import glob

import xarray as xr
import numpy as np

# Import paths
from sclouds.helpers import (path_input, path_ar_results,
                             path_convlstm_results, path_stats_results)

# Import helper xarray functions
from sclouds.helpers import merge
from sclouds.io.utils import (get_xarray_dataset_for_period,
                              dataset_to_numpy_grid, dataset_to_numpy_order,
                              dataset_to_numpy, get_list_of_files)

class DataLaderAR:
    """ The other classes should be subclasses of this having a transform()
        function.

    Normalization is scaling this to be between 0 and 1.
    Standardisation transforms data to have 0 mean and std = 1.


    Attributes
    ---------------------


    Methods
    --------------------


    """
    def __init__(self, start, stop, order):
        self.bias  = true
        self.start = start
        self.stop  = stop
        self.order = order
        return

    def load_pixel(self, lat, lon):
        files = get_list_of_files(start = self.start,
                                  stop = self.stop)
        print("Loading {} files.".format(len(files)))
        self.dataset = merge(files)
        # Move some of this to the dataloader part?

        # TODO: This loops over lat lons
        ds     = get_pixel_from_ds(self.dataset, lat, lon)
        X, y   = dataset_to_numpy(ds, bias = self.bias)
        print('Number of samples prior to removal of nans {}.'.format(len(y)))

        # Removes nan's
        a = np.concatenate([X, y], axis = 1)
        a = a[~np.isnan(a).any(axis = 1)]
        # This is where you do the transformation
        X = a[:, :-1]
        y = a[:, -1, np.newaxis] # not tested
        return X, y

    def load(self, lat, lon):
        files = get_list_of_files(start = self.start,
                                  stop = self.stop)
        print("Loading {} files.".format(len(files)))
        self.dataset = merge(files)
        # Move some of this to the dataloader part?
        means  = np.zeros((len(self.latitude),
                           len(self.longitude),
                           len(self.variables)))

        stds   = np.zeros((len(self.latitude),
                           len(self.longitude),
                           len(self.variables)))

        for i, lat in enumerate(self.latitude): # 81
            for j, lon in enumerate(self.longitude): # 161
                ds     = get_pixel_from_ds(self.dataset, lat, lon)
                X, y   = dataset_to_numpy(ds, bias = self.bias)
                print('Number of samples prior to removal of nans {}.'.format(len(y)))

                # Removes nan's
                a = np.concatenate([X, y], axis = 1)
                a = a[~np.isnan(a).any(axis = 1)]
                # This is where you do the transformation
                X = a[:, :-1]
                y = a[:, -1, np.newaxis]

        self.mean = means
        self.std  = stds
        return X, y



class DataLaderAR_standardize(DataLaderAR):

    def __init__(self, start, stop, order):
        super().__init__(self, start, stop, order)
        self.bias = False
        self.order = order
        print("Test that bias = False : {}".format(self.bias))
        # Store data used in transformation
        self.mean = None
        self.std  = None
        return

    # override load function to perform the transformation
    def load(self, lat, lon):
        from sklearn.preprocessing import StandardScaler
        files = get_list_of_files(start = self.start,
                                  stop = self.stop)
        print("Loading {} files.".format(len(files)))
        self.dataset = merge(files)
        # Move some of this to the dataloader part?
        means  = np.zeros((len(self.latitude),
                           len(self.longitude),
                           len(self.variables)))

        stds   = np.zeros((len(self.latitude),
                           len(self.longitude),
                           len(self.variables)))

        for i, lat in enumerate(self.latitude): # 81
            for j, lon in enumerate(self.longitude): # 161
                ds     = get_pixel_from_ds(self.dataset, lat, lon)
                X, y   = dataset_to_numpy(ds, bias = self.bias)
                print('Number of samples prior to removal of nans {}.'.format(len(y)))

                # Removes nan's
                a = np.concatenate([X, y], axis = 1)
                a = a[~np.isnan(a).any(axis = 1)]
                # This is where you do the transformation
                X = a[:, :-1]
                y = a[:, -1, np.newaxis]


        self.mean = means
        self.std  = stds
        return X, y

class DataLaderAR_normalize(DataLaderAR):

    def __init__(self, start, stop, order):
        super().__init__(self, start, stop, order)
        self.bias = False

        self.min = None
        self.max = None
        return

    # Override load function to perform the transformation
    def load(self):
        """ Includes the nans that are present. They are disregarded in fit()
         and included in transform().

        """
        from sklearn.preprocessing import MinMaxScaler
        files = get_list_of_files(start = self.start,
                                  stop = self.stop)
        print("Loading {} files.".format(len(files)))
        self.dataset = merge(files)
        # Move some of this to the dataloader part?
        mins  = np.zeros((len(self.latitude),
                           len(self.longitude),
                           len(self.variables)))

        maxs   = np.zeros((len(self.latitude),
                           len(self.longitude),
                           len(self.variables)))

        for i, lat in enumerate(self.latitude): # 81
            for j, lon in enumerate(self.longitude): # 161
                # Move some of this to the dataloader part?
                ds     = get_pixel_from_ds(self.dataset, lat, lon)
                X, y   = dataset_to_numpy(ds, bias = self.bias)
                print('Number of samples prior to removal of nans {}.'.format(len(y)))

                # This is where you do the transformation
                minmax = MinMaxScaler()
                X[:, i, j, :] = minmax.fit_transform(pX)
                mins[i, j, :] = minmax.data_min_.flatten()
                maxs[i, j, :] = minmax.data_max_.flatten()
                # where do you intent to store the transformed data of uneven lengths
        self.min = mins
        self.max = maxs
        return X, y


class DataLaderKeras_normalize:

    def __init__(self, seq_length):
        self.bias = False
        self.num_vars = 4 # alltid lik 4
        self.seq_length = seq_length

        # Store data used in transformation
        # Since its all the same model the should be one mean and std for
        # the entire grid
        self.mean = None
        self.std  = None
        return

    def load():
        pass

    def transform():
        pass

    def batch_normalize(data, seq_len = 4):
        samples, metvars, lats, lons = data.shape
        #.mean(axis = 0).mean(axis=1).mean(axis=1)

        normalized = np.zeros((samples, metvars, lats, lons))
        means   = np.zeros(metvars)
        storage = np.zeros(metvars)

        for i in range(metvars):
            raveled = data[:, i, :, :].reshape(-1)
            m = raveled.mean()
            s = raveled.std()
            normalized[:, i, :, :] =  (data[:, i, :, :] - m)/s

        samples, metvars, lats, lons = normalized.shape
        assert seq_len % 4 == 0

        new_samples = int(samples/seq_len)
        normalized  = normalized.reshape( (new_samples, seq_len, metvars, lats, lons ) )

        return normalized, means, storage

class DataLaderKeras_standardize:

    def __init__(self, seq_length):
        self.bias = False
        self.num_vars = 4 # alltid lik 4
        self.seq_length = seq_length

        # Since its all the same model the should be one mean and std for
        # the entire grid
        self.min = None
        self.max = None
        return

    def transform():
        pass
