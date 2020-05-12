""" Explenation of the content of this file.
"""

import os
import glob

import numpy as np
import xarray as xr

from sclouds.helpers import (merge, get_list_of_variables_in_ds,
                             get_pixel_from_ds, path_input, path_ar_results)

from sclouds.io.utils import (dataset_to_numpy, dataset_to_numpy_order,
                              dataset_to_numpy_grid_order,
                              dataset_to_numpy_grid,
                              get_xarray_dataset_for_period)

from sclouds.ml.regression.utils import (mean_squared_error, r2_score,
                                         fit_pixel, predict_pixel,
                                         accumulated_squared_error,
                                         sigmoid, inverse_sigmoid)

base = '/home/hanna/lagrings/results/stats/2014-01-01_2018-12-31/'
#base = '/uio/lagringshotell/geofag/students/metos/hannasv/results/stats/test/'

def min_max_scaling(dummy):
    """ Forces all values to be between 0 and 1.
    """
    n_times, n_lat, n_lon, n_vars = dummy.shape
    transformed = np.zeros(dummy.shape)
    from sclouds.helpers import VARIABLES
    for j, var in enumerate(VARIABLES):

        vmin = xr.open_dataset(base + 'stats_pixel_{}_all.nc'.format(var))['min'].values
        vmax = xr.open_dataset(base + 'stats_pixel_{}_all.nc'.format(var))['max'].values

        if var == 'tcc':
            # Something wierd with the rotation of cloud cover values
            vmin = np.flipud(vmin)
            vmax = np.flipud(vmax)

        for i in range(n_times):
            transformed[i, :, :, j] =  (dummy[i, :, :, j]  - vmin)/(vmax-vmin)
    return transformed

def normalization(dummy = np.random.random(( 744, 81, 161, 5))):
    """ Normalizes the distribution. It is centered around the mean with std of 1.

    Subtract the mean divide by the standard deviation. """
    from sclouds.helpers import VARIABLES
    n_times, n_lat, n_lon, n_vars = dummy.shape
    transformed = np.zeros(dummy.shape)
    for j, var in enumerate(VARIABLES):

        m = xr.open_dataset(base + 'stats_pixel_{}_all.nc'.format(var))['mean'].values
        s = xr.open_dataset(base + 'stats_pixel_{}_all.nc'.format(var))['std'].values

        if var == 'tcc':
            # Something wierd with the rotation of cloud cover values
            m = np.flipud(m)
            s = np.flipud(s)

        for i in range(n_times):
            transformed[i, :, :, j] =  (dummy[i, :, :, j]  - m)/s
    return transformed

def reverse_min_max_scaling():
    """ Forces all values to be between 0 and 1.
    """
    n_times, n_lat, n_lon, n_vars = dummy.shape
    from sclouds.helpers import VARIABLES
    transformed = np.zeros(dummy.shape)
    for j, var in enumerate(VARIABLES):

        vmin = xr.open_dataset(base + 'stats_pixel_{}_all.nc'.format(var))['min'].values
        vmax = xr.open_dataset(base + 'stats_pixel_{}_all.nc'.format(var))['max'].values

        if var == 'tcc':
            # Something wierd with the rotation of cloud cover values
            vmin = np.flipud(vmin)
            vmax = np.flipud(vmax)

        for i in range(n_times):
            transformed[i, :, :, j] =  (dummy[i, :, :, j] + vmin)*(vmax-vmin)
    return transformed

def reverse_normalization(dummy = np.random.random(( 744, 81, 161, 5))):
    """ Normalizes the distribution. It is centered around the mean with std of 1.

    Subtract the mean divide by the standard deviation. """
    from sclouds.helpers import VARIABLES
    n_times, n_lat, n_lon, n_vars = dummy.shape
    transformed = np.zeros(dummy.shape)
    for j, var in enumerate(VARIABLES):

        m = xr.open_dataset(base + 'stats_pixel_{}_all.nc'.format(var))['mean'].values
        s = xr.open_dataset(base + 'stats_pixel_{}_all.nc'.format(var))['std'].values

        if var == 'tcc':
            # Something wierd with the rotation of cloud cover values
            m = np.flipud(m)
            s = np.flipud(s)

        for i in range(n_times):
            transformed[i, :, :, j] =  (dummy[i, :, :, j]  + m)*s
    return transformed



def get_list_of_files_excluding_period(start = '2012-01-01', stop = '2012-01-31'):

    first_period = get_list_of_files(start = '2004-04-01', stop = start,
                                include_start = True, include_stop = False)
    last_period = get_list_of_files(start = stop, stop = '2018-12-31',
                        include_start = False, include_stop = True)
    entire_period = list(first_period) + list(last_period)
    return entire_period

def get_list_of_files(start = '2012-01-01', stop = '2012-01-31', include_start = True, include_stop = True):
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
        subset = glob.glob(os.path.join( path_input, '{}*.nc'.format(start_search_str)))
    else:
        # get all files
        files = glob.glob(os.path.join( path_input, '*.nc' ))
        files = np.sort(files) # sorting then for no particular reson

        if include_start and include_stop:
            min_fil = os.path.join(path_input, start_search_str + '_q.nc')
            max_fil = os.path.join(path_input, stop_search_str + '_tcc.nc')

            smaller = files[files <= max_fil]
            subset  = smaller[smaller >= min_fil] # results in all the files

        elif include_start and not include_stop:
            min_fil = os.path.join(path_input, start_search_str + '_q.nc')
            max_fil = os.path.join(path_input, stop_search_str + '_q.nc')

            smaller = files[files < max_fil]
            subset  = smaller[smaller >= min_fil] # results in all the files

        elif not include_start and include_stop:
            min_fil = os.path.join(path_input, start_search_str + '_tcc.nc')
            print('detected min fil {}'.format(min_fil))
            max_fil = os.path.join(path_input, stop_search_str + '_tcc.nc')

            smaller = files[files <= max_fil]
            subset  = smaller[smaller > min_fil] # results in all the files
        else:
            raise ValueError('Something wierd happend. ')

    #assert len(subset)%5==0, "Not five of each files, missing variables in file list!"
    #assert len(subset)!=0, "No files found, check if you have mounted lagringshotellet."

    return subset



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


        print('Finished loading the dataset ... ')
        self.order = order

        self.longitude = self.dataset.longitude.values
        self.latitude  = self.dataset.latitude.values
        self.variables = ['t2m', 'q', 'r', 'sp'] #get_list_of_variables_in_ds(self.dataset)

        self.test_dataset = None
        self.coeff_matrix = None
        self.evaluate_ds  = None

        self.mse = None
        self.r2 = None
        self.ase = None

        self.num_test_samples = None
        self.num_train_samples = None

        self.transform   = transform
        self.sigmoid     = sigmoid

        # Initialize containers if data should be transformed
        if self.transform:
            """ Read transformation from the correct folder in lagringshotellet """
            self.bias = False
        else:
            self.bias = True

        self.X_train = None
        self.y_train = None
        return

    def transform_data(self, X, y):
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
        #a = np.concatenate([X, y], axis = 1)
        #a = a[~np.isnan(a).any(axis = 1)]

        #X = a[:, :, :, :-1]

        #if self.sigmoid:
        #    y = inverse_sigmoid(a[:-1, np.newaxis]) # not tested
        #else:
        #y_removed_nans = a[:, :, :, -1, np.newaxis]

        order = self.order
        n_times, n_lat, n_lon, n_vars = X.shape
        #VARIABLES = ['t2m', 'q', 'r', 'sp']
        transformed = np.zeros((n_times, n_lat, n_lon, n_vars ))


        for j, var in enumerate(self.variables):

            m = xr.open_dataset(base + 'stats_pixel_{}_all.nc'.format(var))['mean'].values
            s = xr.open_dataset(base + 'stats_pixel_{}_all.nc'.format(var))['std'].values

            for i in range(n_times):
                transformed[i, :, :, j] =  (X[i, :, :, j]  + m)*s

        if order > 0:
            var = 'tcc'
            m = xr.open_dataset(base + 'stats_pixel_{}_all.nc'.format(var))['mean'].values
            s = xr.open_dataset(base + 'stats_pixel_{}_all.nc'.format(var))['std'].values
            for k in range(order):
                # Something wierd with the rotation of cloud cover values
                transformed[:, :, :, k+j+1] = (X[:, :, :, k+j+1]- m)/s

        return transformed

    def load(self, lat, lon):

        # Move some of this to the dataloader part?
        ds     = get_pixel_from_ds(self.dataset, lat, lon)

        if self.order > 0:
            X, y   = dataset_to_numpy_order(ds, order = self.order, bias = self.bias)
        else:
            X, y   = dataset_to_numpy(ds, bias = self.bias)

        # print('Number of samples prior to removal of nans {}.'.format(len(y)))
        # Removes nan's
        a = np.concatenate([X, y], axis = 1)
        B = a[~np.isnan(a).any(axis = 1)]
        X = B[:, :-1]
        y = B[:, -1, np.newaxis] # not tested
        return X, y
    """
    def load_transform(self, lat, lon):

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        return X, y, scaler.mean_, np.sqrt(scaler.var_)
    """

    def reverse_normalization(self, order, X):
        """ Normalizes the distribution. It is centered around the mean with std of 1.

        Subtract the mean divide by the standard deviation. """
        from sclouds.helpers import VARIABLES
        n_times, n_lat, n_lon, n_vars =  X.shape

        transformed = np.zeros((n_times, n_lat, n_lon, 4 + order ))

        for j, var in enumerate(VARIABLES):

            m = xr.open_dataset(base + 'stats_pixel_{}_all.nc'.format(var))['mean'].values
            s = xr.open_dataset(base + 'stats_pixel_{}_all.nc'.format(var))['std'].values

            for i in range(n_times):
                transformed[i, :, :, j] =  (X[i, :, :, j]  + m)*s

        if order > 0:
            var = 'tcc'
            for k in range(order):
                m = xr.open_dataset(base + 'stats_pixel_{}_all.nc'.format(var))['mean'].values
                s = xr.open_dataset(base + 'stats_pixel_{}_all.nc'.format(var))['std'].values
                # Something wierd with the rotation of cloud cover values
                m = np.flipud(m)
                s = np.flipud(s)
                for i in range(n_times):
                    transformed[i, :, :, k+j+1] =  (X[i, :, :, k+j+1]  + m)*s

        return transformed

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
        """ Normalizes the distribution. It is centered around the mean with std of 1.

        Subtract the mean divide by the standard deviation. """
        # Move some of this to the dataloader part?
        ds     = get_pixel_from_ds(self.dataset, lat, lon)

        if self.order > 0:
            X, y   = dataset_to_numpy_order(ds, order = self.order, bias = self.bias)
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

        order = self.order
        n_times, n_vars = X.shape
        VARIABLES = ['t2m', 'q', 'r', 'sp']
        transformed = np.zeros((n_times, 4 + order ))

        for j, var in enumerate(self.variables):

            m = xr.open_dataset(base + 'stats_pixel_{}_all.nc'.format(var))['mean'].sel(latitude = lat, longitude = lon).values
            s = xr.open_dataset(base + 'stats_pixel_{}_all.nc'.format(var))['std'].sel(latitude = lat, longitude = lon).values

            transformed[:, j] = (X[:, j]- m)/s
            #for i in range(n_times):
            #    transformed[i, :, :, j] =  (X[i, :, :, j]  - m)/s
        if order > 0:
            var = 'tcc'
            m = xr.open_dataset(base + 'stats_pixel_{}_all.nc'.format(var))['mean'].sel(latitude = lat, longitude = lon).values
            s = xr.open_dataset(base + 'stats_pixel_{}_all.nc'.format(var))['std'].sel(latitude = lat, longitude = lon).values

            for k in range(order):
                # Something wierd with the rotation of cloud cover values
                transformed[:, k+j+1] = (X[:, j+k+1]- m)/s

        return transformed, y

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
        transformed = np.zeros((X.shape[0], 4 + order ))

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

        if self.test_start is not None and self.test_stop is not None:
            # Load test data
            print('Loads test data')
            files = get_list_of_files(start = self.test_start, stop = self.test_stop,
                        include_start = True, include_stop = True)
            self.test_dataset = merge(files)


        ######### FIT
        # TODO disse må flytten til den funksjonen som
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


        num_train_samples = np.zeros((len(self.latitude),
                                 len(self.longitude)))

        num_test_samples = np.zeros((len(self.latitude),
                                 len(self.longitude)))

        for i, lat in enumerate(self.latitude): # 81
            for j, lon in enumerate(self.longitude): # 161

                if self.transform:
                    coeff, mse, ase, r2, num_test, num_train = self.load_transform_fit(lat, lon)
                    coeff_matrix[i, j, :] = coeff
                    mse_storage[i, j] = mse
                    r2_storage[i, j] = r2
                    ase_storage[i, j] = ase
                    num_train_samples[i,j] = num_test
                    num_test_samples[i,j] = num_train
                else:
                    raise NotImplementedError('Implement this shit .... ')
                    coeff, mse, ase, r2, num_test, num_train = self.load_fit(lat, lon)
                    coeff_matrix[i, j, :] = coeff
                    mse_storage[i, j] = mse
                    r2_storage[i, j] = r2
                    ase_storage[i, j] = ase
                    num_train_samples[i,j] = num_test
                    num_test_samples[i,j] = num_train

            print('Finished with pixel {}/{}'.format((i+1)*j, 81*161))

        self.coeff_matrix = coeff_matrix
        self.mse = mse_storage
        self.ase = ase_storage
        self.r2 = r2_storage
        self.num_test_samples = num_test_samples
        self.num_train_samples = num_train_samples
        return


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
        ds     = get_pixel_from_ds(self.dataset, lat, lon)

        if self.order > 0:
            X, y   = dataset_to_numpy_order(ds, order = self.order, bias = self.bias)

        #print(X.shape)
        #print(y.shape)
        # else:
        #    X, y   = dataset_to_numpy_r_traditional_ar(ds, bias = self.bias)

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
            transformed_train = np.zeros((X.shape[0], 4 + order ))

            for j, var in enumerate(self.variables):

                m = xr.open_dataset(base + 'stats_pixel_{}_all.nc'.format(var))['mean'].sel(latitude = lat, longitude = lon).values
                s = xr.open_dataset(base + 'stats_pixel_{}_all.nc'.format(var))['std'].sel(latitude = lat, longitude = lon).values

                transformed_train[:, j] = (X[:, j]- m)/s
                #for i in range(n_times):
                #    transformed[i, :, :, j] =  (X[i, :, :, j]  - m)/s
            if order > 0:
                var = 'tcc'
                for k in range(order):
                    m = xr.open_dataset(base + 'stats_pixel_{}_all.nc'.format(var))['mean'].sel(latitude = lat, longitude = lon).values
                    s = xr.open_dataset(base + 'stats_pixel_{}_all.nc'.format(var))['std'].sel(latitude = lat, longitude = lon).values
                    # Something wierd with the rotation of cloud cover values
                    transformed_train[:, k+j+1] = (X[:, j+k+1]- m)/s

            X_train = transformed_train

        if self.test_start is not None and self.test_stop is not None:
            # Based on start and stop descide which files it gets.

            ds     = get_pixel_from_ds(self.test_dataset, lat, lon)
            print(ds)
            if self.order > 0:
                X_test, y_test_true = dataset_to_numpy_order(ds, self.order, bias = self.bias)
                n_times, n_vars = X_test.shape
                #VARIABLES = ['t2m', 'q', 'r', 'sp']
                if self.transform:
                    transformed_test = np.zeros((n_times, n_vars ))

                    for j, var in enumerate(self.variables):

                        m = xr.open_dataset(base + 'stats_pixel_{}_all.nc'.format(var))['mean'].sel(latitude = lat, longitude = lon).values
                        s = xr.open_dataset(base + 'stats_pixel_{}_all.nc'.format(var))['std'].sel(latitude = lat, longitude = lon).values

                        transformed_test[:, j] = (X_test[:, j]- m)/s

                    if order > 0:
                        var = 'tcc'
                        m = xr.open_dataset(base + 'stats_pixel_{}_all.nc'.format(var))['mean'].sel(latitude = lat, longitude = lon).values
                        s = xr.open_dataset(base + 'stats_pixel_{}_all.nc'.format(var))['std'].sel(latitude = lat, longitude = lon).values

                        for k in range(order):
                            # Something wierd with the rotation of cloud cover values
                            transformed_test[:, k+j+1] = (X_test[:, k+j+1]- m)/s

                    X_test = transformed_test

                print('Detects shap Xtest {} and ytest {}'.format( np.shape(X_test), np.shape(y_test_true)  ))

        num_test = (~np.isnan(X_test)).sum(axis=0)[0]
        #print(num_test)
        num_train = (~np.isnan(X_train)).sum(axis=0)[0]
        coeffs = fit_pixel(X, y)
        #print(coeffs)
        #print(X_test)
        y_test_pred = predict_pixel(X_test, coeffs)

        if self.sigmoid:
            y_test_pred = inverse_sigmoid(y_test_pred)

        # TODO: upgrade this to compute train  score as well as test score.
        # y_pred = self.predict(X) # prediction based on testset and
        # y_true = self.y_train

        #print('before shape pred {}'.format(np.shape(y_pred)))
        #y_pred = y_pred[:,:,:,0]
        #print('after shape pred {}'.format(np.shape(y_pred)))

        print(y_test_true.shape, y_test_pred.shape)

        # Move most of content in store performance to evaluate
        mse  = mean_squared_error(y_test_true, y_test_pred)[0]
        print('mse shape {}'.format(np.shape(mse)))
        ase  = accumulated_squared_error(y_test_true, y_test_pred)[0]
        r2   = r2_score(y_test_true, y_test_pred)[0]
        #print(mse, ase, r2)
        return coeffs.flatten(), mse, ase, r2, num_test, num_train


    def fit_evaluate(self):
        raise NotImplementedError('Coming soon ... ')

    #def fit(self):
    """ Fits the data retrieved in the constructor, entire grid.
    """
    """
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
                X, y = self.load_transform(lat, lon)
                #self.mean[i, j, :]  =  mean.flatten()
                #self.std[i, j, :] =  std.flatten()
            else:
                X, y = self.load(lat, lon)

            _X[:, i, j, :] = X
            _y[:, i, j, :] = y
    """
    """
            if self.order > 0:
                _X[:, i, j, :] = X
                _y[:, i, j, :] = y
            else:
                _X[:, i, j, :] = X
                _y[:, i, j, :] = y"""
            #print('Number of samples after removal of nans {}.'.format(len(y)))
    """
    coeffs = fit_pixel(X, y)
            coeff_matrix[i, j, :] =  coeffs.flatten()
        print('fit {}/{}'.format((i+1)*j), 81*161)
    self.X_train = _X
    self.y_train = _y
    self.coeff_matrix = coeff_matrix
    return coeff_matrix"""

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
        Deprecated
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

        for i, lat in enumerate(self.latitude.values):
            for j, lon in enumerate(self.longitude.values):
                a = X[:, i, j, :]

                # Checks if data should be transformed and performs
                # transformation.
                if self.transform:
                    _X, _y = transform(X, y, lat, lon)


                #_X = a[~np.isnan(a).any(axis=1)]
                _w = self.coeff_matrix[i, j, :, np.newaxis]

                y_pred = predict_pixel(_X, _w)
                # if trained on sigmoid this contains values in that range.

                Y[:, i, j, 0] = y_pred.flatten()
        return Y
    def get_evaluation(self):
        """Evaluation"""
        vars_dict = {'mse': (['latitude', 'longitude'], self.mse),
                     'r2':  (['latitude', 'longitude'], self.r2),
                     'ase': (['latitude', 'longitude'], self.ase),
                     'num_train_samples': (['latitude', 'longitude'],
                                    self.num_train_samples),
                     'num_test_samples': (['latitude', 'longitude'],
                                    self.num_test_samples),
                     'global_mse': np.mean(self.mse),
                     'global_r2':  np.mean(self.r2),
                     'global_ase': np.mean(self.ase),
                      }

        return vars_dict

    def get_evaluation_old(self):
        """ Get evaluation of data
        """
        print('Evaluation .... ')
        # Checks if test_start and test_stop is provided.
        if self.test_start is not None and self.test_stop is not None:
            # Based on start and stop descide which files it gets.
            dataset = get_xarray_dataset_for_period(start = self.test_start,
                                                    stop = self.test_stop)
            if self.order > 0:
                X, y_true = dataset_to_numpy_grid_order(dataset, self.order, bias = self.bias)
            else:
                X, y_true = dataset_to_numpy_grid(dataset, bias = self.bias)

            # TODO add tranformations and transform back.

            if self.transform:
                X, y_true = self.transform_data(X, y_true)

            y_pred = self.predict(X)

            if self.sigmoid:
                y_pred = self.inverse_sigmoid(y_pred)

        else:
            #raise NotImplementedError('Coming soon ... get_evaluation()')
            y_pred = self.predict(self.X_train)

            if self.sigmoid:
                y_true = self.inverse_sigmoid(self.y_train)

        y_pred = y_pred[:,:,:,0]

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

    def fit(self):
        """ New fit function
        """

        if self.test_start is not None and self.test_stop is not None:
            # Load test data
            print('Loads test data')
            files = get_list_of_files(start = self.test_start, stop = self.test_stop,
                        include_start = True, include_stop = True)
            self.test_dataset = merge(files)


        ######### FIT
        # TODO disse må flytten til den funksjonen som
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


        num_train_samples = np.zeros((len(self.latitude),
                                 len(self.longitude)))

        num_test_samples = np.zeros((len(self.latitude),
                                 len(self.longitude)))

        for i, lat in enumerate(self.latitude): # 81
            for j, lon in enumerate(self.longitude): # 161

                if self.transform:
                    coeff, mse, ase, r2, num_test, num_train = self.load_transform_fit(lat, lon)
                    coeff_matrix[i, j, :] = coeff
                    mse_storage[i, j] = mse
                    r2_storage[i, j] = r2
                    ase_storage[i, j] = ase
                    num_train_samples[i,j] = num_test
                    num_test_samples[i,j] = num_train
                else:
                    raise NotImplementedError('Implement this shit .... ')
            print('Finished with pixel {}/{}'.format((i+1)*j, 81*161))

        self.coeff_matrix = coeff_matrix
        self.mse = mse_storage
        self.ase = ase_storage
        self.r2 = r2_storage
        self.num_test_samples = num_test_samples
        self.num_train_samples = num_train_samples
        return



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

        vars_dict = {'mean_t2m':(['latitude', 'longitude'], self.mean[:, :, 1]),
                     'std_t2m':(['latitude', 'longitude'], self.std[:, :, 1]),
                     'mean_r':(['latitude', 'longitude'], self.mean[:, :, 2]),
                     'std_r':(['latitude', 'longitude'], self.std[:, :, 2]),
                     'mean_q':(['latitude', 'longitude'], self.mean[:, :, 0]),
                     'std_q':(['latitude', 'longitude'], self.std[:, :, 0]),
                     'mean_sp':(['latitude', 'longitude'], self.mean[:, :, 3]),
                     'std_sp':(['latitude', 'longitude'], self.std[:, :, 3]),
                      }
        temp_dict.update(vars_dict)
        return temp_dict

    def save(self):
        """ Saves model configuration, evaluation, transformation into a file
        named by the current time. Repo : /home/hanna/lagrings/results/ar/
        """
        path_ar_results = '/uio/lagringshotell/geofag/students/metos/hannasv/results/ar/'
        filename      = os.path.join(path_ar_results, 'AR_{}.nc'.format(np.datetime64('now')))

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

if __name__ == '__main__':

    start = None
    stop  = None
    test_start = '2014-01-01'
    test_stop  = '2018-12-31'
    sig = False
    trans = True

    m = AR_model(start = None,      stop = None,
                 test_start = '2014-01-01', test_stop = '2018-12-31',
                 order = 1,                 transform = True,
                 sigmoid = False)
    coeff = m.fit()
    m.save()

    print(m.get_configuration())
