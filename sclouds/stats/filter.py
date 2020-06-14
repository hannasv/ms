import os
import glob

import xarray as xr
import numpy as np

path_input = '/uio/lagringshotell/geofag/students/metos/hannasv/ERA5_monthly/'
path_filter = '/uio/lagringshotell/geofag/students/metos/hannasv/results/stats/'
path_filter = '/uio/hume/student-u89/hannasv/MS-suppl/'

#VARIABLES  = ['tcc', 'r', 'q', 't2m', 'sp']
VALID_FILTERS = ['coast', 'sea', 'land', 'artefact']

class Filter:
    """ Filter class for data.
    """
    def __init__(self, filter):
        if not filter:
            raise ValueError('Please provide a filter. \
            Valid filters are {}.'.format( self.VALID_FILTERS ))

        self.filter_key = filter
        self.data = None
        self.filter_ds = None

        self._load() # Automatically loads the filter.
        self.mean = None
        self.variable = None
        return

    def set_data(self, data, variable):
        """ Set the data and filters it.

        Parameters
        ------------------
        data : xr.Dataset
            Data to be filtered.
        """
        if data is None:
            raise ValueError('Empty dataset is passed to filter ...')
        self.data = data.copy()
        self.variable = variable
        self.data['filtered'] = (['time', 'latitude', 'longitude'],
                                   self.filter_ds['land_mask'].values*self.data[variable].values)
        return self

    def get_filtered_data(self):
        """Returns the filtered data."""
        return self.data

    def _load(self):
        """ Sets the filter as a xarray dataset in the constructor.
        """
        filters = glob.glob( os.path.join( path_filter,
                                '*{}*.nc'.format(self.filter_key)))
        assert len(filters) == 1, 'Detected {} filters ... '.format(len(filters))
        filt = xr.open_dataset(filters[0])
        filt = xr.where(filt, 1.0, np.nan)
        self.filter_ds = filt

        return

    def get_mean(self):
        """ Sum all values and divide by the number of non-zero instances.

        Its safe to assume that only the filtered
        data is identically equal zero.
        """
        matrix = self.data['filtered'].values
        mean = np.nanmean(matrix)
        self.mean = mean
        return mean

    def get_spatial_mean(self):
        """ Sum all values and divide by the number of non-zero instances.

        Its safe to assume that only the filtered
        data is identically equal zero.
        """
        matrix = self.data['filtered'].values
        mean = np.nanmean(matrix, axis = (1, 2))

        self.mean = mean
        return mean

    def get_temporal_mean(self):
        """ Sum all values and divide by the number of non-zero instances.

        Its safe to assume that only the filtered
        data is identically equal zero.
        """
        matrix = self.data['filtered'].values
        mean = np.nanmean(matrix, axis = (1, 2))
        self.mean = mean
        return mean.copy()

    def quick_plot_filtered_data(self):
        """ Quick plot to se the region you are filtering.
        returns ax
        """
        return self.data['filtered'].mean('time').plot()

    def quick_plot_filter(self):
        """ Quick plot to se the region you are filtering.
        returns ax
        """
        return self.filter_ds['land_mask'].plot()
