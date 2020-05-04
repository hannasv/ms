import os
import glob

import xarray as xr
import numpy as np

path_input = '/uio/lagringshotell/geofag/students/metos/hannasv/ERA5_monthly/'
path_filter = '/uio/lagringshotell/geofag/students/metos/hannasv/results/stats/'

VARIABLES  = ['tcc', 'r', 'q', 't2m', 'sp']
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
        self.data = data.copy()
        self.variable = variable
        self.data['filtered'] = (['time', 'latitude', 'longitude'],
                                np.flipud(self.filter_ds['land_mask'].values)*
                                        self.data[variable].values)
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
        self.filter_ds = filt
        return

    def get_mean(self):
        """ Sum all values and divide by the number of non-zero instances.

        Its safe to assume that only the filtered
        data is identically equal zero.
        """
        matrix = self.data['filtered'].values
        mean = np.true_divide(matrix.sum(),(matrix!=0).sum())
        self.mean = mean
        return mean

    def get_spatial_mean(self):
        """ Sum all values and divide by the number of non-zero instances.

        Its safe to assume that only the filtered
        data is identically equal zero.
        """
        matrix = self.data['filtered']
        mean = np.true_divide(matrix.sum(['latitude', 'longitude']),
                                (matrix!=0).sum(['latitude', 'longitude']))
        self.mean = mean
        return mean

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
