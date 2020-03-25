import os
import glob

import xarray as xr
import numpy as np

read_dir   = '/home/hanna/lagrings/ERA5_monthly/'
save_dir   = '/home/hanna/lagrings/ERA5_stats/results/'
filter_dir = '/home/hanna/MS-suppl/'

STATS         = ['mean', 'median', 'std', 'min', 'max']
VALID_VARS    = ['tcc', 'r', 'q', 't2m', 'sp']
VALID_FILTERS = ['coast', 'sea', 'land', 'artefact', 'all']

# TODO : Round two only compute the properties for times you have cloud data.

class Stats:
    """
    Filtering only makes sense for global stats!
    """

    def __init__(self, variable, filter_key = 'all'):

        if not variable in VALID_VARS:
            raise ValueError('Provided variable {} is not valid. Available variables are {}.'.format(variable, VALID_VARS))

        if not filter_key in VALID_FILTERS:
            raise ValueError('Provided filter {} is not valid. Available filters are {}.'.format(variable, VALID_VARS))

        self.variable     = variable # assert not valid variable and filter
        self.filter_key   = filter_key

        search_str        = '*{}.nc'.format(variable)
        files             = glob.glob(os.path.join(read_dir, search_str))

        self.dataset        = self.merge_files(files) # TODO : only select hours you have cloud data.
        self.result         = self.produce_results() # store all results as variables in a dataset.
        self.global_result = self.produce_global_results()
        return

    def produce_results(self):
        """ Results """
        #assert self.global_stats, " Only valued for pixel.."
        dimensions = ['latitude', 'longitude']
        res_dict = {}
        lon = self.dataset.longitude.values
        lat = self.dataset.latitude.values

        for statistics in STATS:
            result = eval("self.dataset.{}(dim = 'time')".format(statistics))[self.variable].values
            res_dict[statistics] = (dimensions, result)

        res = (self.dataset - self.dataset.mean(dim = 'time')).median(dim = 'time')[self.variable].values
        res_dict['MAD'] = (dimensions, res)

        result = xr.Dataset(res_dict,
                            coords={'longitude': (['longitude'], lon),
                                    'latitude': (['latitude'], lat),
                                    })
        return result

    def produce_global_results(self):
        """ Results """
        #assert not self.global_stats, " Only valued for pixel.."

        res_dict = {}

        for statistics in STATS:
            result = eval("self.dataset.{}()".format(statistics))[self.variable].values
            res_dict[statistics] = result

        res = (self.dataset - self.dataset.mean()).median()[self.variable].values
        res_dict['MAD'] = res

        result = xr.Dataset(res_dict)
        return result


    def set_filter(self):
        """ Sets the filter as a xarray dataset in the constructor. """
        filters = glob.glob( os.path.join( filter_dir, '*{}*.nc'.format(self.filter_key)))
        assert len(filters) == 1, 'Detected multiple filters ... '
        filt = xr.open_dataset(filters[0])
        self.dataset['filtered'] = filt['land_mask'].values*self.dataset[self.variable].values
        return

    def merge_files(self, files):
        print('Merging {} files ...'.format(len(files)))
        datasets = [xr.open_dataset(fil) for fil in files]
        return xr.merge(datasets)

    def generate_output_filename(self):
        """Generates output filename for stats files."""
        return os.path.join(save_dir, 'stats_{}_{}.nc'.format(self.variable, self.filter_key))

    def save(self):
        print("Saving file for variable {}, filter {}".format(self.variable, self.filter_key))
        self.result.to_netcdf(os.path.join(save_dir, 'stats_pixel_{}_{}.nc'.format(self.variable, self.filter_key)))
        self.global_result.to_netcdf(os.path.join(save_dir, 'stats_global_{}_{}.nc'.format(self.variable, self.filter_key)))
        return

if __name__ == "__main__":
    # Generate the satelite data below here.
    for var in VALID_VARS:
        #for filter in VALID_FILTERS:
        Stats(variable=var, filter_key='all').save()
