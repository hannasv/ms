import os
import glob

import xarray as xr
import numpy as np


# for wessel -- /uio/lagringshotell/geofag/students/metos/hannasv/results
read_dir   = '/uio/lagringshotell/geofag/students/metos/hannasv/ERA5_monthly/'
path_input = read_dir
save_dir   = '/uio/lagringshotell/geofag/students/metos/hannasv/results/stats/'
filter_dir = '/uio/hume/student-u89/hannasv/MS-suppl/'

# for wessel -- /uio/lagringshotell/geofag/students/metos/hannasv/results
#read_dir   = '/uio/lagringshotell/geofag/students/metos/hannasv/ERA5_monthly/'
#save_dir   = '/uio/lagringshotell/geofag/students/metos/hannasv/results/stats/'
#filter_dir = '/uio/hume/student-u89/hannasv/MS-suppl/'

STATS         = ['mean', 'std', 'min', 'max', 'median'] # 'median',
VALID_VARS    = ['r', 'q', 't2m', 'sp', 'tcc']
VALID_FILTERS = ['coast', 'sea', 'land', 'artefact']

# TODO : Round two only compute the properties for times you have cloud data.

# added duplicates since you are using enviornment on wessel
#from sclouds.helpers import merge
from filter import Filter
#from sclouds.io import Filter

def merge(files):
    """ Merging a list of filenames into a dataset.open_mfdataset

    Parameteres
    -----------
    files : List[str]
        List of abolute paths to files.

    Returns
    ------------
     _ : xr.dataset
        Merged files into one dataset.
    """
    #assert len(files) == 5
    #datasets = [xr.open_dataset(fil) for fil in files]
    #return xr.merge(datasets)
    return xr.open_mfdataset(files, compat='no_conflicts')

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

class Stats:
    """
    Filtering only makes sense for global stats!
    """

    def __init__(self, var = None, variable = None, dataset = None,
                    filter_key = 'all', local = True, start = None, stop = None,
                    global_stat = True):

        #if not filter_key in VALID_FILTERS:
        #    raise ValueError('Provided filter {} is not valid. Available filters are {}.'.format(filter_key, VALID_FILTERS))
        self.reserve_dir = '/uio/hume/student-u89/hannasv/reserve_results/'
        self.save_dir    = '/uio/lagringshotell/geofag/students/metos/hannasv/results/stats/'
        print('enters')
        if start is not None and stop is not None:
            self.save_dir = os.path.join(self.save_dir,'{}_{}'.format(start, stop))
            print('updated self save dir {}'.format(self.save_dir))
        self.filter_key = filter_key # used in filename
        self.var = var
        self.start = start
        self.stop = stop
        self.global_stat = global_stat
        if variable is not None:
            #if not variable in VALID_VARS:
            #    raise ValueError('Provided variable {} is not valid. Available variables are {}.'.format(variable, VALID_VARS))
            if start is None and stop is None:
                search_str = '*{}.nc'.format(var)
                print('Searching for {}'.format(read_dir+search_str))
                files             = glob.glob(os.path.join(read_dir, search_str))
                # files = files[:3]
                print('Found {} files ..'.format(len(files)))
                print("Test example using three files ... ")
            else:
                files = get_list_of_files_excluding_period(start = start, stop = stop)

            self.dataset      = merge(files) # TODO : only select hours you have cloud data
            print('finished merging files')
            self.variable     = variable # assert not valid variable and filter
            self.filter_key   = filter_key

        if dataset is not None:
            print('sets dataset')
            self.dataset = dataset

        #if variable is None and dataset is None:
        #    raise ValueError('Please provide either a variable or a dataset.')

        generated_files = glob.glob(os.path.join(self.save_dir, '*.nc'))
        #print('Number of already generated files {}'.format(len(generated_files)))
        if not self.generate_pixel_output_filename() in generated_files and local and filter_key == 'all':
            print('Starting to produce local results. for var {}'.format(var))
            self.result         = self.produce_results() # store all results as variables in a dataset.
            print('about to save pixel')
            self.save_pixel()

        if not self.generate_global_output_filename() in generated_files and global_stat:
            print('Starting to produce global results.')
            self.global_result  = self.produce_global_results()
            self.save_global()
        return


    def produce_results(self):
        """ Results """
        #assert self.global_stats, " Only valued for pixel.."
        print('\n Producing results ... ')
        dimensions = ['latitude', 'longitude']
        res_dict = {}
        lon = self.dataset.longitude.values
        lat = self.dataset.latitude.values

        result = self.dataset[self.variable].values
        for statistics in STATS:
            print('for stat {}'.format(statistics))
            #result = eval("self.dataset.{}(dim = 'time')".format(statistics))[self.variable].values
                #result = self.dataset[self.variable].values
            #print('result shape {}'.format(np.shape(result)))
            result[result == 0] = np.nan
            #print('Shape after stat {}'.format(np.shape(result)))
            computing = eval("np.nan{}(result, axis = 0)".format(statistics))
            #computing = eval("np.{}(no_zeros, axis = 0)".format(statistics))
            #
            print('Computing shape {}'.format(np.shape(result)))
            res_dict[statistics] = (dimensions, computing)
        print('Detected mad started computing')
        res = np.nanmedian(result - res_dict['mean'][1], axis = 0)
        res_dict['mad'] = (dimensions, res)
        print('Updates the netcdf dataset')
        result = xr.Dataset(res_dict,
                            coords={'longitude': (['longitude'], lon),
                                    'latitude': (['latitude'], lat),
                                    })
        return result

    def produce_global_results(self):
        """ Results """
        #assert not self.global_stats, " Only valued for pixel.."
        print('\n Producing global results, var = {}'.format(self.variable))
        res_dict = {}
        result = self.dataset[self.variable].values

        for statistics in STATS:
            #result = eval("self.dataset.{}()".format(statistics))[self.variable].values
            print('result shape {}'.format(np.shape(result)))
            result[result == 0] = np.nan
            #no_zeros = np.argwhere(result, 0, np.nan)
            print('Shape after stat {}'.format(np.shape(result)))
            computing = eval("np.nan{}(result)".format(statistics))
            print(computing)

            print('Computing shape {}'.format(np.shape(computing)))
            res_dict[statistics] = computing

        res = np.nanmedian((result - res_dict['mean']))
        res_dict['mad'] = res

        result = xr.Dataset(res_dict)
        return result

    def get_data(self):
        return self.dataset.copy()

    def generate_pixel_output_filename(self):
        """Generates output filename for stats files."""
        return os.path.join(self.save_dir, 'stats_pixel_{}_{}.nc'.format(self.var, self.filter_key))

    def generate_global_output_filename(self):
        return os.path.join(self.save_dir, 'stats_global_{}_{}.nc'.format(self.var, self.filter_key))

    def save_pixel(self):
        print("Saving file for variable {}, filter {}".format(self.var, self.filter_key))

        try:
            self.result.to_netcdf(os.path.join(self.save_dir,
                    'stats_pixel_{}_{}.nc'.format(self.var, self.filter_key)))
        except PermissionError:
            self.result.to_netcdf(os.path.join(self.reserve_dir,
                    'stats_pixel_{}_{}_{}_{}.nc'.format(self.var, self.filter_key, self.start, self.stop)))
        return

    def save_global(self):
        print('atrtemps to save')
        try:
            self.global_result.to_netcdf(os.path.join(self.save_dir,
                    'stats_global_{}_{}.nc'.format(self.var, self.filter_key)))
        except PermissionError:
            self.global_result.to_netcdf(os.path.join(self.reserve_dir,
                    'stats_global_{}_{}_{}_{}.nc'.format(self.var, self.filter_key, self.start, self.stop)))
        return self

def compute_stats_entire_period():
    """The statistics in this thesis - redo this one for clouds"""
    for var in VALID_VARS:
        stat = Stats(var = var, variable=var)
        data = stat.get_data()
        #stat.save()

        for key in VALID_FILTERS:
            print('Applying filter {}'.format(key))
            filter = Filter(key).set_data(data, var)
            print('Generated filter')
            filtered_data = filter.get_filtered_data()
            print('Recieved data from filter ')
            print(filtered_data)
            st = Stats(var = var,
                       variable  = 'filtered',
                       dataset= filtered_data,
                       filter_key = key, local = True)#.save()
    return

def compute_stats_for_normalization():
    pass

if __name__ == "__main__":
     # TODO run everything again including cloud cover
    # Generate the satelite data below here.
    for var in ['tcc']: #
        for start, stop in [('2004-04-01', '2008-12-31'),
                            ('2009-01-01', '2013-12-31'),
                            ('2014-01-01', '2018-12-31')]:
            # Computes stat by excluding the above period.
            stat = Stats(var = var, variable=var,
                        local = False, start = start, stop = stop,
                        filter_key = 'all', global_stat = True)


    for var in ['r', 'q', 't2m', 'sp']: #
        for start, stop in [('2004-04-01', '2008-12-31'),
                            ('2009-01-01', '2013-12-31'),
                            ('2014-01-01', '2018-12-31')]:
            # Computes stat by excluding the above period.
            stat = Stats(var = var, variable=var,
                        local = True, start = start, stop = stop,
                        filter_key = 'all', global_stat = False)

        """
        for key in VALID_FILTERS:
            stat = Stats(var = var, variable=var)
            data = stat.get_data()

            for key in VALID_FILTERS:
                print('Applying filter {}'.format(key))
                filter = Filter(key).set_data(data, var)
                print('Generated filter')
                filtered_data = filter.get_filtered_data()
                print('Recieved data from filter ')
                print(filtered_data)
                st = Stats(var = var,
                           variable  = 'filtered',
                           dataset= filtered_data,
                           filter_key = key, local = False)
        """
