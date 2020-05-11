import os
import glob

import xarray as xr
import numpy as np

#read_dir   = '/home/hanna/lagrings/ERA5_monthly/'
#save_dir   = '/home/hanna/lagrings/results/stats/season/'
#filter_dir = '/home/hanna/MS-suppl/filters/'

# for wessel -- /uio/lagringshotell/geofag/students/metos/hannasv/results
read_dir   = '/uio/lagringshotell/geofag/students/metos/hannasv/ERA5_monthly/'
save_dir   = '/uio/lagringshotell/geofag/students/metos/hannasv/results/stats/season/'
filter_dir = '/uio/hume/student-u89/hannasv/MS-suppl/'

STATS         = ['mean', 'std', 'min', 'max', 'median'] # 'median',
VALID_VARS    = ['r', 'q', 't2m', 'sp', 'tcc']
VALID_FILTERS = ['coast', 'sea', 'land', 'artefact']
SEASONS = ['DJF', 'MAM', 'JJA', 'SON']
# TODO : Round two only compute the properties for times you have cloud data.

# added duplicates since you are using enviornment on wessel
#from sclouds.helpers import merge
#from filter import Filter
#from sclouds.io import Filter


def get_list_of_files_for_season(season, var):
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
    if season not in ['DJF', 'MAM', 'JJA', 'SON']:
        raise ValueError('{} is not a valied season. Use {}'.format(season,
                            ['DJF', 'MAM', 'JJA', 'SON']))
    if season == 'DJF':
        f1 = glob.glob(os.path.join( read_dir, '*_12_*{}.nc'.format(var)))
        f2 = glob.glob(os.path.join( read_dir, '*_01_*{}.nc'.format(var)))
        f3 = glob.glob(os.path.join( read_dir, '*_02_*{}.nc'.format(var)))
        subset = f1+f2+f3

    if season == 'MAM':
        f1 = glob.glob(os.path.join( read_dir, '*_03_*{}.nc'.format(var)))
        f2 = glob.glob(os.path.join( read_dir, '*_04_*{}.nc'.format(var)))
        f3 = glob.glob(os.path.join( read_dir, '*_05_*{}.nc'.format(var)))
        subset = f1+f2+f3

    if season == 'JJA':
        f1 = glob.glob(os.path.join( read_dir, '*_06_*{}.nc'.format(var)))
        f2 = glob.glob(os.path.join( read_dir, '*_07_*{}.nc'.format(var)))
        f3 = glob.glob(os.path.join( read_dir, '*_08_*{}.nc'.format(var)))
        subset = f1+f2+f3

    if season == 'SON':
        f1 = glob.glob(os.path.join( read_dir, '*_09_*{}.nc'.format(var) ))
        f2 = glob.glob(os.path.join( read_dir, '*_10_*{}.nc'.format(var) ))
        f3 = glob.glob(os.path.join( read_dir, '*_11_*{}.nc'.format(var) ))
        subset = f1+f2+f3

    return subset



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


class SeasonalStats:
    """
    Filtering only makes sense for global stats!
    """

    def __init__(self, season, var = None, variable = None, dataset = None,
                        filter_key = 'all', local = True):

        #if not filter_key in VALID_FILTERS:
        #    raise ValueError('Provided filter {} is not valid. Available filters are {}.'.format(filter_key, VALID_FILTERS))

        self.filter_key = filter_key # used in filename
        self.var = var
        self.season = season

        if variable is not None:
            #if not variable in VALID_VARS:
            #    raise ValueError('Provided variable {} is not valid. Available variables are {}.'.format(variable, VALID_VARS))

            files = get_list_of_files_for_season(season, variable)

            print('Found {} files ..'.format(files))
            print("Test example using three files ... ")

            self.dataset      = merge(files) # TODO : only select hours you have cloud data
            print('finished merging files')
            self.variable     = variable # assert not valid variable and filter
            self.filter_key   = filter_key

        if dataset is not None:
            print('sets dataset')
            self.dataset = dataset

        #if variable is None and dataset is None:
        #    raise ValueError('Please provide either a variable or a dataset.')
        generated_files = glob.glob(os.path.join(save_dir, '*.nc'))

        if not self.generate_pixel_output_filename() in generated_files and local:
            print('Starting to produce local results.')
            self.result         = self.produce_results() # store all results as variables in a dataset.
            self.save_pixel()

        if not self.generate_global_output_filename() in generated_files:
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
            #result = eval("self.dataset.{}(dim = 'time')".format(statistics))[self.variable].values
                #result = self.dataset[self.variable].values
            print('result shape {}'.format(np.shape(result)))
            result[result == 0] = np.nan
            print('Shape after stat {}'.format(np.shape(result)))
            computing = eval("np.nan{}(result, axis = 0)".format(statistics))
            #computing = eval("np.{}(no_zeros, axis = 0)".format(statistics))
            print('Computing shape {}'.format(np.shape(result)))
            res_dict[statistics] = (dimensions, computing)

        res = np.nanmedian((result - res_dict['mean'][1]), axis = 0)
        #.median(dim = 'time')[self.variable].values
        res_dict['mad'] = (dimensions, res)

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

    def merge_files(self, files):
        print('Merging {} files ...'.format(len(files)))
        datasets = [xr.open_dataset(fil) for fil in files]
        print('Finished reading data')
        return xr.merge(datasets)

    def get_data(self):
        return self.dataset.copy()

    def generate_pixel_output_filename(self):
        """Generates output filename for stats files."""
        return os.path.join(save_dir,
        'stats_pixel_{}_{}_{}.nc'.format(self.season, self.var, self.filter_key))

    def generate_global_output_filename(self):
        return os.path.join(save_dir,
        'stats_global_{}_{}_{}.nc'.format(self.season,self.var, self.filter_key))

    def save_pixel(self):
        print("Saving file for season {}, variable {}, filter {}".format(self.season,self.var, self.filter_key))
        self.result.to_netcdf(os.path.join(save_dir,
            'stats_pixel_{}_{}_{}.nc'.format(self.season, self.var, self.filter_key)))
        return

    def save_global(self):
        self.global_result.to_netcdf(os.path.join(save_dir,
                                'stats_global_{}_{}_{}.nc'.format(self.season,
                                                self.var, self.filter_key)))
        return self

if __name__ == "__main__":
    # Generate the satelite data below here.
    for season in SEASONS:
        for var in VALID_VARS:
            stat = SeasonalStats(season = season,var = var, variable=var)
            #data = stat.get_data()
            #stat.save()
            """
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
                           filter_key = key, local = False)#.save()
                #st.set_data(filtered_data).save()
            """
