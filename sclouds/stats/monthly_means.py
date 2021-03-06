import os
import glob

import xarray as xr
import numpy as np

read_dir   = '/home/hanna/lagrings/ERA5_monthly/'
save_dir   = '/home/hanna/lagrings/ERA5_stats/results/'
save_dir   = '/home/hanna/lagrings/results/stats/test/'
filter_dir = '/home/hanna/MS-suppl/filters/'

# for wessel -- /uio/lagringshotell/geofag/students/metos/hannasv/results
read_dir   = '/uio/lagringshotell/geofag/students/metos/hannasv/ERA5_monthly/'
save_dir   = '/uio/lagringshotell/geofag/students/metos/hannasv/results/stats/monthly_mean/'
filter_dir = '/uio/hume/student-u89/hannasv/MS-suppl/'

VARIABLES    = ['r', 'q', 't2m', 'sp', 'tcc']

# added duplicates since you are using enviornment on wessel
#from sclouds.helpers import merge
from filter import Filter
#from sclouds.io import Filter

def get_all_filesnames_from_one_variable(var):
    """Get all filenames from one variable."""
    return glob.glob(os.path.join(read_dir, '*{}*.nc'.format(var)))


def get_date_and_mean_from_one_filename(absolute_path = '/home/hanna/lagrings/ERA5_monthly/2012_01_t2m.nc'):
    """ Computes the mean over the entire domain, only land and only sea

    Parameteres
    ----------------
    absolute_path : str
        The absolute path of the file.

    Returns
    ----------------
    date : np.datetime64
        Date of this monthly average
    mean_all : float
        Mean over entire domain
    mean_land : float
        Mean over land
    mean_sea : float
        mean over sea
    """
    basename = os.path.basename(absolute_path)
    date     = np.datetime64('{}-{}'.format( basename[:4], basename[5:7]))
    var      = basename[8:].split('.')[0]
    # Generating all the data and filters.
    try:
        #print('Attemps to open file {}'.format(absolute_path))
        data     = xr.open_dataset(absolute_path) # read the data
        #print(data)
        f_land   = Filter('land').set_data(data = data, variable = var)
        f_sea    = Filter('sea').set_data(data = data, variable = var)

        mean_all  = np.nanmean(data[var].values)
        mean_land = f_land.get_mean()
        mean_sea  = f_sea.get_mean()
        return date, mean_all, mean_land, mean_sea
    except OSError:
        print("Didn't find file ... {}".format(absolute_path))
        return date, np.nan, np.nan, np.nan

storage = {}
for var in VARIABLES: # VARIABLES[:-1]

    alls  = []
    dates = []
    lands = []
    seas  = []

    files = get_all_filesnames_from_one_variable(var)
    print('Detected {} files.'.format(len(files)))
    for i, fil in enumerate(np.sort(files)):
        d, region, land, sea = get_date_and_mean_from_one_filename(fil)

        dates.append(d)
        alls.append(region)
        lands.append(land)
        seas.append(sea)

        if i%10 == 0:
            print('Number {}. Var {} '.format(i, var))

    storage[var] = alls
    storage['land_{}'.format(var)] = lands
    storage['sea_{}'.format(var)] = seas
    storage['date_{}'.format(var)] = dates # just to check that they are equal

data = xr.Dataset(storage)
data.to_netcdf(save_dir + 'monthly_means_updated_3.nc')
print('Computet monthly means ... ')
