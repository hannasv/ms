""" Generate correlation data ....
"""

import os
import glob

import xarray as xr
import numpy as np

#read_dir   = '/home/hanna/lagrings/ERA5_monthly/'
#save_dir   = '/home/hanna/lagrings/ERA5_stats/results/'
#save_dir   = '/home/hanna/lagrings/results/stats/monthly_mean/'
#filter_dir = '/home/hanna/MS-suppl/filters/'

# for wessel -- /uio/lagringshotell/geofag/students/metos/hannasv/results
read_dir   = '/uio/lagringshotell/geofag/students/metos/hannasv/ERA5_monthly/'
save_dir   = '/uio/lagringshotell/geofag/students/metos/hannasv/results/stats/monthly_mean/'
filter_dir = '/uio/hume/student-u89/hannasv/MS-suppl/'

# added duplicates since you are using enviornment on wessel
#from sclouds.helpers import merge
from filter import Filter
#from sclouds.io import Filter

files = glob.glob(os.path.join(read_dir, '*.nc'))

# This should read all files....


data = xr.open_mfdataset(files, compat='no_conflicts')
ref_data = data['tcc'].values

storang = np.zeros((81, 161, 4))

dictionary_to_store = {}

for k, var in enumerate(['r', 'q', 't2m', 'sp']):
    dta = data[var].values
    for i in range(81):
        for j in range(161):
            storang[i,j,k] = np.corrcoef(ref_data[:, i, j], dta[:, i, j])[0][1]

    dictionary_to_store[var] = storang[:, :, k]

    lon = data.longitude.values
    lat = data.latitude.values

    result = xr.Dataset(dictionary_to_store,
                        coords={'longitude': (['longitude'], lon),
                                'latitude': (['latitude'], lat),
                                })

    result.to_netcdf(os.path.join(save_dir, 'correlation_{}.nc'.format(var)))
