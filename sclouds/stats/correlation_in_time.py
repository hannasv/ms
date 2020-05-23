""" Generate correlation data ....
"""

import os
import glob

import xarray as xr
import numpy as np


from utils import (dataset_to_numpy, dataset_to_numpy_order,
                              dataset_to_numpy_grid_order,
                              dataset_to_numpy_grid,
                              get_xarray_dataset_for_period,
                              dataset_to_numpy_order_traditional_ar,
                              dataset_to_numpy_order_traditional_ar_grid,
                              get_pixel_from_ds)


# for wessel -- /uio/lagringshotell/geofag/students/metos/hannasv/results
read_dir   = '/uio/lagringshotell/geofag/students/metos/hannasv/ERA5_monthly/'
save_dir   = '/uio/lagringshotell/geofag/students/metos/hannasv/results/stats/monthly_mean/'
filter_dir = '/uio/hume/student-u89/hannasv/MS-suppl/'

# added duplicates since you are using enviornment on wessel
#from sclouds.helpers import merge
from filter import Filter
files = glob.glob(os.path.join(read_dir, '*.nc'))

# This should read all files....
import numpy as np

data = xr.open_mfdataset(files, compat='no_conflicts')
ref_data = data['tcc'].values

longitude = data.longitude.values
latitude  = data.latitude.values

storang = np.zeros((81, 161, 1))
dictionary_to_store = {}
var = 'tcc'

order = 1
k = order
for i, lat in enumerate(latitude):
    for j, lon in enumerate(longitude):
        ds     = get_pixel_from_ds(data, lat, lon)
        X, y = dataset_to_numpy_order_traditional_ar(ds, k, False)

        a = np.concatenate( [X, y], axis = 1)
        a = a[~np.isnan(a).any(axis = 1)]
        print(a.shape)
        storang[i,j,0] = np.corrcoef(a[:, 0], a[:, 1])[0][1]
        print('finished {}/{}'.format((i+1)*j,  81*161))

lon = data.longitude.values
lat = data.latitude.values
dictionary_to_store['order{}'.format(k)] = (['latitude', 'longitude'], storang[:, :, 0])

result = xr.Dataset(dictionary_to_store,
                    coords={'longitude': (['longitude'], lon),
                            'latitude': (['latitude'], lat),
                            })

result.to_netcdf(os.path.join(save_dir, 'correlation_in_time_order{}.nc'.format(k)))
