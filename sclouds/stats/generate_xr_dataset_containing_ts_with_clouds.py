import os
import glob

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

def get_list_of_regridded_files():
    """ Returns list of files which is regridded.
    """
    repo = '/home/hanna/miphclac/' # TODO update this to be /home/hanna/lagrings/ERA%_monthly
    all_paths = []

    for y in np.arange(2004, 2019):
        for m in np.arange(1, 13):
            files = glob.glob(os.path.join( repo,'{}_{:02d}/*tcc*.nc'.format(y, m)))
            if len(files) > 0:
                print(files)
                all_paths.append(files[0])
    return all_paths

def get_ts_in_regridded_files():
    """ Gets list of timestamps available in the regridded files.
    """
    all_paths = get_list_of_regridded_files()
    ts = []
    for fil in all_paths:
        data = xr.open_dataset(fil)
        ts.append(data.time.values)
    return np.concatenate(ts)

def generate_xarray_file_containing_the_timestamps():
    """ Regenerate the netcdf file containing the relevant data.
    """
    ts = get_ts_in_regridded_files()
    #cloud_fraction, nans = compute(fil, lats, lons)
    ds = xr.Dataset({'fake_data':np.ones(len(ts))})
    ds['time'] = ts

    # Add time as a coordinate and dimension.
    ds = ds.assign_coords(time = ds.time)
    #ds = ds.expand_dims(dim = 'time')
    ds.to_netcdf('/home/hanna/lagrings/results/stats/timstamps_available_cloud_fractions.nc')
    return

generate_xarray_file_containing_the_timestamps()
