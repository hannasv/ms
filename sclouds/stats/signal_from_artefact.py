""" Store the signal from the artefact at every timestep and then use this to plot
    bar plot of occurences.
"""



import os
import glob

import xarray as xr
import numpy as np

read_dir   = '/home/hanna/lagrings/ERA5_monthly/'
save_dir   = '/home/hanna/lagrings/results/'
filter_dir = '/home/hanna/MS-suppl/filters/'

# for wessel -- /uio/lagringshotell/geofag/students/metos/hannasv/results
#read_dir   = '/uio/lagringshotell/geofag/students/metos/hannasv/ERA5_monthly/'
#save_dir   = '/uio/lagringshotell/geofag/students/metos/hannasv/results/'
#filter_dir = '/uio/hume/student-u89/hannasv/MS-suppl/'

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
    return xr.open_mfdataset(files, compat='no_conflicts', join='outer')

from sclouds.io import Filter


filter_key = 'artefact'
stat = 'mean'
var = 'tcc'


files =glob.glob(os.path.join(read_dir, '*tcc*.nc'))
print('Merges files {}'.format(files))

data = merge(files)

filter = Filter(filter_key).set_data(data, var)
means = filter.get_spatial_mean()
print(type(means))
means.to_netcdf( path = os.path.join(save_dir, 'temporal_signal_artefact.nc' ))
print(means)
