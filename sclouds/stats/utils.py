import os
import glob
import xarray as xr

from sclouds.helpers import path_stats_results

def filter_era5_based_on_ts_in_cloud_cover(ds_era):
    """ Based on the period available in era5

    """
    filename = os.path.join(path_stats_results, 'timstamps_available_cloud_fractions.nc')
    ds_tcc = xr.open_dataset(filename)

    # merge to only include the intersection
    ds = xr.merge([ds_era, ds_tcc], dim = 'time')
    ds = None
    return ds
