""" Plotting routine used to plot subplots of spatially averages monthly means
and filtered by land sea and both.
"""
import os
import glob

import numpy as np
import xarray as xr

from sclouds.helpers import (path_input, path_stats_results, VARIABLES,
                                UNITS, LONGNAME)
from sclouds.plot.helpers import (TEXT_WIDTH_IN, TEXT_HEIGHT_IN,
                                    path_python_figures, import_matplotlib)
n_rows = len(VARIABLES)
n_cols = 1

data = xr.open_dataset('/home/hanna/lagrings/results/stats/monthly_means.nc')

fig, axes =  plt.subplots(nrows = n_rows, ncols = n_cols, sharex=True, sharey=False)
fig.set_size_inches(w = TEXT_WIDTH_IN, h = TEXT_HEIGHT_IN - 1)
plt.subplots_adjust(hspace = 0.2, top=0.97, bottom=0.03, left = 0.14, right = 0.97)

mat = import_matplotlib()
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def filter_data_in_period(start = '2012-07-01', stop  = '2012-07-07'):
    """ Add a land and sea filter of all varibles to existing data frame.

    Parameters
    --------------
    data : xr.Dataset
        Containes for data you want to filter.

    Returns
    --------------
    data : xr.Dataset
        Spatially averaged filtered timeseries.
    """
    from sclouds.helpers import VARIABLES
    from sclouds.io.filter import Filter
    from sclouds.io.utils import get_xarray_dataset_for_period

    data = get_xarray_dataset_for_period(start = start, stop = stop)

    df_return = data.mean(['latitude', 'longitude']).copy()

    for var in VARIABLES:
        fs = Filter('sea').set_data(data, variable = var)
        fl = Filter('land').set_data(data, variable = var)

        df_return['land_{}'.format(var)] = fl.get_spatial_mean()
        df_return['sea_{}'.format(var)]  = fs.get_spatial_mean()

    return df_return


for i in range(1, 13):
    start = '2012-{:02d}-01'.format(i)
    stop  = '2012-{:02d}-07'.format(i)

    df = filter_data_in_period(start = start, stop  = stop)

    n_rows = len(VARIABLES)
    n_cols = 1

    data = df
    date = df.time.values

    fig, axes =  plt.subplots(nrows = n_rows, ncols = n_cols)
    fig.set_size_inches(w = TEXT_WIDTH_IN, h = TEXT_HEIGHT_IN - 1)
    fig.autofmt_xdate()

    plt.subplots_adjust(hspace = 0.5, top=0.97, bottom=0.1, left = 0.14, right = 0.97)

    print('Warning no requirement for data to compute average from, refer to the ')

    for var, ax in zip(VARIABLES, axes):
        #if var != 'tcc':
        vals   = data[var].values
        f_land = data['land_{}'.format(var)].values
        f_sea  = data['sea_{}'.format(var)].values
        #date   = data['date_{}'.format(var)].values

        ax.set_title(LONGNAME[var], fontsize = 14)
        ax.plot(date, vals, label = '{}'.format('both'))
        ax.plot(date, f_land, label = '{}'.format('land'))
        ax.plot(date, f_sea, label = '{}'.format('sea'))

        ax.set_ylabel('{} [{}]'.format(var, UNITS[var]))
        ax.legend()
        # rotate and align the tick labels so they look better


        # use a more precise date string for the x axis locations in the
        # toolbar
        ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
        #ax.set_title('fig.autofmt_xdate fixes the labels')

    plt.savefig(path_python_figures + 'spatially_averaged_one_week_from_{}.png'.format(start))
