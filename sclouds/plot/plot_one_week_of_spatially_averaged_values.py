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

#data = xr.open_dataset('/home/hanna/lagrings/results/stats/monthly_means.nc')

#fig, axes =  plt.subplots(nrows = n_rows, ncols = n_cols, sharex=True, sharey=False)
#fig.set_size_inches(w = TEXT_WIDTH_IN, h = TEXT_HEIGHT_IN - 1)
#plt.subplots_adjust(hspace = 0.2, top=0.97, bottom=0.03, left = 0.14, right = 0.97)

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
    date = np.arange(len(date))

    fig, axes =  plt.subplots(nrows = n_rows, ncols = n_cols, sharex = True)
    fig.set_size_inches(w = TEXT_WIDTH_IN, h = TEXT_HEIGHT_IN - 3)

    for var, ax in zip(VARIABLES, axes):
        vals   = data[var].values
        f_land = data['land_{}'.format(var)].values
        f_sea  = data['sea_{}'.format(var)].values

        ax.set_title(LONGNAME[var], fontsize = 14)
        ax.plot(date, vals, label = '{}'.format('no filter'))
        ax.plot(date, f_land, label = '{}'.format('land'))
        ax.plot(date, f_sea, label = '{}'.format('sea'))
        if var != 'tcc':
            lab = '{} [{}]'.format(var, UNITS[var])
        else:
            lab = 'cfc [1]'

        ax.set_ylabel(lab)
        labels = ['{:02d}-{:02d}'.format(date, i) for date in range(1, 9)]

        ax.set_xticks(np.linspace(0, len(vals), len(labels)))
        ax.set_xticklabels( labels )

    plt.legend(ncol = 3, frameon = False, bbox_to_anchor=(0.8, -0.25))
    #plt.subplots_adjust(wspace = 0.2, hspace = 0.3, top=0.9, bottom=0.1, left = 0.14, right = .95)
    #plt.subplots_adjust(wspace = 0.3, hspace = 0.2, top= 0.95, bottom= 0.1, left= 0.15, right= 0.97)
    plt.subplots_adjust(wspace = 0.2, hspace = 0.3, top=0.9, bottom=0.1, left = 0.14, right = .95)
    plt.savefig(path_python_figures + 'spatially_averaged_one_week_from_{}.png'.format(start))
