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
n_rows = 3
n_cols = 1

#data = xr.open_dataset('/home/hanna/lagrings/results/stats/monthly_means.nc')


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

fig, axes =  plt.subplots(nrows = n_rows, ncols = n_cols, sharex=True)
fig.set_size_inches(w = TEXT_WIDTH_IN, h = TEXT_HEIGHT_IN - 3)

mapping_number_month = {'01':'Jan', '02':'Feb', '03':'Mar', '04':'Apr',
                        '05':'May', '06':'Jun', '07':'Jul', '08':'Aug',
                        '09':'Sep', '10':'Oct', '11':'Nov', '12':'Dec'}


for i in range(1, 13):
    start = '2012-{:02d}-01'.format(i)
    stop  = '2012-{:02d}-07'.format(i)

    df = filter_data_in_period(start = start, stop  = stop)

    data = df
    date = df.time.values
    date = np.arange(len(date))

    var = 'tcc'
    vals   = data[var].values
    f_land = data['land_{}'.format(var)].values
    f_sea  = data['sea_{}'.format(var)].values

    axes[0].set_title('No filter', fontsize = 14)
    axes[0].plot(date, vals, label = mapping_number_month["{:02d}".format(i)])
    #axes[0].legend(ncol = 6, frameon = False, bbox_to_anchor=(0.18, 1.2))

    axes[1].set_title('Land', fontsize = 14)
    axes[1].plot(date, f_land, label = mapping_number_month["{:02d}".format(i)])

    axes[2].set_title('Sea', fontsize = 14)
    axes[2].plot(date, f_sea, label = mapping_number_month["{:02d}".format(i)])

    if var != 'tcc':
        lab = '{} [{}]'.format(var, UNITS[var])
    else:
        lab = 'cfc [1]'
        
    axes[0].set_ylabel(lab)
    axes[1].set_ylabel(lab)
    axes[2].set_ylabel(lab)

    axes[0].set_ylim([0, 1])
    axes[1].set_ylim([0, 1])
    axes[2].set_ylim([0, 1])

    labels = ['{:02d}'.format(date, i) for date in range(1, 9)]

    axes[2].set_xticks(np.linspace(0, len(date), len(labels)))
    axes[2].set_xticklabels( labels )
plt.legend(ncol = 6, frameon = False, bbox_to_anchor=(0.99, -0.25))
plt.subplots_adjust(wspace = 0.2, hspace = 0.3, top=0.9, bottom=0.2, left = 0.14, right = .95)
plt.savefig(path_python_figures + 'spatially_averaged_one_week_tcc_seperated_by_filters.png'.format(start))
