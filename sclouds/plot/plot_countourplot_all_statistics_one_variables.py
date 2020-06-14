""" Plotting routine used to plot subplots of spatially averages monthly means
and filtered by land sea and both.
"""
import numpy as np
import xarray as xr
import seaborn as sns
import glob
import os

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

from sclouds.helpers import (path_input, path_stats_results, VARIABLES,
                                UNITS, LONGNAME, STATISTICS, LONGNAME_STATISTICS)
from sclouds.io.utils import get_xarray_dataset_for_period
from sclouds.plot.helpers import (TEXT_WIDTH_IN, TEXT_HEIGHT_IN,
                                    path_python_figures, import_matplotlib,
                                    cmap_contour_plot, levels_contourplot,
                                    color_maps, add_ticks)
mat = import_matplotlib() #for mye
import matplotlib.pyplot as plt

n_rows = len(STATISTICS)
n_cols = 1
levels_contourplot = 100
path_stats_results = '/home/hanna/lagrings/results/stats'

for var in VARIABLES: #['mean']:#STATISTICS:
    fig, axes =  plt.subplots(nrows = n_rows, ncols = n_cols, sharex=True, sharey=False)
    fig.set_size_inches(w = TEXT_WIDTH_IN, h = TEXT_HEIGHT_IN - 2)
    #fig.suptitle(var, fontsize = 16)

    for stat, ax in zip(STATISTICS, axes):
        #if var != 'tcc':
        #print('Warning this duplicates the RH in plot for tcc')
        files = glob.glob(path_stats_results + '/*pixel*{}*all.nc'.format(var))

        if len(files) != 1:
            print(files)

        data = xr.open_dataset(files[0])

        vals = data[stat].values

        if var != 'tcc':
            vals   = np.flipud(vals)

        cntours = ax.contourf(vals, levels=levels_contourplot, cmap=color_maps[var])

        # Removes white lines
        for c in cntours.collections:
            c.set_edgecolor("face")

        fig.colorbar(cntours, ax=ax, label = '{} [{}]'.format(var, UNITS[var]))
        #a = sns.heatmap(vals, ax = ax, cbar = True, cmap = 'viridis', linewidths=0)
        ax.set_title(LONGNAME_STATISTICS[stat], fontsize = 14)
        ax.set_ylabel('Latitude')
        ax = add_ticks(ax, x_num_tikz = 9, y_num_tikz = 5)

    plt.xlabel('Longitude')
    plt.subplots_adjust(wspace = 0.2, hspace = 0.3, top=0.95, bottom=0.1, left = 0.14, right = .95)
    plt.savefig(path_python_figures + 'all_stat_variable_{}.pdf'.format(var))
    print('Finished {}'.format(var))
