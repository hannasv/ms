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
import matplotlib as mpl

DEVIATIONS = ['std','mad']
STATISTICS = ['mean', 'min', 'max', 'median']

n_rows = len(STATISTICS)
n_cols = 1
levels_contourplot = 100
path_stats_results = '/home/hanna/lagrings/results/stats'

for var in VARIABLES: #['mean']:#STATISTICS:

    fig, axes =  plt.subplots(nrows = n_rows, ncols = n_cols, sharex=True, sharey=False)
    fig.set_size_inches(w = TEXT_WIDTH_IN, h = TEXT_HEIGHT_IN - 2)
    #fig.suptitle(var, fontsize = 16)
    files = glob.glob(path_stats_results + '/*pixel*{}*all.nc'.format(var))
    if len(files) != 1:
        print(files)
    data = xr.open_dataset(files[0])

    mins = []
    maxs = []

    for s in ['mean', 'min', 'max', 'median']:
        sub = data[s]
        mins.append(float(sub.min()))
        maxs.append(float(sub.max()))

    MIN = np.min(mins)
    MAX = np.max(maxs)

    for stat, ax in zip(STATISTICS, axes):
        vals = np.abs(data[stat].values)
        if var != 'tcc':
            vals   = np.flipud(vals)
            lab = '{} [{}]'.format(var, UNITS[var])
        else:
            lab = 'cfc [1]'
        cntours = ax.contourf(vals, levels=levels_contourplot, cmap=color_maps[var],
                                            vmin = MIN, vmax=MAX)
        # Removes white lines
        for c in cntours.collections:
            c.set_edgecolor("face")

        #a = sns.heatmap(vals, ax = ax, cbar = True, cmap = 'viridis', linewidths=0)
        ax.set_title(LONGNAME_STATISTICS[stat], fontsize = 14)
        ax.set_ylabel('Latitude')
        ax = add_ticks(ax, x_num_tikz = 9, y_num_tikz = 5)

    cmap = mpl.cm.get_cmap(color_maps[var])
    norm = mpl.colors.Normalize(vmin=MIN, vmax=MAX)
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    fig.colorbar(mappable, ax = axes,  orientation='vertical',
                    label = lab, anchor = (0.4, -0.5),# 1.3 funker
                    aspect = 40, shrink=1.0)
    plt.xlabel('Longitude')
    plt.subplots_adjust(wspace = 0.2, hspace = 0.3, top=0.95, bottom=0.1, left = 0.14, right = .80)
    plt.savefig(path_python_figures + 'all_stat_variable_{}.png'.format(var))
