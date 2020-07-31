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

path = '/home/hanna/lagrings/ERA5_monthly/'
path_python_figures = '/home/hanna/MS-thesis/python_figs/'
fil = glob.glob(path+'*2012*09*tcc*')
var = 'tcc'

data = xr.open_dataset(fil[0])
rel_indecies = [21, 22, 23, 24, 25]

fig, axes =  plt.subplots(nrows = len(rel_indecies), ncols = 1, sharex=True, sharey=False)
fig.set_size_inches(w = TEXT_WIDTH_IN, h = TEXT_HEIGHT_IN - 2)

for j, i in enumerate(rel_indecies):
    ax = axes[j]

    vals = data.isel(time = i)['tcc'].values
    cntours = ax.contourf(vals, levels=100, cmap='Blues_r')

    # Removes white lines
    for c in cntours.collections:
        c.set_edgecolor("face")

    fig.colorbar(cntours, ax=ax)
    #a = sns.heatmap(vals, ax = ax, cbar = True, cmap = 'viridis', linewidths=0)
    ax.set_title(str(data.isel(time = i)['time'].values)[:-16], fontsize = 14)
    ax.set_ylabel('Latitude')
    ax = add_ticks(ax, x_num_tikz = 9, y_num_tikz = 5)

plt.xlabel('Longitude')
plt.subplots_adjust(wspace = 0.2, hspace = 0.3, top=0.95, bottom=0.1, left = 0.14, right = .95)
plt.savefig(path_python_figures + 'timelapse_tcc_spike_09_2012.pdf')
print('Finished {}'.format(var))
