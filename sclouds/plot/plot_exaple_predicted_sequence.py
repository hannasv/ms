""" Plot a 24 hour timelapse cloud cover....

"""
import os
import glob

import numpy as np
import xarray as xr
import seaborn as sns

from sclouds.helpers import (path_input, path_stats_results, VARIABLES,
                                UNITS, LONGNAME)

from sclouds.plot.helpers import (TEXT_WIDTH_IN, TEXT_HEIGHT_IN,
                                    path_python_figures, import_matplotlib,
                                    cmap_contour_plot, levels_contourplot,
                                    file_format)
mat = import_matplotlib() # for mye
import matplotlib.pyplot as plt
files = glob.glob('/home/hanna/lagrings/ERA5_monthly/*tcc.nc')
#file = '/home/hanna/miphclac/2004_07/2004_07_tcc.nc'
data = xr.open_dataset(files[0])

n_rows = 6
n_cols = 3
var = 'tcc'

fig, axes =  plt.subplots(nrows = n_rows, ncols = n_cols, sharex=True, sharey=False)
#fig.suptitle(LONGNAME[var], fontsize = 14)
#plt.axis('off')
axes = axes.flatten()
fig.set_size_inches(w = TEXT_WIDTH_IN, h = TEXT_HEIGHT_IN-2)
for i, name in enumerate(['AR', 'ConvLSTM', 'Target']):
    axes[i].set_title(name)

for i, ax in enumerate(axes):
    ax.tick_params(labelbottom=False, labeltop=False, labelleft=False, labelright=False , top=False, bottom=False, left=False, right=False)
    test = data.isel(time = i)
    # plot target
    vals    = test[var].values
    cntours = ax.contourf(vals, levels=levels_contourplot, cmap='Blues_r')

    # Removes white lines
    for c in cntours.collections:
        c.set_edgecolor("face")

plt.subplots_adjust(left=0.1, bottom=0.05, right=0.95, top=0.95,hspace=0.1, wspace=0.1)
plt.savefig(path_python_figures + 'example_predicted_sequence_{}.png'.format(str(test.time.values)[:10]))
