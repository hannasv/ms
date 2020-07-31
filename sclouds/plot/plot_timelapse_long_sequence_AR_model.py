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


# LOAD DATA AR PREDICTION
#dummy_ar = np.random.random((24, 81, 161))
fil = os.path.join('/home/hanna/MS-thesis/python_figs/','longer_sequence_brand_new_prediction3.nc')
data = xr.open_dataset(fil)


n_rows  = 6
n_cols  = 4
n_pages = 4

var = 'tcc'

for page in range(n_pages):
    fig, axes =  plt.subplots(nrows = n_rows, ncols = n_cols, sharex=True, sharey=False)
    fig.suptitle('AR: '+LONGNAME[var], fontsize = 14)
    fig.set_size_inches(w = TEXT_WIDTH_IN, h = TEXT_HEIGHT_IN-2)

    for i, ax in enumerate(axes.flatten()):
        ts = int(page*24 + i)
        ax.tick_params(labelbottom=False, labeltop=False, labelleft=False,
            labelright=False , top=False, bottom=False, left=False, right=False)
        test = data.isel(sequence_length = ts)
        # plot target
        vals    = test[var].values.transpose()
        cntours = ax.contourf(vals, levels=levels_contourplot, cmap='Blues_r',
                              vmin=0.0, vmax = 1.0)
        title   = 'T{:0>2d} - {:.4f}'.format(ts, np.nanmean(vals))
        ax.set_title(title, fontsize = 14)

        # Removes white lines
        for c in cntours.collections:
            c.set_edgecolor("face")

        # TODO if right column
        #axes[i, 0].set_ylabel(str(target.isel(time = ts)['time'].values)[-19:-16], fontsize = 14)

    fig.colorbar(cntours, ax = axes,  orientation='horizontal', anchor = (0.5, 0.05),
    label = '{} [{}]'.format(var, UNITS[var]))
    plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.92,
                        wspace = 0.2, hspace = 0.3)
    plt.savefig(path_python_figures + 'AR_long_model_seq_part_{}_of4.png'.format(page+1))
