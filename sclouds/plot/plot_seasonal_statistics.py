import os
import glob

import numpy as np
import xarray as xr
import seaborn as sns

from sclouds.helpers import (path_input, path_stats_results, VARIABLES,
                                UNITS, LONGNAME, SEASONS, STATISTICS, LONGNAME_STATISTICS)

from sclouds.plot.helpers import (TEXT_WIDTH_IN, TEXT_HEIGHT_IN,
                                    path_python_figures, import_matplotlib,
                                    cmap_contour_plot, levels_contourplot,
                                    file_format, color_maps)
mat = import_matplotlib() # for mye
import matplotlib.pyplot as plt

#path = '/home/hanna/lagrings/results/stats/test_season/'
base = '/home/hanna/lagrings/results/stats/season/'
n_rows = 5
n_cols = 4

for stat in STATISTICS:
    fig, axes =  plt.subplots(nrows = n_rows, ncols = n_cols, sharex=True, sharey=True)
    #fig.suptitle(LONGNAME[var], fontsize = 14)
    #plt.axis('off')
    fig.set_size_inches(w = TEXT_WIDTH_IN, h = TEXT_HEIGHT_IN - 3)

    axs = axes.flatten()
    counter = 0

    for var in VARIABLES:

        path = '/home/hanna/lagrings/results/stats/'
        files = glob.glob('/home/hanna/lagrings/results/stats/*global*{}*all.nc'.format(var))

        vmin = xr.open_dataset(files[0])['min'].values
        vmax = xr.open_dataset(files[0])['max'].values

        for season in SEASONS:
            axs[counter].tick_params(labelbottom=False, labeltop=False, labelleft=False, labelright=False , top=False, bottom=False, left=False, right=False)
            files = glob.glob(base +'stats_pixel*{}*{}*all.nc'.format(season, var))

            try:
                data = xr.open_dataset(files[0])
                if var != 'tcc':
                    vals = np.flipud(data[stat].values)
                else:
                    vals = data[stat].values
                cntours = axs[counter].contourf(vals, vmin=vmin, vmax=vmax,
                            levels=levels_contourplot, cmap=color_maps[var])

                # Removes white lines
                for c in cntours.collections:
                    c.set_edgecolor("face")

            except IndexError:
                print('problem {} {}'.format(season, var))


            if counter < 4:
                 axs[counter].set_title('{}'.format(season))

            #if counter%4==0:
            #    axs[counter].set_ylabel('{} [{}]'.format(var, UNITS[var]))

            counter += 1


            if counter%4==0:
                fig.colorbar(cntours, ax=axs[counter-1], label = '{} [{}]'.format(var, UNITS[var]))

    fig.suptitle(LONGNAME_STATISTICS[stat])
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.85, top=0.9, hspace=0.1, wspace=0.1)
    plt.savefig(path_python_figures + 'seasonal_{}_all_variables.png'.format(stat))
    print('Finished {}'.format(stat))
