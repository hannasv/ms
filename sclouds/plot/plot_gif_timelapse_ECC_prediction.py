""" Plot a 24 hour timelapse cloud cover....

"""
import os
import glob

import numpy as np
import xarray as xr
import seaborn as sns

from sclouds.helpers import (path_input, path_stats_results, VARIABLES,
                                UNITS, LONGNAME)

from sclouds.plot.helpers import (PP_WIDTH_IN, PP_HEIGHT_IN,
                                    path_python_figures, import_matplotlib_pp,
                                    cmap_contour_plot, levels_contourplot,
                                    file_format, add_ticks)
mat = import_matplotlib_pp() # for mye
import matplotlib.pyplot as plt
import matplotlib as mpl
#fil = '/home/hanna/EX3_Results/best_model/ConvLSTM-B10-SL24-16-3x3-16-3x3/prediction.nc'

path_presentation = '/home/hanna/MS-presentation/'
start_date = '02-01-2014'
model = 'ECC'

path_store_figures = os.path.join(path_presentation, model)
# IF FOLDER DOESN'T EXITS MAKE FOLDER FOR PRESENTATION
if not os.path.isdir(path_store_figures):
    os.mkdir(path_store_figures)

# LOAD DATA TARGET
files = glob.glob('/home/hanna/lagrings/ERA5_monthly/*2014*01*tcc.nc')
data = xr.open_dataset(files[0]).isel(time=slice(24, 72))

len_seq = 24
var = 'tcc'

for i in range(len_seq):
    fig, ax =  plt.subplots(nrows = 1, ncols = 1, sharex=False, sharey=False)
    title   = '{} (2014-02-01): T{:0>2d}'.format(model, i)
    ax.set_title(title)
    fig.set_size_inches(w = PP_WIDTH_IN, h = PP_HEIGHT_IN)
    #ax.tick_params(labelbottom=False, labeltop=False, labelleft=False, labelright=False,
    #top=False, bottom=False, left=False, right=False)
    test = data.isel(time = i)
    vals    = test[var].values
    cntours = ax.contourf(vals, levels=levels_contourplot, cmap='Blues_r',
                          vmin = 0.0, vmax = 1.0)
    # Removes white lines
    for c in cntours.collections:
        c.set_edgecolor("face")

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax = add_ticks(ax)

    cmap = mpl.cm.get_cmap('Blues_r')
    norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    fig.colorbar(mappable, ax=ax, label = 'cfc [1]')
    plt.subplots_adjust(left=0.1, bottom=0.15, right=.99, top=0.92,
                        wspace = 0.2, hspace = 0.3)
    plt.savefig(os.path.join(path_store_figures, '{}_{}_{}.png'.format(i, model, start_date)))
    plt.close()
