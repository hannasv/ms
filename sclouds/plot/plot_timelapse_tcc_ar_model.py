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
import matplotlib as mpl
#files = glob.glob('/home/hanna/MS-thesis/python_figs/prediction.nc')
#file = '/home/hanna/miphclac/2004_07/2004_07_tcc.nc'
#data = xr.open_dataset('/home/hanna/MS-thesis/python_figs/prediction.nc')
fil = os.path.join('/home/hanna/MS-thesis/python_figs/','brand_new_prediction3.nc')
data = xr.open_dataset(fil)
print(data)
n_rows = 6
n_cols = 4
var = 'tcc'

fig, axes =  plt.subplots(nrows = n_rows, ncols = n_cols, sharex=True, sharey=False)
fig.suptitle('AR (2014-01-01): ' + LONGNAME[var], fontsize = 14)
#plt.axis('off')
fig.set_size_inches(w = TEXT_WIDTH_IN, h = TEXT_HEIGHT_IN-2)

for i, ax in enumerate(axes.flatten()):
    ax.tick_params(labelbottom=False, labeltop=False, labelleft=False, labelright=False , top=False, bottom=False, left=False, right=False)
    test = data.isel(sequence_length = i)
    print(test)
    #test[var].plot()
    #plt.savefig(path_python_figures + 'TEST_TEST_figure.png')

    # plot target
    vals    = test[var].values.transpose() # need to rotate data in ERA5
    cntours = ax.contourf(vals, levels=levels_contourplot, cmap='Blues_r')#, vmin=0, vmax=1)
    title   = 'T{:0>2d} - {:.4f}'.format(i, np.nanmean(vals))
    ax.set_title(title, fontsize = 14)

    # Removes white lines
    for c in cntours.collections:
        c.set_edgecolor("face")

cmap = mpl.cm.get_cmap('Blues_r')
norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

fig.colorbar(mappable, ax = axes,  orientation='horizontal', anchor = (0.5, 0.05), label = '{} [{}]'.format(var, UNITS[var]))
plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.92,
                    wspace = 0.2, hspace = 0.3)
plt.savefig(path_python_figures + 'AR_timelapse_cloud_cover_24hrs_from_2014_01_01.png')
