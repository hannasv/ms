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
mat = import_matplotlib()
import matplotlib.pyplot as plt
HMM = ['q' 't2m' 'r' 'sp' 'bias' 'O1'] # KAN MAN TA SEL DENNE ???
# extending units for bae
UNITS['O1']   = 1
UNITS['bias'] = 1

LONGNAME['bias'] = 'Bias'
LONGNAME['O1']   = 'Cloud Fractional Cover Lag1'

HMM = ['q', 't2m', 'r', 'sp', 'bias', 'O1']
n_rows = len(HMM)
n_cols = 1
#print(n_rows)
#print(n_cols)
levels_contourplot = 100

fig, axes =  plt.subplots(nrows = n_rows, ncols = n_cols, sharex=True, sharey=False)
print(axes.flatten())
fig.set_size_inches(w = TEXT_WIDTH_IN, h = TEXT_HEIGHT_IN - 2)
#print(axes)
files = glob.glob('/home/hanna/EX3_Results_AR/AR-B-5/*weights*AR*L1*')
print('Merging {} files ..'.format(len(files)))
data = xr.open_mfdataset(files, combine='by_coords')
print('Finished merging ....')
data['latitude'] = data.latitude.values.astype(float)
data['longitude'] = data.longitude.values.astype(float)
data = data.sortby('longitude')
data = data.sortby('latitude')
print('Sorted coordinates.')

for i, ax in enumerate(axes.flatten()):
    var = HMM[i]
    subset = data.sel(weights = var)
    subset = subset.where(subset.coeffs<100)
    vals = subset.coeffs.values.transpose()
    cntours = ax.contourf(vals, levels=levels_contourplot, cmap='bwr')

    # Removes white lines
    for c in cntours.collections:
        c.set_edgecolor("face")

    fig.colorbar(cntours, ax=ax, label = '{} [{}]'.format(var, UNITS[var]))
    #cbarlabels = np.linspace(np.floor(np.min(vals)), np.ceil(np.max(vals)), num=6, endpoint=True) )
    #a = sns.heatmap(vals, ax = ax, cbar = True, cmap = 'viridis', linewidths=0)
    ax.set_title(LONGNAME[var], fontsize = 14)
    ax.set_ylabel('Latitude')

    ax = add_ticks(ax, x_num_tikz = 9, y_num_tikz = 5)

plt.xlabel('Longitude')
plt.subplots_adjust(wspace = 0.2, hspace = 0.3, top=0.9, bottom=0.1, left = 0.14, right = .95)
plt.savefig(path_python_figures + 'weights_best_ar_model.png')
    #print('Finished {}'.format(stat))
