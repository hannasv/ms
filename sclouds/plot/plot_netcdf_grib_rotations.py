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
                                    file_format, add_ticks)
mat = import_matplotlib() # for mye
import matplotlib.pyplot as plt
fil = '/home/hanna/MSG2-SEVI-MSGCLMK-0100-0100-20090601090000.000000000Z-20090601091243-1374984.grb'

n_rows = 1
n_cols = 2

fig, axes =  plt.subplots(nrows = n_rows, ncols = n_cols, sharex=False, sharey=True)
fig.set_size_inches(w = TEXT_WIDTH_IN, h = 0.5*TEXT_WIDTH_IN)

data = xr.open_dataset(fil, engine='cfgrib'
            )['p260537'].values.reshape( (3712, 3712) )

sns.heatmap(data,
            cmap = 'viridis', cbar = False, ax = axes[0])
axes[0].invert_yaxis()
axes[0].set_title('.grib')
axes[0].get_xaxis().set_visible(False)
axes[0].get_yaxis().set_visible(False)

sns.heatmap(np.fliplr(data),
            cmap = 'viridis', cbar = False, ax = axes[1])
axes[1].invert_yaxis()
axes[1].set_title('.netCDF')
axes[1].get_xaxis().set_visible(False)
axes[1].get_yaxis().set_visible(False)

plt.subplots_adjust(left=0.1, bottom=0.25, right=0.95, top=0.9, wspace=0.1, hspace=0.1)
plt.savefig(path_python_figures + 'roations_grid.pdf')
