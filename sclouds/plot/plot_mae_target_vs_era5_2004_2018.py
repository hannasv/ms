import xarray as xr
import numpy as np
import os


#ll['tcc'].plot()
#plt.show()
""" Plotting routine used to plot subplots of spatially averages monthly means
and filtered by land sea and both.
"""
import numpy as np
import xarray as xr
import seaborn as sns
import glob
import os

from sclouds.helpers import (path_input, path_stats_results, VARIABLES,
                                UNITS, LONGNAME)

from sclouds.plot.helpers import (TEXT_WIDTH_IN, TEXT_HEIGHT_IN,
                                    path_python_figures, import_matplotlib,
                                    cmap_contour_plot, levels_contourplot,
                                    file_format, add_ticks)
mat = import_matplotlib() # for mye
import matplotlib.pyplot as plt

n_rows = 1
n_cols = 1

var = 'tcc'

fig, ax =  plt.subplots(nrows = n_rows, ncols = n_cols, sharex=True, sharey=False)
fig.set_size_inches(w = TEXT_WIDTH_IN, h = 0.5*TEXT_WIDTH_IN)

data = xr.open_dataset(os.path.join(path_python_figures, 'mae_target_vs_era5.nc'))
vals    = data[var].values

cntours = ax.contourf(vals, levels=levels_contourplot, cmap='hot_r')

# Removes white lines
for c in cntours.collections:
    c.set_edgecolor("face")

#a = sns.heatmap(vals, ax = ax, cbar = True, cmap = 'viridis', linewidths=0)
ax.set_title('ERA5', fontsize = 14)
ax.set_ylabel('Latitude')
ax = add_ticks(ax)
#a.legend()
plt.xlabel('Longitude')
fig.colorbar(cntours, ax=ax, label = 'MAE')
plt.subplots_adjust(left=0.1, bottom=0.25, right=0.97, top=0.9, wspace=0.1, hspace=0.1)
plt.savefig(path_python_figures+'mae_era_vs_target_test_period_2014_to_2018.png')
