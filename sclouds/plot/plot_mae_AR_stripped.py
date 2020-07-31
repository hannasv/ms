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

files = glob.glob('/home/hanna/EX3_Results_AR_STR/AR-B-5/*performance*L1*')
print('Detected {} files'.format(len(files)))

data = xr.open_mfdataset(files, combine='by_coords')
data['latitude'] = data.latitude.values.astype(float)
data['longitude'] = data.longitude.values.astype(float)
data = data.sortby('longitude')

python_path = '/home/hanna/MS-thesis/python_figs/'
vals = data['mae_test'].values.transpose()
TEXT_WIDTH_IN  = 6.1023622
TEXT_HEIGHT_IN = 9.72440945
var = 'MAE'

fig, ax =  plt.subplots(nrows = 1, ncols = 1, sharex=True, sharey=False)
fig.set_size_inches(w = TEXT_WIDTH_IN, h = 0.5*TEXT_WIDTH_IN)

cntours = ax.contourf(vals, levels=100, cmap='hot_r')

# Removes white lines
for c in cntours.collections:
    c.set_edgecolor("face")

fig.colorbar(cntours, ax=ax, label = '{}'.format(var))
ax.set_title('AR-B-L1', fontsize = 14)
#plt.xlabel('Longitude')

ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax = add_ticks(ax)
#a.legend()
plt.xlabel('Longitude')
plt.subplots_adjust(left=0.1, bottom=0.25, right=0.97, top=0.9, wspace=0.1, hspace=0.1)
plt.savefig(python_path+'mea_best_ar_model_{}.png'.format('tcc'))
plt.close()
