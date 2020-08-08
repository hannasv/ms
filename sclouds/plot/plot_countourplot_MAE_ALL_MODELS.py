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
import matplotlib as mpl

fig, axes =  plt.subplots(nrows = 3, ncols = 1, sharex=True, sharey=False)
fig.set_size_inches(w = TEXT_WIDTH_IN, h = TEXT_HEIGHT_IN-2)

################## The AR model
folders = ['AR-B-5']
degree = 'L5'

files = glob.glob('/home/hanna/EX3_Results_AR/AR-B-5/*performance*AR*L5*')
name = '-'.join(files[0].split('_')[-3].split('-5-'))

data = xr.open_mfdataset(files, combine='by_coords')
data['latitude'] = data.latitude.values.astype(float)
data['longitude'] = data.longitude.values.astype(float)
data = data.sortby('longitude')
data = data.sortby('latitude')
#data.to_netcdf('/home/hanna/TEMP_MODELS/{}_L5.nc'.format(fold))
#print('stored nc files')
vals = data['mae_test'].values.transpose() # constant is num hour 2014+-2018
cntours = axes[-1].contourf(vals, levels=100, cmap='hot_r')

# Removes white lines
for c in cntours.collections:
    c.set_edgecolor("face")
#fig.colorbar(cntours, ax=axes[-1], label = '{}'.format(var))
axes[-1].set_title('AR-B-L5', fontsize = 14)
#plt.xlabel('Longitude')
#ax.set_xlabel('Longitude')
axes[-1].set_ylabel('Latitude')
axes[-1] = add_ticks(axes[-1])

#################### Plot ConvLSTM

data = xr.open_dataset(os.path.join('/home/hanna/MS-thesis/python_figs/','mae_convlstm_best_model.nc'))
vals    = data[var].values/43680
print(np.mean(vals))
cntours = axes[1].contourf(vals, levels=levels_contourplot, cmap='hot_r')

# Removes white lines
for c in cntours.collections:
    c.set_edgecolor("face")

#a = sns.heatmap(vals, ax = ax, cbar = True, cmap = 'viridis', linewidths=0)
axes[1].set_title('ConvLSTM-B10-SL24-32-3x3-32-3x3', fontsize = 14)
axes[1].set_ylabel('Latitude')
axes[1] = add_ticks(axes[1])
#a.legend()
plt.xlabel('Longitude')


fig.colorbar(cntours, ax=ax, label = 'MAE')










n_rows = len(STATISTICS)
n_cols = 1
levels_contourplot = 100
path_stats_results = '/home/hanna/lagrings/results/stats'




cmap = mpl.cm.get_cmap(color_maps[var])
norm = mpl.colors.Normalize(vmin=MIN, vmax=MAX)
mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

fig.colorbar(mappable, ax=axes, orientation = 'vertical',
         label = '{} [{}]'.format(var, UNITS[var]))
plt.xlabel('Longitude')
plt.subplots_adjust(wspace = 0.2, hspace = 0.3, top=0.95, bottom=0.1, left = 0.14, right = .75)
plt.savefig(path_python_figures + 'MAE_all_vars{}.pdf'.format(var))
