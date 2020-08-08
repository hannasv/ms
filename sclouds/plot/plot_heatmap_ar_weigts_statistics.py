import os
import glob

import numpy as np
import seaborn as sns
import pandas as pd
import xarray as xr

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

fig, ax =  plt.subplots(nrows = n_rows, ncols = n_cols, sharex=True, sharey=False)
fig.set_size_inches(w = TEXT_WIDTH_IN, h = 0.5*TEXT_WIDTH_IN)

files = glob.glob('/home/hanna/EX3_Results_AR/AR-B-5/*weights*AR*L5*')
data = xr.open_mfdataset(files, combine='by_coords')

data['latitude'] = data.latitude.values.astype(float)
data['longitude'] = data.longitude.values.astype(float)
data = data.sortby('longitude')
data = data.sortby('latitude')

coefs = data.coeffs.values

store = {}
for i, key in enumerate(data.weights.values):
    portion = coefs[:, :, i]
    if 'O' in key:
        num = key.split('O')[-1]
        print(key.split('O'))
        key='L{}'.format(num)
    store[key] = np.array([ np.nanmin(portion), np.nanmax(portion), np.nanmean(portion),
    np.nanmedian(portion), np.nanstd(portion)  ])

y_ticklabels = ['Min', 'Max', 'Mean', 'Median', 'STD']

df = pd.DataFrame.from_dict(store)
ax=sns.heatmap(df, annot = True, yticklabels=y_ticklabels, ax=ax, cmap = 'bwr')
ax.set_xlabel('Weights')
#ax.set_ylabel(  )
plt.yticks(rotation=0)
plt.subplots_adjust(left=0.15, bottom=0.25, right=0.99, top=0.9, wspace=0.1, hspace=0.1)
plt.savefig('/home/hanna/MS-thesis/python_figs/heatmap_weights_ar_model.png')
