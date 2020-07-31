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

# CRITERION IS BEST AVERAGE MAE
models  = []
mins = []
maxs = []
avr = []

import json

#file = '/home/hanna/miphclac/2004_07/2004_07_tcc.nc'
#file = '/home/hanna/lagrings/ERA5_monthly/2004_07_tcc.nc'
#data = xr.open_dataset(file)

#subset = data.sel(time = '2004-07-02T12')
var = 'mae_test'

python_path = '/home/hanna/MS-thesis/python_figs/status_AR/'

#for path in glob.glob('/home/hanna/EX3_Results_AR/*'):
for path in glob.glob('/home/hanna/EX3_Results_AR/*')[:2]:
    # Autoregressive models
    for i in range(6):
        fig, ax =  plt.subplots(nrows = n_rows, ncols = n_cols, sharex=True, sharey=False)
        fig.set_size_inches(w = TEXT_WIDTH_IN, h = 0.5*TEXT_WIDTH_IN)

        files = glob.glob(path+'/performance*AR*o{}*'.format(i))
        print('Detected {} files'.format(len(files)))
        if len(files) > 0:
            title = files[0].split('_')[-3]
            data = xr.open_mfdataset(files, combine='by_coords')
            vals    = data[var].values

            models.append(title)
            mins.append(np.nanmin(vals))
            maxs.append(np.nanmax(vals))
            avr.append(np.nanmean(vals))

            if len(vals.shape) > 1:
                cntours = ax.contourf(vals, levels=levels_contourplot,
                                cmap='hot_r')

                # Removes white lines
                for c in cntours.collections:
                    c.set_edgecolor("face")

                fig.colorbar(cntours, ax=ax, label = '{}'.format(var))
                ax.set_title(title, fontsize = 14)
                #plt.xlabel('Longitude')
                ax.set_ylabel('Longitude')
                ax.set_xlabel('Latitude')

                plt.subplots_adjust(left=0.1, bottom=0.25, right=0.97, top=0.9, wspace=0.1, hspace=0.1)
                plt.savefig(python_path+title+'_{}.png'.format(var))
                plt.close()
            else:
                print('Currently one dimensional ...')
    # Traditional models
    for i in range(1, 6):
        fig, ax =  plt.subplots(nrows = n_rows, ncols = n_cols, sharex=True, sharey=False)
        fig.set_size_inches(w = TEXT_WIDTH_IN, h = 0.5*TEXT_WIDTH_IN)

        files = glob.glob(path+'/performance*TR*o{}*'.format(i))
        print('Detected {} files '.format(len(files)))
        if len(files) > 0:
            title = files[0].split('_')[-3]
            data = xr.open_mfdataset(files, combine='by_coords')
            # ax = sns.heatmap(data['mae_test'].values)
            vals    = data[var].values
            models.append(title)
            mins.append(np.nanmin(vals))
            maxs.append(np.nanmax(vals))
            avr.append(np.nansum(vals))

            if len(vals.shape) > 1:
                cntours = ax.contourf(vals, levels=levels_contourplot, cmap='hot_r')

                # Removes white lines
                for c in cntours.collections:
                    c.set_edgecolor("face")

                fig.colorbar(cntours, ax=ax, label = '{}'.format(var))
                ax.set_title(title, fontsize = 14)
                #plt.xlabel('Longitude')

                ax.set_ylabel('Longitude')
                ax.set_xlabel('Latitude')
                ax.set_title(title)

                plt.subplots_adjust(left=0.1, bottom=0.25, right=0.97, top=0.9, wspace=0.1, hspace=0.1)
                plt.savefig(python_path+title+'_{}.png'.format(var))
                plt.close()
            else:
                print('Currently one dimensional ...')

    data = {'models':models, 'min':mins, 'max':maxs, 'sum': avr}
    import json
    with open(python_path+'summary_AR_models.json', 'w') as outfile:
        json.dump(data, outfile)
