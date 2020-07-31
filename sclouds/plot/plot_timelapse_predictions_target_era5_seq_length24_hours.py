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

# LOAD DATA ERA5
files = glob.glob('/home/hanna/lagrings/ERA5_tcc/*2014*01*.nc')
era5 = xr.open_dataset(files[0])

# LOAD DATA TARGET
files = glob.glob('/home/hanna/lagrings/ERA5_monthly/*2014*01*tcc.nc')
target = xr.open_dataset(files[0])

# LOAD DATA AR PREDICTION
#dummy_ar = np.random.random((24, 81, 161))
fil = os.path.join('/home/hanna/MS-thesis/python_figs/','brand_new_prediction3.nc')
ar_data = xr.open_dataset(fil)

# LOAD DATA CONVLSTM PREDICTION
# TODO new file new best model.
fil = '/home/hanna/EX3_Results/ConvLSTM-B10-SL24-32-3x3-32-3x3/prediction.nc'
convlstm = xr.open_dataset(fil)
convlstm = convlstm.sel(batch=0)

n_rows  = 6
n_cols  = 4
n_pages = 4

var = 'tcc'

for page in range(n_pages):
    fig, axes =  plt.subplots(nrows = n_rows, ncols = n_cols, sharex=True, sharey=False)
    #fig.suptitle('ERA5: '+LONGNAME[var], fontsize = 14)
    fig.set_size_inches(w = TEXT_WIDTH_IN, h = TEXT_HEIGHT_IN-2)

    axes[0, 0].set_title('ECC', fontsize = 14)
    axes[0, 1].set_title('ERA5', fontsize = 14)
    axes[0, 2].set_title('ConvLSTM', fontsize = 14)
    axes[0, 3].set_title('AR', fontsize = 14)

    for i in range(n_rows):
        axes[i, 0].tick_params(labelbottom=False, labeltop=False, labelleft=False,
                               labelright=False , top=False, bottom=False, left=False, right=False)
        axes[i, 1].tick_params(labelbottom=False, labeltop=False, labelleft=False,
                               labelright=False , top=False, bottom=False, left=False, right=False)
        axes[i, 2].tick_params(labelbottom=False, labeltop=False, labelleft=False,
                               labelright=False , top=False, bottom=False, left=False, right=False)
        axes[i, 3].tick_params(labelbottom=False, labeltop=False, labelleft=False,
                               labelright=False , top=False, bottom=False, left=False, right=False)

        ts = int(page*6 + i)
        print(ts)
        axes[i, 0].set_ylabel(str(target.isel(time = ts)['time'].values)[-19:-16], fontsize = 14)

        tar = target.isel(time = ts)
        vals    = tar[var].values
        cntours = axes[i, 0].contourf(vals, levels=levels_contourplot, cmap='Blues_r',
                                      vmin = 0, vmax=1.0)
        # Removes white lines
        for c in cntours.collections:
            c.set_edgecolor("face")

        era = era5.isel(time = ts)
        vals    = np.flipud(era[var].values)
        cntours = axes[i, 1].contourf(vals, levels=levels_contourplot, cmap='Blues_r',
                                      vmin = 0, vmax=1.0)
        # Removes white lines
        for c in cntours.collections:
            c.set_edgecolor("face")

        conv = convlstm.isel(sequence_length = ts)
        vals    = conv[var].values
        cntours = axes[i, 2].contourf(vals, levels=levels_contourplot, cmap='Blues_r',
                                      vmin = 0, vmax=1.0)
        # Removes white lines
        for c in cntours.collections:
            c.set_edgecolor("face")

        arr = ar_data.isel(sequence_length = ts)
        vals   = arr[var].values.transpose()
        #vals = dummy_ar[i, :, :]
        cntours = axes[i, 3].contourf(vals, levels=levels_contourplot, cmap='Blues_r',
                                      vmin = 0, vmax=1.0)
        # Removes white lines
        for c in cntours.collections:
            c.set_edgecolor("face")

    fig.colorbar(cntours, ax = axes,  orientation='horizontal', anchor = (0.5, 0.05),
                 label = '{} [{}]'.format(var, UNITS[var]))
    plt.subplots_adjust(left=0.1, bottom=0.2, right=0.95, top=0.92,
                        wspace = 0.03, hspace = 0.03)
    plt.savefig(path_python_figures + 'comparting_seq_part_{}_of4.png'.format(page+1))
