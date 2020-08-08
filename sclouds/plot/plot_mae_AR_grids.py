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
python_path = '/home/hanna/MS-thesis/python_figs/'
folders = ['AR-B-5', 'AR-5']# 'AR-T-5', 
#folders = ['AR-B-5']
degree = 'L1'
for fold in folders:
    for tpe in ['AR', 'TR']:#, 'TR']:
        print('Searching for {}'.format('/home/hanna/EX3_Results_AR/{}/*performance*{}*{}*'.format(fold, tpe, degree)))
        files = glob.glob('/home/hanna/EX3_Results_AR/{}/*performance*{}*{}*'.format(fold, tpe, degree))
        name = '-'.join(files[0].split('_')[-3].split('-5-'))

        #if not os.path.isfile(python_path+'mea_best_ar_model_{}_{}_in_folder_{}.png'.format('tcc', degree, name)):
        #    print('Merging {} files ...'.format(len(files)))
        try:
            data = xr.open_mfdataset(files, combine='by_coords')
            data['latitude'] = data.latitude.values.astype(float)
            data['longitude'] = data.longitude.values.astype(float)
            data = data.sortby('longitude')
            data = data.sortby('latitude')
            #data.to_netcdf('/home/hanna/TEMP_MODELS/{}_L5.nc'.format(fold))
            #print('stored nc files')
            vals = data['mae_test'].values.transpose() # constant is num hour 2014+-2018
            print(np.nanmean(vals))
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

            ax.set_title('{}'.format(name), fontsize = 14)
            #plt.xlabel('Longitude')

            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax = add_ticks(ax)
            #a.legend()
            plt.xlabel('Longitude')
            plt.subplots_adjust(left=0.1, bottom=0.25, right=0.97, top=0.9, wspace=0.1, hspace=0.1)
            plt.savefig(python_path+'mea_best_ar_model_{}_{}_in_folder_{}.png'.format('tcc', degree, name))
            plt.close()
        except OSError as e:
            print('unable to genrate plot for {}'.format(name))
