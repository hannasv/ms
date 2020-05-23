import glob
import os, sys
import json

import numpy as np
import xarray as xr
import pandas as pd
import seaborn as sns
## Read statistics
from sclouds.plot.helper_bar_plot import (autolabel,
                    read_global_statistics_to_dataframe)
from sclouds.helpers import (VARIABLES, FILTERS, LONGNAME, path_stats_results,
                                UNITS, path_store_plots)

from sclouds.plot.helpers import (import_matplotlib, TEXT_WIDTH_IN,
                                    TEXT_HEIGHT_IN, cmap_contour_plot, add_ticks)
#mat = import_matplotlib()
import matplotlib.pyplot as plt

save_dir = '/home/hanna/MS-suppl/files/'
files = glob.glob(save_dir + '*ERA5*.json')

x_num_tikz = 20
y_num_tikz = 15

lat_min = 30.0
lat_max = 50.0

lon_min = -15
lon_max = 25.0

num_x = 161
num_y = 81

xticks        = np.linspace(0.5, num_x-0.5, x_num_tikz, endpoint = True)
xticks_labels = np.linspace(lon_min, lon_max, x_num_tikz, endpoint = True)

yticks        = np.linspace(0.5, num_y-0.5, y_num_tikz, endpoint = True)
yticks_labels = np.linspace(lat_min, lat_max, y_num_tikz, endpoint = True)

def get_dict_with_all_keys():
    ex_fil = glob.glob(save_dir + '*ERA5*.json')
    ex_fil = np.sort(ex_fil)[:-1]
    merged_dict = {}
    for fil in ex_fil:
        with open(fil, 'r') as f:
            data_grid = json.load(f)
        merged_dict.update(data_grid)
    return merged_dict
data = get_dict_with_all_keys()

areas = np.zeros((len(data.keys()), len(list(data['30.0'].keys()))))
for i, lat in enumerate(list(data.keys())):
    for j, lon in enumerate(list(data[lat].keys())):
        #print(data[lat][lon].keys())
        centre = np.sum(data[lat][lon]['centre']['area'])
        left = np.sum(data[lat][lon]['left']['area'])
        up = np.sum(data[lat][lon]['up']['area'])
        right = np.sum(data[lat][lon]['right']['area'])
        down = np.sum(data[lat][lon]['down']['area'])

        areas[i][j] = centre+left+up+down+left

fig, ax = plt.subplots(1, 1, figsize = (TEXT_WIDTH_IN, 0.5*TEXT_WIDTH_IN) )
ax = sns.heatmap(areas[:, :161], ax = ax, cmap = 'viridis', cbar_kws = {'label':'accumulated area in km2'})
ax.invert_yaxis()
#ax.plot(dataset['filtered'].values, bins=100, orientation='horizontal')
plt.title('Area contrubution to a fraction in km2?')

plt.ylabel('Latitude')
plt.xlabel('Longitude')

# impossible to distinguish this signal from whan all area is coverage by all clouds

plt.yticks(yticks, labels = yticks_labels.astype(int), rotation='horizontal');
plt.xticks(xticks, labels = xticks_labels.astype(int),  rotation='horizontal');

plt.subplots_adjust(bottom = 0.2, right = 0.97)
plt.savefig(os.path.join(path_store_plots, 'signal_area_pixel.pdf'))

############ MAKE PLOT OF EACH

areas = np.zeros((len(data.keys()), len(list(data['30.0'].keys()))))


for k in ['centre', 'down', 'up', 'right', 'left', 'corner']:
    for i, lat in enumerate(list(data.keys())):
        for j, lon in enumerate(list(data[lat].keys())):
            areas[i][j] = np.sum(data[lat][lon][k]['area'])


    fig, ax = plt.subplots(1, 1, figsize = (TEXT_WIDTH_IN, 0.5*TEXT_WIDTH_IN) )
    ax = sns.heatmap(areas[:, :161], ax = ax, cmap = 'viridis', cbar_kws = {'label':'accumulated area in %'})
    ax.invert_yaxis()
    plt.ylabel('Latitude')
    plt.xlabel('Longitude')

    # impossible to distinguish this signal from whan all area is coverage by all clouds
    plt.yticks(yticks, labels = yticks_labels.astype(int), rotation='horizontal');
    plt.xticks(xticks, labels = xticks_labels.astype(int),  rotation='horizontal');
    # ax.plot(dataset['filtered'].values, bins=100, orientation='horizontal')

    plt.title('Area contrubution from {} to a fraction in km2?'.format(k))
    # impossible to distinguish this signal from whan all area is coverage by all clouds
    plt.subplots_adjust(bottom = 0.2, right = 0.97)
    plt.savefig(os.path.join(path_store_plots, 'signal_area_{}_pixel.pdf'.format(k)))
