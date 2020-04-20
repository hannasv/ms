from sclouds.plot.helpers import TEXT_WIDTH_IN, path_python_figures, import_matplotlib
#matplotlib = import_matplotlib()
#plt.rcParams.update({'figure.max_open_warning': 0})
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from sclouds.helpers import FILTERS
from sclouds.helpers import LAT, LON, SPATIAL_RESOLUTION, get_lon_array, get_lat_array

from sclouds.io.filter import Filter

import numpy as np
import seaborn as sns

lons = get_lon_array()
lats = get_lat_array()

import seaborn as sns


fig, axes = plt.subplots(2, 2, figsize = (TEXT_WIDTH_IN, TEXT_WIDTH_IN),
                         sharex = True, sharey = True)
flat_axes = axes.flatten()
counter = 0
for flter, ax in zip(FILTERS, flat_axes):
    #ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    fil = Filter(flter).filter_ds['land_mask'].values
    cntours = ax.contourf(fil, levels=100, cmap='Blues')
    #fig.colorbar(cntours, ax=ax)

    if counter == 0 or counter == 2:
        ax.set_ylabel('Latitude')

    if counter == 3 or counter == 2:
        ax.set_xlabel('Longitude')

    counter += 1
    ax.set_yticklabels(labels = np.linspace(30, 50, 5))
    ax.set_xticklabels(labels = ['{}'.format(round(i)) for i in np.linspace(-15, 25, 4)], rotation = 45)

    #test = np.flipud(Filter(flter).filter_ds['land_mask'].values)
    #a = sns.heatmap(test, ax = ax, cbar = False) # xticklabels=get_lon_array(), yticklabels=get_lat_array()
    ax.set_title(flter)


plt.savefig(path_python_figures + 'filters.pdf')
