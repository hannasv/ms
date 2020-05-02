from sclouds.plot.helpers import TEXT_WIDTH_IN, path_python_figures, import_matplotlib
#matplotlib = import_matplotlib()
#plt.rcParams.update({'figure.max_open_warning': 0})
mat = import_matplotlib() # for mye
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

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
    cntours = ax.contourf(fil, levels=100, cmap='bone')
    #fig.colorbar(cntours, ax=ax)

    if counter == 0 or counter == 2:
        ax.set_ylabel('Latitude')

    if counter == 3 or counter == 2:
        ax.set_xlabel('Longitude')

    # Make a plot with major ticks that are multiples of 20 and minor ticks that
    # are multiples of 5.  Label major ticks with '%d' formatting but don't label
    # minor ticks.
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))

    # For the minor ticks, use no labels; default NullFormatter.
    #ax.xaxis.set_minor_locator(MultipleLocator(5))
    #ax.yaxis.set_minor_locator(MultipleLocator(5))

    counter += 1
    ax.set_yticklabels(labels = np.linspace(30, 50, 5))
    ax.set_xticklabels(labels = np.linspace(-20, 25, 10), rotation = 45)

    #test = np.flipud(Filter(flter).filter_ds['land_mask'].values)
    #a = sns.heatmap(test, ax = ax, cbar = False) # xticklabels=get_lon_array(), yticklabels=get_lat_array()
    ax.set_title(flter)

plt.savefig(path_python_figures + 'filters.pdf')
