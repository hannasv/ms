""" Code used for generating map used in thesis results.
"""
import os
import numpy as np
import cartopy as cp

import pandas as pd
import matplotlib.pyplot as plt

import cartopy.crs as ccrs

from sclouds.plot.helpers import (TEXT_WIDTH_IN, TEXT_HEIGHT_IN,
                                  path_python_figures, import_matplotlib,
                                  file_format)

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

matplotlib = import_matplotlib()
matplotlib.rcParams.update({
    'figure.figsize' : (TEXT_WIDTH_IN, 0.5*TEXT_WIDTH_IN)
})


def plot_satellite_projection():
    """Stored for future use."""
    ax = plt.axes(projection=ccrs.Geostationary(central_longitude=0.0, satellite_height=35785831,
                   false_easting=0, false_northing=0, globe=None))
    ax.coastlines()

    ax.add_feature(cp.feature.OCEAN, zorder=0)
    ax.add_feature(cp.feature.LAND, zorder=0, edgecolor='black')

    ax.set_extent([-15, 25.25, 30, 55.25])

    plt.subplots_adjust(left=0.01, bottom=0.01, right=0.9, top=0.9, wspace=0.1, hspace=0.1)
    plt.savefig(os.path.join(save_dir, "Domain_SAT.png"), bbox_inches='tight')
    ax.set_xticklabels([3, 5, 6, 7])
    return

ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0.0, linestyle='--')
gl.xlabels_top = False
gl.ylabels_left = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER


ax.add_feature(cp.feature.OCEAN, zorder=0)
ax.add_feature(cp.feature.LAND, zorder=0, edgecolor='black')

ax.set_extent([-15, 25., 30, 50])

plt.subplots_adjust(left=0.01, bottom=0.1, right=0.97, top=0.9, wspace=0.1, hspace=0.1)
plt.savefig(os.path.join(path_python_figures, "Domain.{}".format(file_format)))
