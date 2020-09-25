""" Code used for generating map used in thesis results.
"""
import os
import numpy as np
import cartopy as cp

import pandas as pd


import cartopy.crs as ccrs

from sclouds.plot.helpers import (PP_WIDTH_IN, PP_HEIGHT_IN,
                                    path_python_figures, import_matplotlib_pp,
                                    cmap_contour_plot, levels_contourplot,
                                    file_format, add_ticks)
mat = import_matplotlib_pp() # for mye
import matplotlib

path_presentation = '/home/hanna/MS-presentation/figures/'
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
matplotlib.rcParams.update({
    'figure.figsize' : (PP_WIDTH_IN, PP_HEIGHT_IN)
})
import matplotlib.pyplot as plt
import matplotlib as mpl






plt.subplots_adjust(left=0.05, bottom=0.1, right=0.9, top=0.99, wspace=0.1, hspace=0.1)
plt.savefig(os.path.join(path_presentation, "weights_AR-B-L5.png".format(file_format)))
