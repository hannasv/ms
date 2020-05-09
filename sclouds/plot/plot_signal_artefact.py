import glob
import os

import numpy as np
import xarray as xr
import pandas as pd

## Read statistics
from sclouds.plot.helper_bar_plot import autolabel, read_global_statistics_to_dataframe
from sclouds.helpers import VARIABLES, FILTERS, LONGNAME, path_stats_results, UNITS, path_store_plots

from sclouds.plot.helpers import import_matplotlib, TEXT_WIDTH_IN, TEXT_HEIGHT_IN
#mat = import_matplotlib()
import matplotlib.pyplot as plt

search_str = '/home/hanna/lagrings/results/*signal*.nc'
files = glob.glob( search_str )
dataset = xr.open_dataset(files[0])

fig, ax = plt.subplots(1, 1, figsize = (TEXT_WIDTH_IN, 0.5*TEXT_WIDTH_IN) )
plt.hist(dataset['filtered'].values, bins=100, orientation='horizontal')
plt.xlabel('Number of instances')
plt.ylabel('Average cloud fractional cover')
plt.title('Signal from artefact; average tcc in area defined as artefact ')
# impossible to distinguish this signal from whan all area is coverage by all clouds
plt.subplots_adjust(bottom = 0.15)
plt.savefig(os.path.join(path_store_plots, 'signal_artefact.pdf'))
