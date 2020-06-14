import os
import glob

import numpy as np
import xarray as xr
import pandas as pd

## Read statistics
from sclouds.plot.helper_bar_plot import autolabel, read_global_statistics_to_dataframe
from sclouds.helpers import VARIABLES, FILTERS, LONGNAME, path_stats_results, UNITS, path_store_plots

from sclouds.plot.helpers import import_matplotlib, TEXT_WIDTH_IN, TEXT_HEIGHT_IN
mat = import_matplotlib()
import matplotlib.pyplot as plt

STATISTICS = ['mean', 'min', 'max', 'std', 'median']
FILTERS    = ['land', 'sea', 'all', 'coast']

def retrive_stat_from_stores_statistics(statistic = 'min'):
    """Regrouped to easy acess the statistics"""
    dd = read_global_statistics_to_dataframe()
    df_stat = dd.unstack().swaplevel().unstack().swaplevel().transpose()
    return df_stat[statistic]

def retrive_filter_from_stores_statistics(filter_key = 'all'):
    """Regrouped to easy acess the statistics"""
    dd = read_global_statistics_to_dataframe()
    return dd[filter_key]

def retrive_variable_from_stores_statistics(variable = 'tcc'):
    """Regrouped to easy acess the statistics"""
    dd = read_global_statistics_to_dataframe()
    df_stat = dd.unstack().swaplevel().unstack()#.swaplevel().transpose()
    return df_stat[variable].unstack()


fig, axes = plt.subplots(len(VARIABLES), 1, figsize = (TEXT_WIDTH_IN, TEXT_HEIGHT_IN - 2), sharex = True)

width = 0.4
w = width/2

for i, ax in enumerate(axes.flatten()):
    var = VARIABLES[i]
    TCC = retrive_variable_from_stores_statistics(variable = var)
    TCC.drop('artefact', axis = 0, inplace = True)
    labels = TCC['mean'].index.values
    x_ticks_labels = TCC.transpose()['all'].index.values
    list_y_values = np.array([ TCC.transpose()[lab].values for lab in labels ])
    y_pos = np.arange(len(list_y_values[0]))

    #for i, y_values in enumerate(list_y_values):
    rect = ax.bar(y_pos - 3*width/4, list_y_values[0], width=w, align = 'center', alpha = 0.5, label = '{}'.format(labels[0]))
    #autolabel(rect, ax) # add annotations to plot

    rect2 = ax.bar(y_pos - width/4, list_y_values[1], width=w, align='center', alpha=0.5, label='{}'.format(labels[1]))
    #autolabel(rect2, ax, below = False)  # add annotations to plot

    rect3 = ax.bar(y_pos + width/4, list_y_values[2], width=w, align='center', alpha=0.5, label='{}'.format(labels[2]))
    #autolabel(rect3, ax)  # add annotations to plot

    rect4 = ax.bar(y_pos + 3*width/4, list_y_values[3], width=w, align='center', alpha=0.5, label='{}'.format(labels[3]))
    #autolabel(rect4, ax, below = False)  # add annotations to plot
    ax.set_ylabel('{} [{}]'.format(var, UNITS[var]))
    ax.set_title(LONGNAME[var])

plt.xticks(y_pos, x_ticks_labels)
plt.legend(ncol = len(FILTERS), frameon = False, bbox_to_anchor=(0.8, -0.25))
#plt.subplots_adjust(wspace = 0.2, hspace = 0.3, top=0.9, bottom=0.1, left = 0.14, right = .95)
plt.subplots_adjust(wspace = 0.3, hspace = 0.2, top= 0.95, bottom= 0.1, left= 0.15, right= 0.97)
plt.savefig(os.path.join(path_store_plots, 'bar_plot_global_statistics_new_legend.pdf'))
