import os
import glob

import numpy as np
import xarray as xr
import pandas as pd

from sclouds.plot.helpers import import_matplotlib
mat = import_matplotlib()
import matplotlib.pyplot as plt

def read_global_statistics_to_dataframe():
    """ Read computed statistics to dataframe.
    """
    read_dir   = '/home/hanna/lagrings/ERA5_monthly/'
    save_dir   = '/home/hanna/lagrings/ERA5_stats/results/'
    save_dir   = '/home/hanna/lagrings/results/stats/'

    STATS         = ['mean', 'std', 'min', 'max', 'median', 'mad'] # 'median',
    VALID_VARS    = ['r', 'q', 't2m', 'sp', 'tcc']
    VALID_FILTERS = ['coast', 'sea', 'land', 'artefact', 'all']

    results = {}
    for f in VALID_FILTERS:
        results[f] = {}

        for var in VALID_VARS:
            files = glob.glob(os.path.join( save_dir, '*global*{}*{}*.nc'.format(var, f) ))
            if len(files) != 1:
                print(os.path.join( save_dir, '*global*{}*{}*.nc'.format(var, f) ))
                print(files)
            fil = files[0]
            data = xr.open_dataset(fil)
            results[f][var] = {}

            for stats in STATS:
                vals = data[stats].values
                results[f][var][stats] = vals

    reform = {(outerKey, innerKey): values for outerKey, innerDict in results.items() for innerKey, values in innerDict.items()}
    return pd.DataFrame(reform)


def autolabel(rects, ax, below = False, formatter = '{:.2f}'):
    """Attach a text label above each bar in *rects*, displaying its height.

    Parameters
    --------------------------
    rect : returned from ax.bar
        The square you want to annotate.
    ax : matplotlib axis
    """
    for rect in rects:
        height = rect.get_height()
        #if height < 0:
        #    rect.set_color('red')
        if below:
            vert_offset = -15
        else:
            vert_offset = 3
        ax.annotate(formatter.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, vert_offset),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    return

def plot_bar_plot(data, title, ylabel, filename = None, formatter = '{:.1f}'):
    """ Helper function used to compute bar plots.

    :param y_values:
    :param title:
    :param xlabel:
    :param filename:
    :return:

    """

    y_vals = data.values
    y_pos = np.arange(len(y_vals))
    x_ticks_labels = data.index.values
    matplotlib = import_matplotlib() # this need to be imported before matplotlib plt.
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize = (PP_TEXT_WIDTH_IN, PP_TEXT_HEIGHT_IN) )
    rect = ax.bar(y_pos, y_vals, align = 'center', alpha = 0.5)
    ax.set_ylabel(ylabel)
    autolabel(rect, ax, False, formatter) # add annotations to plot
    plt.xticks(y_pos, x_ticks_labels, rotation = 45)
    plt.title(title)
    plt.subplots_adjust(hspace= 0.5, top= 0.9, bottom= 0.2, left= 0.07, right= 0.97)
    #plt.legend()

    if filename:
        plt.savefig(os.path.join(base, save_dir, '{}.png'.format(filename)))

    return



def plot_2grouped_bar_plot(list_y_values, labels, x_ticks_labels, title, ylabel, filename = None):

    """ Helper function used to compute bar plots.

    TODO add list_labels,

    :param list_y_values:

    :param title:

    :param xlabel:

    :param filename:

    :return:

    """

    y_pos = np.arange(len(list_y_values[0]))
    width = 0.35
    matplotlib = import_matplotlib() # this need to be imported before matplotlib plt.
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize = (PP_TEXT_WIDTH_IN, PP_TEXT_HEIGHT_IN))
    #for i, y_values in enumerate(list_y_values):
    rect = ax.bar(y_pos - width/2, list_y_values[0], width, align='center', alpha = 0.5, label = '{}'.format(labels[0]))
    autolabel(rect, ax) # add annotations to plot
    rect2 = ax.bar(y_pos + width/2, list_y_values[1], width, align='center', alpha=0.5, label='{}'.format(labels[1]))
    autolabel(rect2, ax)  # add annotations to plot

    ax.set_ylabel(ylabel)
    plt.xticks(y_pos, x_ticks_labels, rotation = 45)
    plt.title(title)
    plt.legend(loc='lower left')
    plt.subplots_adjust(hspace= 0.5, top= 0.9, bottom= 0.2, left= 0.07, right= 0.97)

    if filename:
        plt.savefig(os.path.join(base, save_dir, '{}.png'.format(filename)))

    return



def plot_3grouped_bar_plot(list_y_values, labels, x_ticks_labels, title, ylabel, filename = None):

    """ Helper function used to compute bar plots.

    TODO add list_labels,

    :param list_y_values:
    :param title:
    :param xlabel:
    :param filename:
    :return:

    """

    y_pos = np.arange(len(list_y_values[0]))
    width = 0.25
    matplotlib = import_matplotlib() # this need to be imported before matplotlib plt.
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize = (PP_TEXT_WIDTH_IN, PP_TEXT_HEIGHT_IN))
    #for i, y_values in enumerate(list_y_values):
    rect = ax.bar(y_pos - width, list_y_values[0], width, align = 'center', alpha = 0.5, label = '{}'.format(labels[0]))
    autolabel(rect, ax) # add annotations to plot

    rect2 = ax.bar(y_pos , list_y_values[1], width, align='center', alpha=0.5, label='{}'.format(labels[1]))
    autolabel(rect2, ax, below = True)  # add annotations to plot

    rect2 = ax.bar(y_pos + width, list_y_values[2], width, align='center', alpha=0.5, label='{}'.format(labels[2]))
    autolabel(rect2, ax)  # add annotations to plot

    ax.set_ylabel(ylabel)
    plt.xticks(y_pos, x_ticks_labels, rotation = 45)
    plt.title(title)
    plt.legend()
    plt.subplots_adjust(hspace= 0.5, top= 0.9, bottom= 0.2, left= 0.07, right= 0.97)

    if filename:

        plt.savefig(os.path.join(base, save_dir, '{}.png'.format(filename)))

    return
