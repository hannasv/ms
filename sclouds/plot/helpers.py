
# For margins of 6cm (my thesis)
TEXT_WIDTH_CM  = 15.5
TEXT_HEIGHT_CM = 24.7

TEXT_WIDTH_IN  = 6.1023622
TEXT_HEIGHT_IN = 9.72440945

# (16:9) slides have a size of 13.33 x 7.5
PP_WIDTH_IN = 13.33
PP_HEIGHT_IN = 7.5

## FontSizes on PP
def import_matplotlib_pp():
    import matplotlib
    matplotlib.use("pgf") # use pgf as a backend
    print('Warning.. Using pgf backend, no GUI available. use plt.savefig() for inpection')
    matplotlib.rcParams.update({
        #"pgf.texsystem": "lualatex",
        'font.family': 'serif',
        'font.size': 28,
        'legend.fontsize': 28,
        'figure.titlesize': 36,
        'axes.titlesize':32,
        #'text.usetex': True,
        'pgf.rcfonts': False,
    })
    return matplotlib


ANGLE_ROTATION = 45

cmap_contour_plot = 'BuGr_r'
levels_contourplot = 100

color_maps = {'tcc'  : 'Blues_r', 'sp'   : 'winter', 'q': 'pink',
                'r': 'copper', 't2m'  : 'coolwarm'}

path_python_figures = '/home/hanna/MS-thesis/python_figs/'
file_format = 'pdf'

SPACEING = {'hspace' : 0.5, 'top':0.97, 'bottom':0.03, 'left' : 0.125, 'right' : 0.97}

def autolabel(rects, ax):
    """ Attach a text label above each bar in *rects*, displaying its height.
    Parameters
    --------------------------
    rect : returned from ax.bar
        The square you want to annotate.
    ax : matplotlib axis
    """
    for rect in rects:
        height = rect.get_height()
        if height < 0:
            rect.set_color('red')

        vert_offset = 3
        ax.annotate(height,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, vert_offset),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    return ax

def import_matplotlib():
    import matplotlib
    matplotlib.use("pgf") # use pgf as a backend
    print('Warning.. Using pgf backend, no GUI available. use plt.savefig() for inpection')
    matplotlib.rcParams.update({
        "pgf.texsystem": "lualatex",
        'font.family': 'serif',
        'font.size': 12,
        'legend.fontsize': 'small',
        'figure.titlesize': 14,
        'axes.titlesize':14,
        'text.usetex': True,
        'pgf.rcfonts': False,
    })
    return matplotlib

def add_ticks(ax, x_num_tikz = 20, y_num_tikz = 15, rotation = False):
    """ Add ticks to axes ...
    """
    import numpy as np
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

    ax.set_yticks(yticks);
    ax.set_xticks(xticks);

    if rotation:
        ax.set_xticklabels(xticks_labels.astype(int), rotation = 90);
        ax.set_yticklabels(yticks_labels.astype(int), rotation = 90);
    else:
        ax.set_xticklabels(xticks_labels.astype(int));
        ax.set_yticklabels(yticks_labels.astype(int));

    return ax

def save_test_ar(test_fil):
    import os
    import glob

    import numpy as np
    import xarray as xr

    from sclouds.plot.helpers import TEXT_HEIGHT_IN, TEXT_WIDTH_IN, import_matplotlib
    mat = import_matplotlib()
    import matplotlib.pyplot as plt


    from sclouds.ml.regression.AR_model_loader import AR_model_loader

    m1 = AR_model_loader().load_model_to_xarray(test_fil)

    save_dir = '/home/hanna/MS-thesis/python_figs/test/{}'.format(test_fil.split('.nc')[0].split('/')[-1])

    if not os.path.isdir(save_dir):
        try:
            os.makedirs(save_dir, exist_ok = True)
            #print("Directory '%s' created successfully" %save_dir)
        except OSError as error:
            #print("Directory '%s' can not be created")
            print(error)

    per = ['mse', 'ase', 'r2']
    fig, axes = plt.subplots(len(per), 1, figsize = (TEXT_WIDTH_IN, TEXT_HEIGHT_IN))
    for i, p in enumerate(per):
        m1[p].plot(ax = axes[i])
    fig.suptitle('Sig:{}, Trans:{}, bias:{}, order:{}'.format( m1['sigmoid'].values, m1['transform'].values, m1['bias'].values, m1['order'].values ))
    plt.savefig(os.path.join(save_dir, 'performance.png'))

    per = ['num_train_samples', 'num_test_samples']
    fig, axes = plt.subplots(len(per), 1, figsize = (TEXT_WIDTH_IN, TEXT_HEIGHT_IN))
    for i, p in enumerate(per):
        m1[p].plot(ax = axes[i])
    fig.suptitle('Sig:{}, Trans:{}, bias:{}, order:{}'.format( m1['sigmoid'].values, m1['transform'].values, m1['bias'].values, m1['order'].values ))
    plt.savefig(os.path.join(save_dir, 'num_samples.png'))

    va = []
    for i in range(1, +1):
        va.append('W{}'.format(i))

    try:
        print(m1['Wt2m'])
        var = ['Wt2m', 'Wsp', 'Wr', 'Wq']
    except KeyError:
        #print('failed ...')
        var = []

    order = m1['order'].values

    per = va + var

    if m1['bias'].values:
        per += ['bias']

    fig, axes = plt.subplots(len(per)+order, 1, figsize = (TEXT_WIDTH_IN, TEXT_HEIGHT_IN))
    for i, p in enumerate(per):
        m1[p].plot(ax = axes[i])

    fig.suptitle('Sig:{}, Trans:{}, bias:{}, order:{}'.format( m1['sigmoid'].values, m1['transform'].values, m1['bias'].values, m1['order'].values ))
    plt.savefig(os.path.join(save_dir, 'weights.png'))
    return
