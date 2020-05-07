
# For margins of 6cm (my thesis)
TEXT_WIDTH_CM  = 15.5
TEXT_HEIGHT_CM = 24.7

TEXT_WIDTH_IN  = 6.1023622
TEXT_HEIGHT_IN = 9.72440945

ANGLE_ROTATION = 45

cmap_contour_plot = 'BuGr_r'
levels_contourplot = 100

color_maps = {'tcc'  : 'Blues_r', 'sp'   : 'winter', 'q': 'pink',
                'r': 'copper', 't2m'  : 'coolwarm'}

path_python_figures = '/home/hanna/MS-thesis/python_figs/'
file_format = 'pdf'

SPACEING = {'hspace' : 0.5, 'top':0.97, 'bottom':0.03, 'left' : 0.125, 'right' : 0.97}

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
