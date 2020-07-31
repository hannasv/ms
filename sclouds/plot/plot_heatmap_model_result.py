# create this plotting routine
import os
import glob
import json

import numpy as np
import xarray as xr
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
#from sclouds.plot.helpers import TEXT_WIDTH_IN, path_python_figures
from sclouds.helpers import (path_input, path_stats_results, VARIABLES,
                             UNITS, LONGNAME)
from sclouds.plot.helpers import (TEXT_WIDTH_IN, TEXT_HEIGHT_IN,
                                  path_python_figures, import_matplotlib,
                                  file_format)

matplotlib = import_matplotlib()

data = {"AR-B-5-L5": {"mae_train": [np.nan, 671.3465085862642, 663.168293601844, 659.7584775029061, 658.3037892026349, 657.5310568623834],
"mae_test": [np.nan, 650.5521953120024, 643.6644140773706, 640.6431257749405, 639.2869404499774, 638.5771301517702],
"num_trained_models": [13041, 13041, 13041, 13041, 13041, 13041]},
"TR-B-5-L5": {"mae_train": [np.nan, 1959.9831871359675, 1931.7075512039257, 1919.1626480225382, 1913.2349780199302, 1909.8078351281385],
"mae_test": [np.nan, 1927.8631768704374, 1900.9120094325754, 1889.1204821526762, 1883.5870120414293, 1880.4800433776545],
"num_trained_models": [0, 13041, 13041, 13041, 13041, 13041]},
"AR-T-5-L5": {"mae_train": [np.nan, 5482.146935013481, np.nan, 5408.834698734015, 5372.999894682853, 5338.918759492995],
"mae_test": [np.nan, 5780.75499021119, np.nan, 5723.062510722784, 5694.592531966111, 5667.046378989056],
"num_trained_models": [13041, 13041, 13041, 13041, 13040, 13040]},
"TR-T-5-L5": {"mae_train": [np.nan, 7237.10965912469, 7196.273804414343, 7156.34854913884, 7118.479446227317, 7084.163338551019],
"mae_test": [np.nan, 7383.201724831454, 7345.783088126037, 7309.530899036785, 7275.469940927508, 7244.630546102634],
"num_trained_models": [0, 13041, 13041, 13041, 13040, 13040]},
"AR-S-5-L5": {"mae_train": [np.nan, 556.7622472798296, 549.6607018608044, 546.6659539025945, 545.3686834384089, 544.6196384511707],
"mae_test": [np.nan, 539.2984977496845, 533.3375826575414, 530.6971953455991, 529.4967061592761, 528.8116791245442],
"num_trained_models": [13041, 13041, 13041, 13041, 13041, 13040]},
"TR-S-5-L5": {"mae_train": [np.nan, 1574.1236439763045, 1570.499736501391, 1574.2517519538246, 1579.4245482568972, 1584.2910028883505],
"mae_test": [np.nan, 1558.1370023510692, 1553.4838482922603, 1556.4713063636857, 1561.0448658234614, 1565.5426291689096],
"num_trained_models": [0, 13041, 13041, 13041, 13041, 13040]},
"AR-B-S-5-L5": {"mae_train": [np.nan, 554.8318252779043, 548.0729699188793, 545.2549400850462, 544.0060873718264, 543.367522223059],
"mae_test": [np.nan, 537.6464424066213, 531.9540612209616, 529.4571287396225, 528.2871190379533, 527.7005992577199],
"num_trained_models": [13041, 13041, 13041, 13041, 13040, 13040]},
"TR-B-S-5-L5": {"mae_train": [np.nan, np.nan, 1756.0977738217423, 1744.6933163841286, 1739.1591600497368, 1736.0438767468108],
"mae_test": [np.nan, np.nan, 1728.101826756878, 1717.382256502436, 1712.2007344533545, 1709.3764638767989],
"num_trained_models": [0, 13041, 13041, 13041, 13040, 13040]},
"AR-T-S-5-L5": {"mae_train": [np.nan, 4530.699946292132, 4499.698778068336, 4470.111321267781, 4440.813329443834, np.nan],
"mae_test": [np.nan, 4777.4834629844545, 4753.2062788334115, 4729.80372787007, 4706.611543381248, np.nan],
"num_trained_models": [13041, 13041, 13041, 13041, 13041, 13041]},
"TR-T-S-5-L5": {"mae_train": [np.nan, 6579.190599204263, 6542.067094922129, 6505.7714083080355, 6471.820554035821, 6440.621011132292],
"mae_test": [np.nan, 6712.001568028596, 6677.9846255691255, 6645.02809003344, 6614.542971717883, 6586.504345818614],
"num_trained_models": [0, 13041, 13041, 13041, 13041, 13041]},
"AR-5-L5": {"mae_train": [1947.4894137813267, 673.6823192085939, 665.0894492515732, 661.4658042221395, 659.8961069604748, 659.0486844321945],
"mae_test": [1920.1493975150174, 652.551182277117, 645.3384750156258, 642.1436063681756, 640.6910144527233, 639.9186085676238],
"num_trained_models": [13041, 13041, 13041, 13041, 13041, 13041]},
"TR-5-L5": {"mae_train": [np.nan, np.nan, 1727.549710151542, 1731.6769271492224, 1737.3670030825883, 1742.8781048295596],
"mae_test": [np.nan, np.nan, 1708.832233121499, 1712.1184370000701, 1717.1493524058092, 1722.2531325013183],
"num_trained_models": [0, 13041, 13041, 13041, 13041, 13041]}}


# For when actually reading the json file.
#with open('/home/hanna/EX3_AR_SUMMARY/ar_mae_performance.json') as f:
    #data = json.load(f)
#print(data)

mae = 'num_trained_models'
mae = 'mae_test'

for mae in ['num_trained_models', 'mae_test']:
    subset = {}
    for key, item in data.items():
        key = key.split('-5-')[0]
        subset[key] = item[mae]*43824
    df = pd.DataFrame.from_dict(subset).transpose()

    fig, ax = plt.subplots(1, 1, figsize = (TEXT_WIDTH_IN, 0.5*TEXT_WIDTH_IN) )

    ax = sns.heatmap(df, linewidths=0.1, annot=True, # fmt="d",
                cbar_kws={'label':'{}'.format(mae)}, ax = ax, cmap = 'viridis', fmt = '5.2f')

    ax.set_ylabel('Model')
    ax.set_xlabel('Lag')

    plt.subplots_adjust(hspace = 0.2, top=0.97, bottom=0.20, left = 0.15, right = 1.0)
    fig.savefig(path_python_figures + 'heat_ar_model_{}_score.png'.format(mae))
    print('Finished {}'.format(mae))
