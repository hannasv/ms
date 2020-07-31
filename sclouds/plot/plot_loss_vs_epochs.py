import os
import glob
import json

import matplotlib.pyplot as plt
import xarray as xr
import numpy as np

def get_array(data, key = 'loss'):
    return [data[key][k] for k in data[key].keys()]

test_models = glob.glob('/home/hanna/EX3_Results/*')
print(test_models)
fig, axes = plt.subplots(2, 1, figsize = (6.1023622, 6.1023622), sharex=True)
num_converged_models = 0
for path in test_models:
    try:
        with open(path + '/history.json') as json_file:
            data = json.load(json_file)

        model = path.split('/')[-1]
        num_converged_models +=1
        # summarize history for accuracy
        axes[0].plot(get_array(data, key = 'loss'), label = model)
        axes[1].plot(get_array(data, key = 'val_loss'), label = model)
    except FileNotFoundError:
        print('file not found for model {}'.format(path))
    except NotADirectoryError:
        print('not a directory')
axes[0].set_ylabel('Training Loss')
axes[1].set_ylabel('Validation Loss')
axes[1].set_xlabel('Epoch')
#axes[0].set_xlabel('Epoch')
#plt.legend(ncol = 1, frameon = True, bbox_to_anchor=(1.3, 2.7))

axes[0].legend(ncol = 1, frameon = False, loc='upper right', # bbox_to_anchor=(x0, y0, width, height),
            edgecolor='w', fontsize='small', framealpha=0.5, #frameon = False,
            borderaxespad=0)
#axes[0].grid(True)
#axes[1].grid(True)
#plt.legend(['train', 'test'], loc='upper left')
plt.subplots_adjust(wspace = 0.2, hspace = 0.3, top=0.98, bottom=0.15, left = 0.14, right =.98)
#plt.subplots_adjust(wspace = 0.3, hspace = 0.2, top= 0.8, bottom= 0.1, left= 0.1, right= 0.8)
plt.savefig(os.path.join('/home/hanna/MS-thesis/python_figs/', 'epoch_vs_loss.pdf'))
