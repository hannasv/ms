import os
import glob

import numpy as np
import xarray as xr

from sclouds.plot.helpers import TEXT_HEIGHT_IN, TEXT_WIDTH_IN, import_matplotlib
mat = import_matplotlib()
import matplotlib.pyplot as plt


from sclouds.ml.regression.AR_model_loader import AR_model_loader
print(AR_model_loader)
results_dir = '/home/hanna/lagrings/results/ar/'


test_fil = '/home/hanna/lagrings/results/ar/AR_2020-05-18T08:30:30.nc'
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
    print(m1[key])
    var = ['Wt2m', 'Wsp', 'Wr', 'Wq']
except KeyError:
    #print('failed ...')
    var = []

ORDER = m1['order'].values

per = va + var

if m1['bias'].values:
    per += ['bias']

fig, axes = plt.subplots(len(per)+order, 1, figsize = (TEXT_WIDTH_IN, TEXT_HEIGHT_IN))
for i, p in enumerate(per):
    m1[p].plot(ax = axes[i])

fig.suptitle('Sig:{}, Trans:{}, bias:{}, order:{}'.format( m1['sigmoid'].values, m1['transform'].values, m1['bias'].values, m1['order'].values ))
plt.savefig(os.path.join(save_dir, 'weights.png'))
