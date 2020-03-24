import xarray as xr
import numpy as np
import os

import matplotlib.pyplot as plt

file = '/home/hanna/miphclac/2004_07/2004_07.nc'
data = xr.open_dataset(file)

ll = data.sel(time = '2004-07-02T12')
ll['tcc'].plot()
plt.show()
