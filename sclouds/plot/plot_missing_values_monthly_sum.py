import os
import glob

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
                                  file_format, autolabel)

matplotlib = import_matplotlib()

def get_path(year, month, base = '/home/hanna/lagrings/ERA5_monthly/'):
    month ="%2.2d" %month # includng leading zeros.
    search_str = '{}*{}*tcc.nc'.format(year, month)
    if not (year==2005 and month == 4):
        return glob.glob(os.path.join(base, search_str))
    else:
        return ['/home/hanna/MS/sclouds/io/2005_04_tcc.nc']

def get_missing_hours(year, month):
    files = get_path(year, month)
    print(files)
    if len(files) == 0:
        print("year: {}, month: {}".format(year, month))
        return np.nan
    else:
        fil = files[0]
        if month < 10:
            month1 = "%2.2d" %month
            month2 = "%2.2d" %(month+1)
            year2 = year

        elif month == 12:
            year2 = year+1
            month1 = month
            month2="01"
        else:
            month1 = month
            month2 = month + 1
            year2  = year
        if year==2005 and month == 4:
            fil = '/home/hanna/MS/sclouds/io/2005_04_tcc.nc'
        data = xr.open_dataset(fil) # , decode_times = False
        #print(data)
        #print(data)
        start = '{}-{}-01'.format(year, month1)
        stop = '{}-{}-01'.format(year2, month2)

        timearray = np.arange(start, stop, np.timedelta64(1,'h'), dtype='datetime64[ns]')
        #print(len(timearray))
        ll = data.time.values.astype(np.datetime64)

        counter = 0
        for element in timearray:
            if element not in ll:
                counter += 1

        assert len(timearray) >= counter, "how, start {}, stop {}, "\
                    "len timearray {}, counter {}".format(start, stop, len(timearray), counter)
        return counter

years = np.arange(2004, 2019)
months = np.arange(1, 13)
sum = []

for m in months:
    tempsum = []
    for y in years:
        missing = get_missing_hours(y, m)
        tempsum.append(missing)
    sum.append(np.nansum(tempsum))

#fig, ax = plt.subplots(1, 1, figsize = (TEXT_WIDTH_IN, 0.5*TEXT_WIDTH_IN))
month = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
         'August', 'September', 'October', 'November', 'December']

fig, ax = plt.subplots(1, 1, figsize = (TEXT_WIDTH_IN, 0.5*TEXT_WIDTH_IN) )
rect = ax.bar(np.arange(len(month)), np.array(sum).astype(int), align = 'center',
                alpha = 0.9, color = 'yellow')
ax = autolabel(rect, ax)
plt.xticks(np.arange(len(month)), rotation=45)
ax.set_xticklabels(month)
ax.set_ylim([0, np.max(sum) + 20])
plt.title('Number of Missing Hours')
plt.ylabel('Number of Missing Hours')
plt.subplots_adjust(hspace = 0.2, top=0.9, bottom=0.3, left = 0.15, right = 0.95)
fig.savefig(path_python_figures + 'NEW_heatmap_missing_values_monthly_sum.{}'.format(file_format))
