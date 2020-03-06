import xarray as xr
import numpy as np
from datetime import datetime, timedelta
import glob
import os

#read_clouds = '/home/hannasv/Desktop/miphclac/'
#store_p = '/home/hannasv/Desktop/lagrings/from_simula/'
read_data = '/home/hannasv/Desktop/lagrings/'
test = '2005-05'
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


data_folder = 'ERA5_monthly'
var = 'tcc'
ll = os.path.join( read_data, data_folder, "*{}*.nc".format(var))
files = glob.glob(ll)

test_fil = files[0]
data = xr.open_dataset(test_fil)

# Get year and month from file.

# Make heatmap with the numbers of hours missing every year.
# Not sure what is best on each axis.


out_of_sample_values = []
# 2) Need to figure out how many is missing.
for f in files:
    print("Compuiting file {}".format(f))
    data = xr.open_dataset(f)
    if f == '/home/hannasv/Desktop/lagrings/ERA5_monthly/2007_11_tcc_tcc.nc':
        y = 2007
        m = 11
    else:
        year, month, _ = f.split('/')[-1].split('_')
        y = int(year)
        m = int(month)
    # Arange doesn't include the last one.
    if m < 12:
        end_date = datetime(y, m+1, 1)
    else:
        end_date = datetime(y+1, 1, 1)

    t = np.arange(datetime(y, 1, 1),
                  end_date,
                  timedelta(hours=1)).astype('datetime64[h]')
    inc_values = data.time.values.astype('datetime64[h]')


    min_bound = np.min(t)
    max_bound = np.max(t)

    if (np.min(inc_values) < min_bound) or (np.max(inc_values) > max_bound):
        out_of_sample_values.append('{}_{}'.format(y, m))


# 1) TODO : Need to add file to existing dataframe - can do this on metos..
# Based on the new filename. Find regridded file to read.

# Read that file
data = xr.open_dataset(f)

# Make the dataset contianing the new file
new_ds = xr.Dataset({'tcc': (['latitude', 'longitude'], cloud_fraction),
                 'nr_nans': (['latitude', 'longitude'], nans),
                 # 'nr_cells':(['latitude', 'longitude'], cnt_cells)
                 },
                coords={'longitude': (['longitude'], lon),
                        'latitude': (['latitude'], lat),
                        })

# this returnes None.
data.merge(new_ds)




