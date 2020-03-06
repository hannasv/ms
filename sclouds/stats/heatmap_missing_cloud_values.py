import xarray as xr
import numpy as np
from datetime import datetime, timedelta
import glob
import os

#read_clouds = '/home/hannasv/Desktop/miphclac/'
#store_p = '/home/hannasv/Desktop/lagrings/from_simula/'
read_data = '/home/hannasv/Desktop/lagrings/'
test = '2005-05'

data_folder = 'ERA5_monthly'
var = 'tcc'
ll = os.path.join( read_data, data_folder, "*{}*.nc".format(var))
files = glob.glob(ll)

test_fil = files[0]
data = xr.open_dataset(test_fil)

# Get year and month from file.

# Make heatmap with the numbers of hours missing every year.
# Not sure what is best on each axis.


missing_dict = {}

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
                  timedelta(hours=1)).astype(datetime)

    inc_values = data.time.values

    #key = '{}_{}'.format(y, m)
    diff = len(t) - len(inc_values)
    try:
        o = missing_dict[str(y)]
    except KeyError:
        missing_dict[str(y)] = {}

    missing_dict[str(y)][str(m)] = diff

import pandas as pd
kk = pd.DataFrame(missing_dict)
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(kk)
plt.show()
# 1) Need to add file to existing dataframe








