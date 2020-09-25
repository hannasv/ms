import xarray as xr
from sclouds.ml.regression.utils import dataset_to_numpy_order
fil = '/global/D1/homes/hannasv/ar_data/all_vars_lat_lon_40.5_1.5.nc'
fil = '/global/D1/homes/hannasv/ar_data/all_vars_lat_lon_37.25_16.25.nc'

data = xr.open_dataset(fil).sel(time = slice('2004', '2013'))
X, y = dataset_to_numpy_order(dataset = data.sel(time = slice('2004', '2013')),
                        order = 1,  bias = False)

print(X)
print(y)
