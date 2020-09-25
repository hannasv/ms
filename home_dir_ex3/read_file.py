import xarray as xr
fil = '/global/D1/homes/hannasv/ar_data/all_vars_lat_lon_35.0_-15.0.nc'
print(xr.open_dataset(fil))
