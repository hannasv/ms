print('started')
import glob
import xarray as xr 

files = glob.glob('/home/hannasv/results_ar/AR-5/*performance*TR*L1*')
print('len files {}'.format(len(files)))
data = xr.open_mfdataset(files, combine='by_coords')

data['latitude'] = data.latitude.values.astype(float)
data['longitude'] = data.longitude.values.astype(float)
data = data.sortby('longitude')
data = data.sortby('latitude')
print(data)
#data.to_netcdf('/home/hannasv/STORE_AGGREGATED_RESULTS/AR-S-L0.nc')
print('stored to disk')
