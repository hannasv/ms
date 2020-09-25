import xarray as xr
import numpy as np

import glob
import os


path = '/global/D1/homes/hannasv/data/'
files = glob.glob(path + '*.nc')

print('Detected {}'.format(len(files)))

data = xr.open_mfdataset(files, compat='no_conflicts', engine = 'h5netcdf')
print(data)
save_dir = '/global/D1/homes/hannasv/ar_data/' 

LAT = (30,50)
LON = (-15,25)

SPATIAL_RESOLUTION = 0.25

latitude = np.arange(LAT[0], LAT[1]+SPATIAL_RESOLUTION,
                        step = SPATIAL_RESOLUTION)
longitude = np.arange(LON[0], LON[1]+SPATIAL_RESOLUTION,
                        step = SPATIAL_RESOLUTION)

e_dict = {'t2m':{'compression': 'gzip', 'compression_opts': 9}, 
          'tcc':{'compression': 'gzip', 'compression_opts': 9}, 
          'sp':{'compression': 'gzip', 'compression_opts': 9}, 
          'r':{'compression': 'gzip', 'compression_opts': 9}, 
          'q':{'compression': 'gzip', 'compression_opts': 9}, 
          'nr_nans':{'compression': 'gzip', 'compression_opts': 9}}

for lat in latitude:
  for lon in longitude:
    subset = data.sel(latitude = lat, longitude = lon)
    subset.to_netcdf(save_dir + 'all_vars_lat_lon_{}_{}.nc'.format(lat, lon), engine = 'h5netcdf', encoding = e_dict)
  
