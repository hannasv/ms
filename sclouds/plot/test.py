import glob
import xarray as xr
import os
import numpy as np

path = '/global/D1/homes/hannasv/new_data/'
files = glob.glob(path + '*.nc')

print('Detected {}'.format(len(files)))

data = xr.open_mfdataset(files, compat='no_conflicts')

save_dir ='/global/D1/homes/hannasv/corrupt/'
pairs = [(37.25, 16.25), (37.75, 10.5)]
#LAT = (43.5, 45)
#LON = (-15, 25)

SPATIAL_RESOLUTION = 0.25

latitude = np.arange(LAT[0], LAT[1]+SPATIAL_RESOLUTION,
                        step = SPATIAL_RESOLUTION)
longitude = np.arange(LON[0], LON[1]+SPATIAL_RESOLUTION,
                        step = SPATIAL_RESOLUTION)
#print(latitude)
#print(longitdue)
e_dict = {'t2m':{'compression': 'gzip', 'compression_opts': 9},
          'tcc':{'compression': 'gzip', 'compression_opts': 9},
          'sp':{'compression': 'gzip', 'compression_opts': 9},
          'r':{'compression': 'gzip', 'compression_opts': 9},
          'q':{'compression': 'gzip', 'compression_opts': 9},
          'nr_nans':{'compression': 'gzip', 'compression_opts': 9}}

for lat, lon in pairs:
    fil = save_dir + 'all_vars_lat_lon_{}_{}.nc'.format(lat, lon)
    print('time for {}'.format(fil))
    if not os.path.exists(fil):
        #files.append(fil)
        subset = data.sel(latitude = lat, longitude = lon)
        try:
            subset.to_netcdf(fil)#, encoding = e_dict)
        except PermissionError:
            print('can not access fil')
    print('finished lat {}, lon {}'.format(lat, lon))
