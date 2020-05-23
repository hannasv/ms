import os
import glob
import xarray as xr
import numpy as np

store_dir = '/uio/lagringshotell/geofag/students/metos/hannasv/satelite_coordinates/msthesis/'

coords = xr.open_dataset(store_dir + 'input_grid_cloud_mask.nc')

lat_array = coords.latitude.values  # 2 dimensional array
lon_array = coords.longitude.values # 2 dimensional array
print('prior')
print(np.shape(lat_array))
lat_array[lat_array < -99] = np.nan # updates of disk values to nan
lon_array[lon_array < -99] = np.nan # updates of disk values to nan
print(np.shape(lat_array))
d_phi   = np.zeros(np.shape(lat_array))
d_theta = np.zeros(np.shape(lon_array))

d_phi2   = np.zeros(np.shape(lat_array))
d_theta2 = np.zeros(np.shape(lon_array))

for i in range(1, 3711):
    for j in range(1, 3711):
        d_phi[i][j] = ( np.abs(lon_array[i-1][j]) - np.abs(lon_array[i+1][j]) )/4
        d_theta[i][j] = ( np.abs(lat_array[i][j-1]) - np.abs(lat_array[i][j+1]) )/4

        d_phi2[i][j] = ( np.abs(lon_array[i][j-1]) - np.abs(lon_array[i][j+1]) )/4
        d_theta2[i][j] = ( np.abs(lat_array[i-1][j]) - np.abs(lat_array[i+1][j]) )/4


lat_index = np.arange(3712)
lon_index = np.arange(3712)
# pad zeroes around the edges.

dix = {'d_phi':(['latitude', 'longitude'], d_phi),
      'd_theta':(['latitude', 'longitude'], d_theta),
      'd_phi2':(['latitude', 'longitude'], d_phi2),
      'd_theta2':(['latitude', 'longitude'], d_theta2),
      'lat': (['latitude', 'longitude'], lat_array),
      'lon': (['latitude', 'longitude'], lon_array),
      }

result = xr.Dataset(dix,
                    coords={'longitude': (['longitude'], lon_index),
                            'latitude': (['latitude'], lat_index),
                            })
result.to_netcdf(os.path.join(store_dir, 'changes_lat_lon_for_plot2.nc'))
