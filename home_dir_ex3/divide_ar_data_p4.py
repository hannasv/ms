import glob
import xarray as xr 
import os 
import numpy as np 

path = '/global/D1/homes/hannasv/new_data/'
files = glob.glob(path + '*.nc')

print('Detected {}'.format(len(files)))

data = xr.open_mfdataset(files, compat='no_conflicts')#, engine = 'h5netcdf')
print(data)
#tcc = glob.glob(path+'*tcc*.nc')
#print(len(tcc))
#sp = glob.glob(path+'*sp*.nc')
#print(len(sp))
#q = glob.glob(path+'*q*.nc')
#print(len(q))
#r = glob.glob(path+'*r*.nc')
#print(len(r))
#t2m = glob.glob(path+'*t2m*.nc')
#print(len(t2m))

# asser len not equal

#sp_data = xr.open_mfdataset(sp, combine='by_coords', engine = 'h5netcdf').reset_coords()
#print(sp_data)
#q_data  = xr.open_mfdataset(q, combine='by_coords', engine = 'h5netcdf').reset_coords()
#print(q_data)
#r_data = xr.open_mfdataset(r, combine='by_coords', engine = 'h5netcdf').reset_coords()
#print(r_data)
#t2m_data = xr.open_mfdataset(t2m, combine='by_coords', engine = 'h5netcdf').reset_coords()
#print(t2m_data)
#datasets = [xr.open_mfdataset(sp, compat = 'no_conflicts', engine = 'h5netcdf'), xr.open_mfdataset(q, compat = 'no_conflicts', engine = 'h5netcdf'), 
#            xr.open_mfdataset(r, compat = 'no_conflicts',  engine = 'h5netcdf'), xr.open_mfdataset(t2m, compat = 'no_conflicts',  engine = 'h5netcdf')]#     compat = 'no_conflicts')

#data = xr.open_mfdataset(tcc, combine='by_coords', engine = 'h5netcdf')
#print(data)
#for d2 in [sp_data, q_data, r_data, t2m_data]:
#    data = data.merge(d2, combine='by_coords')
#print(data)
#path = '/global/D1/homes/hannasv/data/'
#files = glob.glob(path + '*tcc*.nc')
#print('detected {} tcc files '.format(len(files)))
#tccdata = xr.open_mfdataset(files, compat='no_conflicts', engine = 'h5netcdf')
#print(tccdata)
save_dir ='/global/D1/homes/hannasv/ar_data/'

#print('len files on ar {}'.format(
#files = len(glob.glob(save_dir + '*.nc'))
#data = xr.open_dataset(files[0], engine = 'h5netcdf')
#print(data['t2m'])
#print(data['tcc'])
#print(data['sp'])
#print(data['r'])
#print(data['q'])

LAT = (48.5, 50)
LON = (-15,25)

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

for lat in latitude:
  for lon in longitude:
    fil = save_dir + 'all_vars_lat_lon_{}_{}.nc'.format(lat, lon)
    print('time for {}'.format(fil))
    if not os.path.exists(fil):
        #files.append(fil)
        subset = data.sel(latitude = lat, longitude = lon)
        try:
            subset.to_netcdf(fil, engine = 'netcdf4')#, encoding = e_dict)
        except PermissionError:
            print('can not access fil')
    print('finished lat {}, lon {}'.format(lat, lon))

