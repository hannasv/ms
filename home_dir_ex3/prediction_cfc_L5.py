import glob
import numpy as np
import xarray as xr
#import matplotlib.pyplot as plt
import os

lh_files = glob.glob('/global/D1/homes/hannasv/new_data/*2014*01*.nc')
print(len(lh_files))

given_name = 'AR-B-L1'
files = glob.glob('/home/hannasv/results_ar/AR-B-5/*weights*AR*L1*') 
print(len(files))
print('Merginging')
data = xr.open_mfdataset(files, combine='by_coords')
print('Completed')
data['latitude'] = data.latitude.values.astype(float)
data['longitude'] = data.longitude.values.astype(float)
data = data.sortby('longitude')
data = data.sortby('latitude')

lh_files = glob.glob('/global/D1/homes/hannasv/new_data/*2014*01*.nc')
print(len(lh_files))
ecc = xr.open_mfdataset(lh_files, combine='by_coords')
input_data = ecc.isel(time=slice(24, 24*2))
init_conditions = ecc.isel(time=slice(19, 24)) # alter for other lags

longitude = data.longitude.values
latitude = data.latitude.values

store_prediction = np.zeros((24, 161, 81))
coefs = data.coeffs.values

q_idx   = 0
t2m_idx = 1
r_idx   = 2
sp_idx  = 3
bias_idx = 4
tcc_idx = 5
print('restart')

for i in range(24):
    sub = input_data.isel(time=i)
    # HARDCODED DIFFERENT FOR ALL MODELS.
    Q = sub.q.values.T * coefs[:, :, 0] 
    print(Q.shape)
    T2M = sub.t2m.values.T * coefs[:, :, 1] 
    R = sub.r.values.T * coefs[:, :, 2] 
    SP = sub.sp.values.T * coefs[:, :, 3]
    BIAS = np.ones(sub.sp.values.T.shape) * coefs[:, :, 4]

    #subset = input_data.isel(time=i)
    
    if i==0:
        tcc = init_conditions.isel(time=0).tcc.values.T * coefs[:, :, 5]
        #tcc2 = init_conditions.isel(time=1).tcc.values.T * coefs[:, :, 6]
        #tcc3 = init_conditions.isel(time=2).tcc.values.T * coefs[:, :, 7]
        #tcc4 = init_conditions.isel(time=3).tcc.values.T * coefs[:, :, 8]
        #tcc5 = init_conditions.isel(time=4).tcc.values.T * coefs[:, :, 9]
    else:
        tcc = tcc * coefs[:, :, 5]
        #tcc2 = tcc1 * coefs[:, :, 6]
        #tcc3 = tcc2 * coefs[:, :, 7]
        #tcc4 = tcc3 * coefs[:, :, 8]
        #tcc5 = tcc4 * coefs[:, :, 9]
        
    tcc = np.nansum([Q, T2M, R, SP, BIAS, tcc], axis =0) #, tcc2, tcc3, tcc4, tcc5], axis = 0 )   
    #tcc = inverse_sigmoid(tcc)  
    #prev_tcc = np.nansum( np.stack([Q, T2M, R, SP, BIAS, tcc]), axis=0)
    store_prediction[i, :, :] = tcc
    print('added_data {}/24'.format(i))
    

data_dict = {'tcc': ([ 'sequence_length','longitude', 'latitude',], store_prediction)}
ds = xr.Dataset(data_dict,
         coords={'longitude': (['longitude'], longitude),
                 'latitude': (['latitude'], latitude),
                 'sequence_length': (['sequence_length'], np.arange(24))
                })
ds = ds.tcc.where(tcc < 10)
ds['date_seq']  = '2014-01-01'
ds.to_netcdf(os.path.join('/home/hannasv/predictions/','prediction_{}.nc'.format(given_name))) 
# obs file can't be opened anywhere.
