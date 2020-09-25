import glob
import numpy as np
import xarray as xr
import os


lh_files = glob.glob('/global/D1/homes/hannasv/new_data/*2014*01*.nc')

prediction = np.nan*np.ones((24, 81, 161)) #  # (sequence, latitude, longitude)
input_data = xr.open_mfdataset(lh_files, combine='by_coords').sel(time='2014-01-01')
longitude  = np.arange(-15.0, 25.0+0.25, step = 0.25)
latitude   =  np.arange(30.0, 50.0+0.25, step = 0.25)

for j, lat in enumerate(latitude):
    for k, lon in enumerate(longitude):
        print('Working on lat {}, lon {}'.format(lat, lon))
        files = glob.glob('/home/hannasv/results_ar_stripped/AR-B-5/*weights*L1*_{}_*{}.nc*'.format(lon, lat))
        data = xr.open_dataset(files[0])
        coefs = data.coeffs.values
        inp = input_data.sel(latitude=lat, longitude=lon)
        
        for i in range(24):
            
            sub = inp.isel(time=i)
            if i==0:
                # bruk første til å lage prediksjonen. Burde jeg brukt nyyåtsaften til å sette i gang prediksjonen..?
                prev_tcc = float(sub.tcc.values)

            arr = np.array([float(sub.q.values), float(sub.t2m.values), float(sub.r.values), float(sub.sp.values), 1, prev_tcc])  
            prev_tcc = arr.T@coefs     
            prediction[i, j, k] = prev_tcc

    data_dict = {'tcc': (['sequence_length', 'latitude', 'longitude'], prediction)}

    ds = xr.Dataset(data_dict,
             coords={'longitude': (['longitude'], longitude),
                     'latitude': (['latitude'], latitude),
                     'sequence_length': (['sequence_length'], np.arange(24))
                    })
    ds['date_seq']  = '2014-01-01'
    print(ds)
    ds.to_netcdf(os.path.join('/home/hannasv/','ARprediction.nc'))

