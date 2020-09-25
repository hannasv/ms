
import glob
import numpy as np
import xarray as xr

# Read data

longitude  = np.arange(-15.0, 25.0+0.25, step = 0.25)
latitude   =  np.arange(30.0, 50.0+0.25, step = 0.25)

storage = np.ones((len(latitude), len(longitude)))*np.nan

# lopp over lat long?

for i, lon in enumerate(longitude):
    for j, lat in enumerate(latitude):
        print('On lat {} lon {} nr {}'.format(lat, lon, (i+1)*(j+1)))
        try:
            fil = glob.glob('/global/D1/homes/hannasv/ar_data/*{}*{}*.nc'.format(lat, lon))[0]
            splits = fil.split('_')
            lat = splits[-2]
            lon = splits[-1][:-3]

            try:
                data = xr.open_dataset(fil, engine = 'h5netcdf')
                status = True
                try:
                    vals = data.sp.values
                except OSError as e:
                    print('Detected corrupt file. {}'.format(e))
                    status = False

            except Exception as e:
                status = False

        except IndexError:
            status = False

        storage[j, i] = status
data_dict = {'status': (['latitude', 'longitude'], storage)}

longitude  = np.arange(-15.0, 25.0+0.25, step = 0.25)
latitude   =  np.arange(30.0, 50.0+0.25, step = 0.25)
#seq_length = sequence_prediction
#time = np.arange(seq_length)
#print(sequence_prediction[:, :, :, :, 0].shape)
ds = xr.Dataset(data_dict,
                coords={'longitude': (['longitude'], longitude),
                        'latitude': (['latitude'], latitude)
                       })
ds.to_netcdf('status_datafiles.nc', engine = 'h5netcdf')

