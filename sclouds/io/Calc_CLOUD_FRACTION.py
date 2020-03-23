#!/usr/bin/env python
# coding: utf-8

# ## Calculate Cloud Fractions

# In[1]:

import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import xarray as xr

import glob
import xarray as xr
import numpy as np

data_dir = '/home/hanna/MS-suppl/'
path = '/home/hanna/miphclac/'

def generate_save_dir(year, month):
    path = '/home/hanna/miphclac/'
    return os.path.join(path, '{}_{:02d}'.format(year, month))

import logging
LOG_FILENAME = 'example.log'
logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG)

logging.debug('This message should go to the log file')

def clean_file(satfil):
    """Removing the land, sea mask"""
    if satfil.split('.')[-1] == 'grb':
        try:
            cloudMask = xr.open_dataset(satfil, engine = 'cfgrib')
        except EOFError as e:
            logging.debug(e)
            return None

        o = np.fliplr(cloudMask['p260537'].values.reshape( (3712, 3712) ))

        o[o>=3.0]=np.nan
        o[o==1.0]=0
        o[o==2.0]=1.0
    else:
        cloudMask = xr.open_dataset(satfil)
        o = cloudMask['cloudMask'].values.reshape( (3712, 3712) )
        o[o>=3.0]=np.nan
        o[o==1.0]=0
        o[o==2.0]=1.0
    return o

def area_grid_cell(c_lat, d_lat, d_lon):
    """Get area for one pixel. """
    R = 6371000  # in M
    # area er egentlig R**2
    area = R*(np.sin((c_lat + d_lat)*np.pi/180) - np.sin((c_lat - d_lat)*np.pi/180) )*(2*d_lon*np.pi/180) # R**2
    return np.abs(area)

def get_dict_with_all_keys():
    data_dir = '/home/hanna/MS-suppl/'
    ex_fil = glob.glob(os.path.join(data_dir, 'files', '*ERA5*.json'))
    assert len(ex_fil) != 0, 'Have you cloned the data suppplementary repository?'
    #print(ex_fil)
    merged_dict = {}

    for fil in ex_fil:
        with open(fil, 'r') as f:
            data_grid = json.load(f)
        merged_dict.update(data_grid)

    return merged_dict

#data_dict = get_dict_with_all_keys()

def calc_fraction_one_cell(lat = '30.25', lon = '19.25', cmk = None, data = None):

    if data:
        ## Improvements : This should read the files.
        ex = data[lat][lon]
        fraction = 0.0
        SAT_area = 0.0

        for key, item in ex.items():
            """Loops over all corners"""
            index = ex[key]['index']
            area  = ex[key]['area']

            if len(index) == len(area):
                fraction += np.nansum(np.array(area)*np.array(cmk[index]) )
                SAT_area += np.nansum(area)
            else:
                print('Returns nan for lat {}, lon {}'.format(lat, lon))
                #return np.nan, (lat, lon)

        return fraction/SAT_area, np.isnan(np.array(cmk[index])).sum()
    else:
        print('Please send data as a attribute.')
        return

def compute(satfil, lats = np.arange(30.0, 50.25, 0.25), lons = np.arange(-15.0, 25.25, 0.25) ):

    o = clean_file(satfil)

    if o is not None:
        #d_phi, d_theta, cell_areas, lat_array, lon_array = read_dlon_dlat(data_dir)
        clouds = o.reshape(-1)

        data_dict = get_dict_with_all_keys()

        GRID = np.zeros((len(lats), len(lons)) )
        GRID_NAN = np.zeros((len(lats), len(lons)) )

        for i,lat in enumerate(lats):
            for j, lon in enumerate(lons):
                fraction, nbr_nan = calc_fraction_one_cell(lat = str(lat),
                                                           lon = str(lon),
                                                           cmk = clouds,
                                                           data = data_dict)

                GRID[i][j] = fraction
                GRID_NAN[i][j] = nbr_nan

        return GRID, GRID_NAN
    else:
        return None, None

def timestamp(filename):
    """
    Returns the numpy datetime 64 [ns] for the current date.
    This is a bit hardcoded at the moment ....
    """
    splits = filename.split('-')
    ts = splits[5]
    year = ts[:4]
    month = ts[4:6]
    day = ts[6:8]
    hr = ts[8:10]
    minuts = ts[10:12]
    sek = ts[12:14]
    # TODO make sure all filenames have seconds
    return np.datetime64( year+"-"+month+"-"+day+"T"+hr ) # +":"+minuts+":"+sek

def make_folder_str(year, month):
    """ Generates the folder search str
    month : int
    year : int

    Returns : str
        year_month
    """

    month = "%2.2d" % month
    return "{}_{}".format(year, month)

def timestamp_str(filename):
    """
    Returns the numpy datetime 64 [ns] for the current date.
    This is a bit hardcoded at the moment ...
    """
    splits = filename.split('-')
    ts = splits[5]
    year = ts[:4]
    month = ts[4:6]
    day = ts[6:8]
    hr = ts[8:10]
    minuts = ts[10:12]
    sek = ts[12:14]
    # TODO make sure all filenames have seconds
    return np.datetime64( year+"-"+month+"-"+day+"T"+hr+":00:00"+".000000").astype(str)

def get_missing_vals(folder):
    """ Returns missing timesteps in folder. """
    year, month = folder.split('_')
    year  = int(year)
    month = int(month)

    t = np.arange(datetime(year,month,1), datetime(year,month+1,1), timedelta(hours=1)).astype(str)
    folder = make_folder_str(month, year)
    files_in_folder = glob.glob(os.path.join(path, folder, '*grb'))
    times = [timestamp_str(fil) for fil in files_in_folder]
    a = times
    b = t
    c = [x for x in a if x not in b]+[x for x in b if x not in a]
    return c

def timestamp_to_file_search_str(timestamp):
    print(timestamp)
    timestamp = timestamp.tostring()
    print(timestamp)
    splits = [part.split('T') for part in timestamp.split(':')[0].split('-')]
    s = ''
    for a in np.concatenate(splits):
        s+=a
    return s

def removes_duplicates(year, month):
    folder = make_folder_str(year, month)
    #logging.debug( get_missing_vals(folder) ) # TODO

    files_in_folder = glob.glob(os.path.join(path, folder, '*grb'))
    times = [timestamp_str(fil) for fil in files_in_folder]

    if len(np.unique(times)) != len(times):
        keeping = []
        missing = []
        for fil in files_in_folder:
            # if timestep is already there don't append
            print(fil)
            search_for = timestamp_to_file_search_str( timestamp_str(fil) )
            files =  glob.glob(os.path.join(path, folder, '*-{}*grb'.format(search_for)))
            if len(files) > 0:
                # Keeping the operational satelite largest number of characher in place 3.
                max = 1
                fil = None
                # TODO spøit first at / then get char 3.
                for f in files:
                    filesname = f.split('/')[-1]
                    char = filename[3]
                    if int(char) >= max:
                        fil = f
                        max = int(char)
                keeping.append(fil) # only keep the first one for multiple files of the same data
        return keeping
    else:
        return glob.glob(os.path.join(path, folder, '*.grb'))

def merge_ts_to_one_dataset(grb_files,
                            lat = np.arange(30.0, 50.25, 0.25),
                            lon = np.arange(-15.0, 25.25, 0.25)):
    """ grib_files : list of files. One month. """
    data_grid = get_dict_with_all_keys()

    counter = 0
    for filename in grb_files:
        cloud_fraction, nans = compute(filename, lat, lon)
        if cloud_fraction is not None:
            # this becomes true if the grib file of the satelite images is corrupt.
            if counter == 0:
                # if the computation worked
                ds = xr.Dataset({'tcc': (['latitude', 'longitude'],   cloud_fraction),
                                 'nr_nans':(['latitude', 'longitude'], nans),
                                 #'nr_cells':(['latitude', 'longitude'], cnt_cells)
                                     },
                                 coords={'longitude': (['longitude'], lon),
                                         'latitude': (['latitude'], lat),
                                        })
                ts = timestamp(filename)
                ds['time'] = ts

                # Add time as a coordinate and dimension.
                ds = ds.assign_coords(time = ds.time)
                ds = ds.expand_dims(dim = 'time')
                counter += 1
                print('Finished nbr {}/{}'.format(counter, len(grb_files)))
            else:
                #clm, cnt_cells, cnt_nans = calc_all(filename, nc_file = nc_files[0])
                #cloud_fraction, nans = compute(filename, lat, lon)
                new_ds = xr.Dataset({'tcc': (['latitude', 'longitude'],  cloud_fraction),
                                     'nr_nans':(['latitude', 'longitude'], nans),
                                     #'nr_cells':(['latitude', 'longitude'], cnt_cells)
                                    },
                                      coords={'longitude': (['longitude'], lon),
                                              'latitude': (['latitude'], lat),
                                               })

                ts = timestamp(filename)
                new_ds['time'] = ts

                # Add time as a coordinate and dimension.
                new_ds = new_ds.assign_coords(time = new_ds.time)
                new_ds = new_ds.expand_dims(dim = 'time')

                try:
                    ds = ds.merge(new_ds)
                except xr.MergeError:
                    # Happens if MS1 and MS2 have taken a image at the same time
                    print("Filename not included {}".format(filename))

                counter += 1
                print('Finished nbr {}/{}'.format(counter, len(grb_files)))
        #print("completed {}/{} files".format(counter, len(grb_files)))
    return ds

def compute_one_folder(subset, year, month):
    import os
    ds = merge_ts_to_one_dataset(subset,
                                 lat =  np.arange(30.0, 50.25, 0.25) ,
                                 lon = np.arange(-15.0, 25.25, 0.25) )
    save_dir = generate_save_dir(year, month)
    ds.to_netcdf(path = os.path.join(save_dir,'{}_{:02d}_tcc.nc'.format(year, month)),
                 engine='netcdf4',
                 encoding ={'tcc': {'zlib': True, 'complevel': 9},
                           'nr_nans': {'zlib': True, 'complevel': 9} })
    # TODO : update to save on both lagringshotellet and miphclac
    return

def already_regridded(year, month):
    path = generate_save_dir(year, month)
    #folder = make_folder_str(year = year, month = month)
    month = "%2.2d"%month # Skriver
    #key = "*-{}{}*.grb".format(year, month)
    full = os.path.join(path, '{}_{}_tcc.nc'.format(year, month) )
    return os.path.isfile(full)

years = np.arange(2005, 2019)
months = np.arange(1, 13)

for y in years:
    for m in months:
        folder = make_folder_str(y, m)
        files_to_read = removes_duplicates(y, m)

        if len(files_to_read) > 0 and not already_regridded(y, m):
            print("Starts computation for folder : {}, containing {} files.".format(folder, len(files_to_read)))
            compute_one_folder(subset=files_to_read, year=y, month = m)
            #print(already_regridded(year = y, month = m))
