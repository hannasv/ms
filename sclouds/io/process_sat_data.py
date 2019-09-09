#!/usr/bin/env python
# coding: utf-8

# ## Crop/select aera and calculate cloud fraction should be done in the same period.

# In[1]:


import glob
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import datetime
from netCDF4 import Dataset # used for the netcdf files which contain lat, lon.
import seaborn as sns

#path      = '//uio/lagringshotell/geofag/students/metos/hannasv/satelite_data_raw/'
#path_era  = '//uio/lagringshotell/geofag/students/metos/hannasv/era_interim_data/'
#nc_path   = '//uio/lagringshotell/geofag/students/metos/hannasv/satelite_coordinates/'

directory = '/home/hanna/Desktop/examples_master_thesis/'
path_era  = '/home/hanna/Desktop/master_thesis/era/'
#nc_path   = '//uio/lagringshotell/geofag/students/metos/hannasv/satelite_coordinates/'

nc_files  = glob.glob(directory + '*.nc')
grb_files = glob.glob(directory + "*.grb")
era       = glob.glob(path_era+"*relative*humidity*.nc")

grb_file = grb_files[0]
era_file = era[0]
nc_file = nc_files[0]

#data = xr.open_dataset(grb_file, engine = "pynio")
era = xr.open_dataset(era_file)

def timestamp(filename):
    """
    Returns the np.datetime64 [ns] for the current date.
    """
    splits = filename.split('-')
    # print(splits)
    ts = splits[5]
    year = ts[:4]
    month = ts[4:6]
    day = ts[6:8]
    hr = ts[8:10]
    minuts = ts[10:12]
    sek = ts[12:14]
    return np.datetime64( year+"-"+month+"-"+day+"T"+hr+":"+minuts+":"+sek )


# ## Retrieving coordinates from nc file.
# ### Notes a nc is 10 times as large as a grib file that is why we only have one of them

# In[5]:

"""
PROBLEM WITH PYNIO

ts = timestamp(grb_file)
data['time'] = ts
# Add time as a coordinate and dimension.
data = data.assign_coords(time = data.time)
cmk = data['CLOUDM_P30_GSV0_I207'].values


cloud_mask_array = rootgrp.variables["cloudMask"][:].data
cloud_mask_array[cloud_mask_array == 1] = 0
cloud_mask_array[cloud_mask_array == 2] = 1
cloud_mask_array[cloud_mask_array == 3] = np.nan
cloud_mask_array = cloud_mask_array.reshape(-1)

# as a reference that its upside down --> explain negative latitude values.
# 3 -off earth disk
# 2 - cloud
# 1 - not cloud over ocean and 0 not cloud over land
"""

rootgrp = Dataset(nc_file, "r", format="NETCDF4")
cloud_mask_array = rootgrp.variables["cloudMask"][:].data
lat_array = rootgrp.variables["lat"][:].data
lon_array = rootgrp.variables["lon"][:].data
lat_array[lat_array < -99] = np.nan # updates of disk values to nan
lon_array[lon_array < -99] = np.nan # updates of disk values to nan

#d_theta = (lat_array[1:, :].transpose() - lat_array[:-1, :].transpose() ).transpose()

d_phi   = lon_array[:, 1:] - lon_array[:, :-1]
d_theta = lat_array[:-1] - lat_array[1:]

pad     = np.ones((3712, 1))*np.nan # adding numpy to the axis the values
d_phi   = np.concatenate((pad, d_phi), axis = 1)/2 #paddes forran
d_theta = np.concatenate((pad.transpose(), d_theta), axis = 0)/2 # paddes forran


def area_grid_cell(c_lat, d_lat, d_lon):
    """
    c_lat : float
        Centre point longitude, latitude in degrees

    d_lat, d_lon : float
        delta lat lon in degrees

    Returns : area in km^2

    cdo : If the grid cell area have to be computed it is scaled with the earth radius to square meters.
    """
    R = 6371000  # in M
    # area er egentlig R**2
    area = R*(-np.sin((c_lat - d_lat)*np.pi/180)+np.sin((c_lat + d_lat)*np.pi/180) )*(2*d_lon*np.pi/180) # R**2
    return area

# # TODO: this is not availabel on laptop
#  assert ((era.cell_area.values/6371000)[:,0] -
# area_grid_cell(era.latitude.values, 0.375, 0.375))/(era.cell_area.values/6371000)[:,0] < 0.001

# these should be sent in as attributes.
era_lon = -15
era_lat = 30
era_AREA = area_grid_cell(30.0, 0.375, 0.375)
# Make this a loop over lat_lons?
lat_bondaries = np.array([[era_lon],
                          [era_lat]])

BOUND =  np.array([[-0.75/2, 0.75/2],
                   [-0.75/2, 0.75/2]])

ranges = lat_bondaries + BOUND

lon_range = ranges[0, :]
lat_range = ranges[1, :]
min_lon, max_lon = lon_range
min_lat, max_lat = lat_range

d_phi   = d_phi.reshape(-1)
d_theta = d_theta.reshape(-1)

era_up    = ranges[1, 1]
era_down  = ranges[1, 0]
era_left  = ranges[0, 0]
era_right = ranges[0, 1]

c_lon = lon_array.reshape(-1) #+ d_phi
c_lat = lat_array.reshape(-1) #+ d_theta

cmk_left  = c_lon - d_phi   #- era_right
cmk_right = c_lon + d_phi   #- era_left

# TODO : Sjekk d theta og lignende.
cmk_up    = c_lat + np.abs(d_theta) #- era_down
cmk_down  = c_lat - np.abs(d_theta) #- era_up

idx_left_boundary  = np.intersect1d(np.argwhere(cmk_right > era_left),  np.argwhere(cmk_left < era_left) )
idx_right_boundary = np.intersect1d(np.argwhere(cmk_right > era_right), np.argwhere(cmk_left < era_right) )
idx_up_boundary    = np.intersect1d(np.argwhere(cmk_up > era_up), np.argwhere(cmk_down < era_up) )
idx_down_boundary  = np.intersect1d(np.argwhere(cmk_up > era_down), np.argwhere(cmk_down < era_down) )

lower_right_corner = np.intersect1d(idx_down_boundary, idx_right_boundary)
lower_left_corner  = np.intersect1d(idx_down_boundary, idx_left_boundary)
upper_left_corner  = np.intersect1d(idx_up_boundary, idx_left_boundary)
upper_right_corner = np.intersect1d(idx_up_boundary, idx_right_boundary)

corner_idx = np.concatenate([lower_right_corner, lower_left_corner,
                             upper_left_corner, upper_right_corner]) # corner idx

max_lon, min_lon = np.max(c_lon[corner_idx]), np.min(c_lon[corner_idx])
max_lat, min_lat = np.max(c_lat[corner_idx]), np.min(c_lat[corner_idx])

# removes corners
idx_down_boundary = idx_down_boundary[idx_down_boundary != lower_right_corner]
idx_down_boundary = idx_down_boundary[idx_down_boundary != lower_left_corner]

idx_up_boundary = idx_up_boundary[idx_up_boundary != upper_right_corner]
idx_up_boundary = idx_up_boundary[idx_up_boundary != upper_left_corner]

idx_left_boundary = idx_left_boundary[idx_left_boundary != upper_left_corner]
idx_left_boundary = idx_left_boundary[idx_left_boundary != lower_left_corner]

idx_right_boundary = idx_right_boundary[idx_right_boundary != upper_right_corner]
idx_right_boundary = idx_right_boundary[idx_right_boundary != lower_right_corner]


# ## Need to do this by boundary.

# In[578]:


# subsection left boundary OLD
low_bound = np.argwhere( c_lat[idx_left_boundary] > min_lat  )
up_bound  = np.argwhere( c_lat[idx_left_boundary] < max_lat  )
sub_section_left = np.intersect1d(low_bound, up_bound)


# In[580]:


# subsection right boundary
low_bound = np.argwhere( c_lat[idx_right_boundary] > min_lat )
up_bound  = np.argwhere( c_lat[idx_right_boundary] < max_lat)
sub_section_right = np.intersect1d(low_bound, up_bound)


# In[583]:


# Subsection Down Boundary
one = np.argwhere( c_lon[idx_down_boundary] > min_lon )
two = np.argwhere( c_lon[idx_down_boundary] < max_lon)
sub_section_down = np.intersect1d(one, two)


# In[585]:


# subsection up boundary
one = np.argwhere( c_lon[idx_up_boundary] > min_lon)
two = np.argwhere( c_lon[idx_up_boundary] < max_lon)
sub_section_up = np.intersect1d(one, two)


# In[588]:


# test that these are empty
t1 = np.intersect1d(idx_down_boundary, idx_up_boundary)
t2 = np.intersect1d(idx_right_boundary, idx_left_boundary)
assert len(t1) == len(t2) == 0


# # THE ABOVE LINE SHOULD BE TRUE

# # Take intersection between boundary indexes in the original long format to detect corners.
# * Store all the new dlon and lat and dlat.
# * For each edge only one of these needs to be calculated.
# * area_grid_cell(c_lat, d_lat, d_lon)

# THE first two centres discussed are the longitude centres.
#
# ### dx_left = (clm centre + dlon(phi) - era_centre)/2  (right boundary)
#
# Above eq describes the centere of the new area.
#
# * If not corner -- one adjust one centre.
#
# ### dx_right = (era_centre lon - (clm centre lon- dlon(phi)  this is the left bound  ) )/2

# In[589]:


largeAREA = area_grid_cell(30, 0.375, 0.375)
largeAREA


# In[659]:


# AREA left boundary
dlon_lf = ( era_left - (c_lon[idx_left_boundary][sub_section_left] +
                         d_phi[idx_left_boundary][sub_section_left]))/2 # negative values

dlat_lf = d_theta[idx_left_boundary][sub_section_left]

lat_lf = c_lat[idx_left_boundary][sub_section_left]
lon_lf = c_lon[idx_left_boundary][sub_section_left]

a = np.sum(area_grid_cell(lat_lf, dlat_lf, np.abs(dlon_lf)))

fraction_left = (cloud_mask_array[idx_left_boundary][sub_section_left]*area_grid_cell(lat_lf,
                                                                                      dlat_lf,
                                                                                      dlon_lf)/largeAREA).sum()
# AREA right boundary
# dlon right boundary - of one era interim
dlon_rb = (c_lon[idx_right_boundary][sub_section_right] - d_phi[idx_right_boundary][sub_section_right] -
           era_right)/2

dlat_rb = d_theta[idx_right_boundary][sub_section_right]

lat_rb = c_lat[idx_right_boundary][sub_section_right]
lon_rb = c_lon[idx_right_boundary][sub_section_right]

b = np.sum(area_grid_cell(lat_rb, dlat_rb, np.abs(dlon_rb)))

fraction_right = (cloud_mask_array[idx_right_boundary][sub_section_right]*area_grid_cell(lat_rb,
                                                                                      dlat_rb,
                                                                                      dlon_rb)/largeAREA).sum()


# In[654]:


# AREA down boundary
dlat_down = (era_down - c_lat[idx_down_boundary][sub_section_down] +
              d_theta[idx_down_boundary][sub_section_down])/2

lat_down = era_down + dlat_down

dlon_down = d_phi[idx_down_boundary][sub_section_down]
lon_down =  c_lon[idx_down_boundary][sub_section_down]

c = np.sum(area_grid_cell(lat_down, dlat_down, dlon_down))

fraction_down = (cloud_mask_array[idx_down_boundary][sub_section_down]*area_grid_cell(lat_down,
                                                                                      dlat_down,
                                                                                      dlon_down)/largeAREA).sum()

# AREA up boundary
dlat_up = (era_up - (c_lat[idx_up_boundary][sub_section_up] -
                     d_theta[idx_up_boundary][sub_section_up]) )/2

lat_up = era_up - dlat_up
lon_up = c_lon[idx_up_boundary][sub_section_up]
dlon_up = d_phi[idx_up_boundary][sub_section_up]

d = np.sum(area_grid_cell(lat_up, dlat_up, np.abs(dlon_up)))

fraction_up = (cloud_mask_array[idx_up_boundary][sub_section_up]*area_grid_cell(lat_up,
                                                             dlat_up,
                                                             dlon_up)/largeAREA).sum()




# In[604]:


lat_centre_cells =   c_lat[idx_centre_cells]
lon_centre_cells =   c_lon[idx_centre_cells]
dlon_centre      =   d_phi[idx_centre_cells]
dlat_centre      = d_theta[idx_centre_cells]


# In[605]:


fig, ax = plt.subplots(figsize=(15, 10))

plt.scatter(lon_centre_cells, lat_centre_cells, color = 'r')
plt.scatter(lon_lf, lat_lf, color = 'b')
plt.scatter(lon_rb, lat_rb, color = 'g')
plt.scatter(lon_down, lat_down, color = 'y')
plt.scatter(lon_up, lat_up, color = 'm')
plt.scatter( c_lon[corner_idx], c_lat[corner_idx], color = 'c', marker = 'v' )
plt.grid( )
#plt.()


# In[ ]:


idx_centre_one = np.intersect1d(np.argwhere(cmk_left  > era_left),
                                np.argwhere(cmk_right < era_right))


idx_centre_two = np.intersect1d(np.argwhere(cmk_up   < era_up),
                                np.argwhere(cmk_down >  era_down))

idx_centre_cells = np.intersect1d( idx_centre_one, idx_centre_two )

fraction_centre = (cloud_mask_array[idx_centre_cells]*area_grid_cell(lat_centre_cells,
                                                             dlat_centre,
                                                             dlon_centre)/largeAREA).sum()



counter = 0
for filename in grb_files:# grb file of satellite image...
    print(filename)
    if counter == 0:
        print("enters 0")
        clm, cnt_cells, cnt_nans = calc_all(filename, nc_file = nc_files[0])
        ds = xr.Dataset({'tcc': (['latitude', 'longitude'],  tcc),
                         'nr_nans':(['latitude', 'longitude'], cnt_nans),
                        'nr_cells':(['latitude', 'longitude'], cnt_cells)},
                         coords={'longitude': (['longitude'], lon),
                                 'latitude': (['latitude'], lat),
                                })

        ts = timestamp(filename)
        ds['time'] = ts

        # Add time as a coordinate and dimension.
        ds = ds.assign_coords(time = data.time)
        ds = ds.expand_dims(dim = 'time')
        counter += 1

    else:
        print("enters 1")
        clm, cnt_cells, cnt_nans = calc_all(filename, nc_file = nc_files[0])
        new_ds = xr.Dataset({'tcc': (['latitude', 'longitude'],  tcc),
                             'nr_nans':(['latitude', 'longitude'], cnt_nans),
                             'nr_cells':(['latitude', 'longitude'], cnt_cells)},
                              coords={'longitude': (['longitude'], lon),
                                      'latitude': (['latitude'], lat),
                                       })

        ts = timestamp(filename)
        new_ds['time'] = ts

        # Add time as a coordinate and dimension.
        new_ds = new_ds.assign_coords(time = data.time)
        new_ds = new_ds.expand_dims(dim = 'time')

        ds.merge(new_ds)

        counter += 1

new_ds = xr.Dataset({'tcc': (['latitude', 'longitude'],  tcc),
                     'nr_nans':(['latitude', 'longitude'], cnt_nans),
                     'nr_cells':(['latitude', 'longitude'], cnt_cells)},

                      coords={'longitude': (['longitude'], lon),
                              'latitude': (['latitude'], lat) })

ts = timestamp(filename)
new_ds['time'] = ts

# Add time as a coordinate and dimension.
new_ds = new_ds.assign_coords(time = data.time)
new_ds = new_ds.expand_dims(dim = 'time')

def create_grb_file_by_filenames(filenames):
    counter = 0
    for filename in filenames:# grb file of satellite image...
        print()
        if counter == 0:
            print("enters 0")
            clm, cnt_cells, cnt_nans = calc_all(filename, nc_file = nc_files[0])
            ds = xr.Dataset({'tcc': (['latitude', 'logitude'],  tcc),
                             'nr_nans':(['latitude', 'logitude'], cnt_nans),
                            'nr_cells':(['latitude', 'logitude'], cnt_cells)},
                             coords={'longitude': (['longitude'], lon),
                                     'latitude': (['latitude'], lat),
                                    })

            ts = timestamp(filename)
            ds['time'] = ts

            # Add time as a coordinate and dimension.
            ds = ds.assign_coords(time = data.time)
            ds = ds.expand_dims(dim = 'time')
            counter += 1

        else:
            print("enters 1")
            clm, cnt_cells, cnt_nans = calc_all(filename, nc_file = nc_files[0])
            new_ds = xr.Dataset({'tcc': (['latitude', 'logitude'],  tcc),
                             'nr_nans':(['latitude', 'logitude'], cnt_nans),
                            'nr_cells':(['latitude', 'logitude'], cnt_cells)},
                             coords={'longitude': (['longitude'], lon),
                                     'latitude': (['latitude'], lat),
                                    })

            ts = timestamp(filename)
            new_ds['time'] = ts

            # Add time as a coordinate and dimension.
            new_ds = new_ds.assign_coords(time = data.time)
            new_ds = new_ds.expand_dims(dim = 'time')

            ds.merge(new_ds)

            counter += 1

    print(ds)
    return
