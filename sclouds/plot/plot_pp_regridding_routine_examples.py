import glob
import os, sys
import json

import numpy as np
import xarray as xr
import pandas as pd
import seaborn as sns
## Read statistics
from sclouds.plot.helper_bar_plot import (autolabel,
                    read_global_statistics_to_dataframe)
from sclouds.helpers import (VARIABLES, FILTERS, LONGNAME, path_stats_results,
                                UNITS, path_store_plots)

from sclouds.plot.helpers import (import_matplotlib_pp, PP_WIDTH_IN,
                                    PP_HEIGHT_IN, cmap_contour_plot, add_ticks)
mat = import_matplotlib_pp()
import matplotlib.pyplot as plt

save_dir = '/home/hanna/MS-suppl/files/'
files = glob.glob(save_dir + '*ERA5*.json')

store_dir = '/home/hanna/lagrings/satelite_coordinates/msthesis/'
coords = xr.open_dataset(store_dir + 'changes_lat_lon_for_plot2.nc')

d_lon = np.abs(coords['d_phi2'].values.flatten())
d_lat = np.abs(coords['d_theta2'].values.flatten())
lat_array = coords.lat.values.flatten()  # 2 dimensional array
lon_array = coords.lon.values.flatten() # 2 dimensional array
print('loads dataset ')


def get_dict_with_all_keys():
    ex_fil = glob.glob(save_dir + '*ERA5*.json')
    ex_fil = np.sort(ex_fil)[:-1]
    merged_dict = {}
    for fil in ex_fil:
        with open(fil, 'r') as f:
            data_grid = json.load(f)
        merged_dict.update(data_grid)
    return merged_dict
data = get_dict_with_all_keys()
print('loads form json')
#ax = sns.heatmap(areas[:, :161], ax = ax, cmap = 'viridis', cbar_kws = {'label':'accumulated area in km2'})

n = 1000
one = np.ones(n)

color = 'black'
color_ref = 'pink'

lat = '45.0'
lon = '25.0'

lw = 0.5
fig, ax = plt.subplots(1, 1, figsize = (PP_WIDTH_IN, PP_HEIGHT_IN) )

centre = data[lat][lon]['centre']['index']
left = data[lat][lon]['left']['index']
up = data[lat][lon]['up']['index']
right = data[lat][lon]['right']['index']
down = data[lat][lon]['down']['index']
corner = data[lat][lon]['corner']['index']

d_era = 0.25/2

lat = float(lat)
lon = float(lon)

dlat = d_era
dlon = d_era

vertical   = np.linspace( lat - dlat, lat + dlat , n )
horizontal = np.linspace( lon - dlon, lon + dlon , n )

color = 'green'

ax.scatter(lon, lat, c = color)
ax.plot( horizontal, (lat + dlat)*one,  c = color, linewidth=2)
ax.plot( horizontal, (lat - dlat)*one,  c = color,  linewidth=2)
ax.plot( (lon + dlon)*one, vertical,    c = color, linewidth=2  )
ax.plot( (lon - dlon)*one, vertical,    c = color, linewidth=2 )

for i in centre:
    lat =  lat_array[i]
    lon = lon_array[i]
    ax.scatter(lon, lat, c = color_ref)

    dlat = abs(d_lat[i])
    dlon = abs(d_lon[i])
    color = 'purple'

    vertical   = np.linspace( lat - dlat, lat + dlat , n )
    horizontal = np.linspace( lon - dlon, lon + dlon , n )

    ax.scatter(lon, lat, c = color)
    ax.plot( horizontal, (lat + dlat)*one,  c = color  , linewidth=lw)
    ax.plot( horizontal, (lat - dlat)*one,  c = color , linewidth=lw)
    ax.plot( (lon + dlon)*one, vertical,    c = color  , linewidth=lw)
    ax.plot( (lon - dlon)*one, vertical,    c = color , linewidth=lw)

for i in up:
    lat =  lat_array[i]
    lon = lon_array[i]
    ax.scatter(lon, lat, c = color_ref)

    dlat = abs(d_lat[i])
    dlon = abs(d_lon[i])
    color = 'purple'

    vertical   = np.linspace( lat - dlat, lat + dlat , n )
    horizontal = np.linspace( lon - dlon, lon + dlon , n )

    ax.scatter(lon, lat, c = color)
    ax.plot( horizontal, (lat + dlat)*one,  c = color , linewidth=lw)
    ax.plot( horizontal, (lat - dlat)*one,  c = color , linewidth=lw)
    ax.plot( (lon + dlon)*one, vertical,    c = color  , linewidth=lw)
    ax.plot( (lon - dlon)*one, vertical,    c = color , linewidth=lw)

for i in down:
    lat =  lat_array[i]
    lon = lon_array[i]

    dlat = abs(d_lat[i])
    dlon = abs(d_lon[i])
    color = 'purple'

    vertical   = np.linspace( lat - dlat, lat + dlat , n )
    horizontal = np.linspace( lon - dlon, lon + dlon , n )

    ax.scatter(lon, lat, c = color)
    ax.plot( horizontal, (lat + dlat)*one,  c = color , linewidth=lw)
    ax.plot( horizontal, (lat - dlat)*one,  c = color , linewidth=lw)
    ax.plot( (lon + dlon)*one, vertical,    c = color  , linewidth=lw)
    ax.plot( (lon - dlon)*one, vertical,    c = color , linewidth=lw)

for i in right:
    lat =  lat_array[i]
    lon = lon_array[i]

    dlat = abs(d_lat[i])
    dlon = abs(d_lon[i])
    color = 'purple'

    vertical   = np.linspace( lat - dlat, lat + dlat , n )
    horizontal = np.linspace( lon - dlon, lon + dlon , n )

    ax.scatter(lon, lat, c = color)
    ax.plot( horizontal, (lat + dlat)*one,  c = color  , linewidth=lw)
    ax.plot( horizontal, (lat - dlat)*one,  c = color , linewidth=lw)
    ax.plot( (lon + dlon)*one, vertical,    c = color  , linewidth=lw)
    ax.plot( (lon - dlon)*one, vertical,    c = color , linewidth=lw)

for i in left:
    lat =  lat_array[i]
    lon = lon_array[i]

    dlat = abs(d_lat[i])
    dlon = abs(d_lon[i])
    color = 'purple'

    vertical   = np.linspace( lat - dlat, lat + dlat , n )
    horizontal = np.linspace( lon - dlon, lon + dlon , n )

    ax.scatter(lon, lat, c = color)
    ax.plot( horizontal, (lat + dlat)*one,  c = color , linewidth=lw)
    ax.plot( horizontal, (lat - dlat)*one,  c = color, linewidth=lw)
    ax.plot( (lon + dlon)*one, vertical,    c = color , linewidth=lw)
    ax.plot( (lon - dlon)*one, vertical,    c = color , linewidth=lw)

for i in corner:
    lat =  lat_array[i]
    lon = lon_array[i]

    dlat = abs(d_lat[i])
    dlon = abs(d_lon[i])
    color = 'purple'

    vertical   = np.linspace( lat - dlat, lat + dlat , n )
    horizontal = np.linspace( lon - dlon, lon + dlon , n )

    ax.scatter(lon, lat, c = color)
    ax.plot( horizontal, (lat + dlat)*one,  c = color , linewidth=lw )
    ax.plot( horizontal, (lat - dlat)*one,  c = color , linewidth=lw)
    ax.plot( (lon + dlon)*one, vertical,    c = color , linewidth=lw )
    ax.plot( (lon - dlon)*one, vertical,    c = color, linewidth=lw)


plt.title('Regridding ({}, {})'.format(int(lon), int(lat)))

plt.ylabel('Latitude')
plt.xlabel('Longitude')

plt.subplots_adjust(bottom = 0.2, right = 0.97)
path_presentation = '/home/hanna/MS-presentation/figures/'
plt.savefig(os.path.join(path_presentation, 'example_remapping_lat{}_lon{}.png'.format(int(lat), int(lon))))
