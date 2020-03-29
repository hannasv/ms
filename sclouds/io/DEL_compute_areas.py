
import glob
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import datetime
from netCDF4 import Dataset # used for the netcdf files which contain lat, lon.
import seaborn as sns
import pandas as pd

class Resample:

    def __init__(self):

        self.path_grb      = '//uio/lagringshotell/geofag/students/metos/hannasv/satelite_data_raw/'
        self.path_era  = '//uio/lagringshotell/geofag/students/metos/hannasv/era_interim_data/'
        self.nc_path   = '//uio/lagringshotell/geofag/students/metos/hannasv/satelite_coordinates/'
        # Update
        self.store_dir = '//uio/lagringshotell/geofag/students/metos/hannasv/satelite_coordinates/'

        nc_files  = glob.glob(self.nc_path + '*.nc')
        grb_files = glob.glob(self.path_grb + "*.grb")
        grb_files = glob.glob(self.path_grb + "*.grb")
        era       = glob.glob(self.path_era + "*q.nc")

        self.grb_file = grb_files[0]
        self.era_file = era[0]
        self.nc_file = nc_files[0]

        self.data = xr.open_dataset(self.grb_file, engine = "pynio")
        self.era = xr.open_dataset(self.era_file)

        return

    def compute_dlat_dlon(self, save_file = False):
        rootgrp = Dataset(self.nc_file, "r", format="NETCDF4")
        cloud_mask_array = rootgrp.variables["cloudMask"][:].data
        lat_array = rootgrp.variables["lat"][:].data
        lon_array = rootgrp.variables["lon"][:].data
        lat_array[lat_array < -99] = np.nan # updates of disk values to nan
        lon_array[lon_array < -99] = np.nan # updates of disk values to nan
        d_phi   = np.zeros(np.shape(lat_array))
        d_theta = np.zeros(np.shape(lat_array))

        for i in range(1, 3711):
            for j in range(1, 3711):
                d_phi[i][j] = ( np.abs(lon_array[i-1][j]) - np.abs(lon_array[i+1][j]) )/4
                d_theta[i][j] = ( np.abs(lat_array[i][j-1]) - np.abs(lat_array[i][j+1]) )/4

        dictionary = {'lon': lon_array.reshape(-1),
                      'lat': lat_array.reshape(-1),
                      'dlat':  d_theta.reshape(-1),
                      'dlon': d_phi.reshape(-1)}

        df = pd.DataFrame.from_dict(dictionary)
        if save_file:
            df.to_csv( '~/Desktop/lat_lon_dlat_dlon.csv' )

        return

    def area_grid_cell(self, c_lat, d_lat, d_lon):
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


    def compute_area_one_cell(self, lat = 30, lon = -15):
        era_AREA = self.area_grid_cell(lat, 0.375, 0.375)
        coord_info = pd.read_csv('~/Desktop/lat_lon_dlat_dlon.csv')

        c_lat = coord_info.lat.values
        c_lon = coord_info.lon.values
        d_phi = coord_info.d_lon.values
        d_theta = coord_info.dlat.values

        # Make this a loop over lat_lons?
        lat_bondaries = np.array([[lon],
                                  [lat]])

        BOUND =  np.array([[-0.75/2, 0.75/2],
                           [-0.75/2, 0.75/2]])

        ranges = lat_bondaries + BOUND

        lon_range = ranges[0, :]
        lat_range = ranges[1, :]
        min_lon, max_lon = lon_range
        min_lat, max_lat = lat_range

        #d_phi   = d_phi.reshape(-1)
        #d_theta = d_theta.reshape(-1)

        era_up    = ranges[1, 1]
        era_down  = ranges[1, 0]
        era_left  = ranges[0, 0]
        era_right = ranges[0, 1]

        #c_lon = lon_array.reshape(-1) #+ d_phi
        #c_lat = lat_array.reshape(-1) #+ d_theta

        cmk_left  = c_lon - np.abs(d_phi)   #- era_right
        cmk_right = c_lon + np.abs(d_phi)   #- era_left

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

        corner_idx         = np.concatenate([lower_right_corner, lower_left_corner,
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

        # subsection left boundary OLD
        low_bound = np.argwhere( c_lat[idx_left_boundary] > min_lat  )
        up_bound  = np.argwhere( c_lat[idx_left_boundary] < max_lat  )
        sub_section_left = np.intersect1d(low_bound, up_bound)

        # subsection right boundary
        low_bound = np.argwhere( c_lat[idx_right_boundary] > min_lat )
        up_bound  = np.argwhere( c_lat[idx_right_boundary] < max_lat)
        sub_section_right = np.intersect1d(low_bound, up_bound)

        # Subsection Down Boundary
        one = np.argwhere( c_lon[idx_down_boundary] > min_lon )
        two = np.argwhere( c_lon[idx_down_boundary] < max_lon)
        sub_section_down = np.intersect1d(one, two)

        # subsection up boundary
        one = np.argwhere( c_lon[idx_up_boundary] > min_lon)
        two = np.argwhere( c_lon[idx_up_boundary] < max_lon)
        sub_section_up = np.intersect1d(one, two)

        # test that these are empty
        t1 = np.intersect1d(idx_down_boundary, idx_up_boundary)
        t2 = np.intersect1d(idx_right_boundary, idx_left_boundary)
        print( len(t1))
        print(len(t2) )
        #assert len(t1) == len(t2) == 0

        # Calculate Boundaries

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
        dlon_rb = (c_lon[idx_right_boundary][sub_section_right] - d_phi[idx_right_boundary][sub_section_right] -
                   era_right)/2

        dlat_rb = d_theta[idx_right_boundary][sub_section_right]

        lat_rb = c_lat[idx_right_boundary][sub_section_right]
        lon_rb = c_lon[idx_right_boundary][sub_section_right]

        b = np.sum(area_grid_cell(lat_rb, dlat_rb, np.abs(dlon_rb)))

        fraction_right = (cloud_mask_array[idx_right_boundary][sub_section_right]*area_grid_cell(lat_rb,
                                                                                              dlat_rb,
                                                                                              dlon_rb)/largeAREA).sum()

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
        # AREA up
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


        # Index centres.
        idx_centre_one = np.intersect1d(np.argwhere(cmk_left  > era_left),
                                        np.argwhere(cmk_right < era_right))


        idx_centre_two = np.intersect1d(np.argwhere(cmk_up   < era_up),
                                        np.argwhere(cmk_down >  era_down))

        idx_centre_cells = np.intersect1d( idx_centre_one, idx_centre_two )

        fraction_centre = (cloud_mask_array[idx_centre_cells]*area_grid_cell(lat_centre_cells,
                                                                     dlat_centre,
                                                                     dlon_centre)/largeAREA).sum()


        Test_weight = (area_grid_cell(lat_up, dlat_up, dlon_up) +
                      area_grid_cell(lat_centre_cells, dlat_centre, dlon_centre) +
                      area_grid_cell(lat_down, dlat_down, dlon_down) +
                      area_grid_cell(lat_rb, dlat_rb, dlon_rb) +
                     area_grid_cell(lat_lf, dlat_lf, dlon_lf) ) /era_AREA
        print("SUM weight {}".format(Test_weight))
        return



if __name__ == "__main__":
    #Resample().compute_dlat_dlon(save_file=True)
    Resample().compute_area_one_cell()
