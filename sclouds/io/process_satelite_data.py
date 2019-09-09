import glob
import xarray as xr
import numpy as np
from sclouds.helpers import LAPTOP_REPO, LAPTOP_RESULTS_REPO

class ProcessSateliteData:
    """
    Class for processing sataelite data.

    It needs to be:
        1. Cropped to the correct regions.
        2. Replace "no cloud over ocean and land" with "no cloud".
        3. Regridd and calculate cloud fractions
            IDEA :: use sel.(slice=lat, lon) and take a mean.
                    and save somewhere else or ??? in the same??

        4. Merge timesteps together. Preferably get it into one file.
            * consider seasonal files
            * consider train test split files.

        OBS! Here you can considerer the
        train test split when merging the timesteps.

        TODO: train test split should maybee be inported from a config.
        And should have one script to run which updates all files with a new train test split.

    """

      def __init__(self, var = "tcc", season = None):
          self.var = var
          self.season = season
          self.path = path_to_all_sat_data
          self.filenames = glob.glob(self.path+var+"*.grb")
          self.ds = None # xarray dump to netcdf in the end.
          return

    def timestamp(self, filename):
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
        return np.datetime64( year+"-"+month+"-"+day+"T"+hr+":"+minuts+":"+sek )

    def area_grid_cell(c_lat, d_lat, d_lon):
        """
        c_lon, c_lat : float
            Centre point longitude, latitude in degrees
        d_lat, d_lon : float
            delta lat lon in degrees

        Returns : area in km^2

        cdo : If the grid cell area have to be computed it is scaled with the earth radius to square meters.
        """
        R = 6371000  # in M TO BE COMPARABLE WITH
        # area er egentlig R**2
        area = R*(np.cos((c_lat - d_lat)*np.pi/180) - np.cos((c_lat + d_lat)*np.pi/180) )*(2*d_lon*np.pi/180) # R**2
        return area

    def calculate_cloud_fraction(self, centre_era_lon, centre_era_lat):
        """
        Calculate cloud fraction of one cell.
        Keep masks, lons, lats in memory by storing them in the constructure.
        """

        nc_path   = '//uio/lagringshotell/geofag/students/metos/hannasv/satelite_coordinates/'
        nc_file  = glob.glob(nc_path + '*.nc')[0]
        rootgrp = Dataset(nc_file, "r", format="NETCDF4")
        cloud_mask = rootgrp.variables["cloudMask"][:].data
        lat_array = rootgrp.variables["lat"][:].data
        lon_array = rootgrp.variables["lon"][:].data
        lat_array[lat_array < -99] = np.nan # updates of disk values to nan
        lon_array[lon_array < -99] = np.nan # updates of disk values to nan
        d_phi   = lon_array[:, 1:] - lon_array[:, :-1]
        d_theta = lat_array[1:] - lat_array[:-1]
        # Padding the axis -- we removed by taking the difference
        pad     = np.ones((3712, 1))*np.nan # adding numpy to the axis the values
        d_phi   = np.concatenate((pad, d_phi), axis = 1)/2
        d_theta = np.concatenate((pad.transpose(), d_theta), axis = 0)/2

        # reshapes everything to arrays
        d_phi = d_phi.reshape(-1)
        d_theta = d_theta.reshape(-1)
        longitude = lon_array.reshape(-1)
        latitude = lat_array.reshape(-1)

        # The indexes of the marked cell with centre inside the era cell
        # TODO REDO this. Remove the indecies which are in the intersection of inside and out.
        #These should be kept in the boundaries and removed from the inside..
        idx_centre_inside  = np.intersect1d( np.argwhere( np.abs(longitude - centre_era_lon) < 0.375 ),
                             np.argwhere( np.abs(latitude - centre_era_lat) < 0.375 ) )

        era_up_boundary    = centre_era_lat + 0.375
        era_down_boundary  = centre_era_lat - 0.375
        era_left_boundary  = centre_era_lon + 0.375
        era_right_boundary =  centre_era_lon - 0.375

        # boundaries of the
        cmk_left_boundary  = longitude - d_phi
        cmk_right_boundary = longitude + d_phi
        cmk_up_boundary    = latitude + d_theta
        cmk_down_boundary  = latitude - d_theta

        # Taking the dfferences of the oposite sides to make sure that they are
        d_left    = cmk_right - era_left
        d_right   = era_right - cmk_left
        d_down    = cmk_up - era_down
        d_up      = era_up - cmk_down

        # Indexes of the
        left_idx   = np.argwhere(d_left  < 0)
        right_idx  = np.argwhere(d_right > 0)
        up_idx     = np.argwhere(d_up    > 0)
        down_idx   = np.argwhere(d_down  < 0)

        # Corners are interesting because they should only be computed once
        lower_left_idx  = np.intersect1d(left_idx, down_idx)
        upper_left_idx  = np.intersect1d(left_idx, up_idx)
        lower_right_idx = np.intersect1d(right_idx, down_idx)
        upper_right_idx = np.intersect1d(right_idx, up_idx)

        assert len(np.intersect1d(right_idx, left_idx)) == 0, 'The intersection of left and right indexes should be zero?'


        # latitude lower left corner
        lat_LL = c_lat[lower_left_idx]
        d_lon_LL = d_phi[lower_left_idx]
        d_lat_LL = d_theta[lower_left_idx]

        idx_bondary = None

        # idx



        # Should be a pandas dataframe
        centre_inside_lat = None
        centre_inside_dtheta = None
        centre_inside_dphi = None

        # Should be a pandas dataframe
        centre_bound_lat   = None
        centre_bound_dphi   = None
        centre_bound_dtheta = None

        area_era     =  area_grid_cell(centre_era_lat, 0.375, 0.375)
        area_bondary =  area_grid_cell(centre_bound_lat, centre_bound_dtheta, centre_bound_dphi)
        area_inside  = area_grid_cell(centre_inside_lat, centre_inside_dtheta, centre_inside_dphi)

        sum_weights = np.sum( area_boundary/area_era + area_inside/area_era )
        assert sum_weights  < 1.100, 'sum_weights is to large {}'.format(sum_weights)
        #self.data = self.data.sel('latitude' slice = (), "longitude" slice = ())
        return

    def calc_one_mean(lon, lat, lon_nc, lat_nc, clm_grb):
        """
        lon : float
            single value of the cell you want to calc mean to.

        lat : float
            same as above

        lon_nc : 2D array
            array containg all the lons fron the netcdf files.

        lat_nc : 2D array
            array containg all the lat fron the netcdf files.

        clm_grb : 2Darray
            data from the grib file.

        """

        dlon = 0.75/2
        dlat = 0.75/2

        minLat = lat - dlat
        maxLat = lat + dlat

        # Since latitude > 0 and longitude can be plus or minus.
        a = lon - dlon
        b = lon + dlon
        minLon = np.min([a, b])
        maxLon = np.max([a, b])

        # TODO: maybee you need to look at the box around the satelitte image
        # TODO need to take into account the area the gridbox represent.

        # 1) Calculate a dLon matrix and dLat
        # sum to consecutive lat lons and divide by two.

        # 2) Problem the area is not square. How to calculate a weighted cloud fraction.
        # 3) How can I calculate percentage of contribution from a boundary.


        # Find the index of the cell which correspond to the
        x_idx, y_idx = np.where((lon_nc > minLon) & (lon_nc < maxLon) & (lat_nc > minLat) & (lat_nc < maxLat))
        X = np.unique(x_idx)
        Y = np.unique(y_idx)

        # This is done with the era images ...
        meanLon = np.mean(  lon_nc[X, :][:, Y]    )
        meanLat = np.mean(  lat_nc[X, :][:, Y]   )


        # retrieve cloud mask
        submat = clm_grb[X, :][:, Y]
        submat[submat == 3.] = np.nan
        count_nan = (np.isnan(submat)).sum()
        if count_nan  > 0: # 3 denotes of earth disk. The cloud data can only originate from inside the earth disk
            print("OBS the nr 3. lon: {} lat: {}   || meanlon {} mean lat {}".format(lon, lat, meanLon, meanLat))
        submat[submat == 1] = 0 # no cloud over ocean --> no cloud
        submat[submat == 2] = 1 # skyer denoted 1 --> praktisk for å beregne cloud fraction
        cloud_fraction = np.nanmean(submat)
        _x, _y = submat.shape
        count_cells = int(_x*_y)
        return cloud_fraction, count_cells, count_nan

    def delta_lon():
        # 1) make a copy and shift it to the left or right THINK!
        d_lon = lon_nc[X, :][:, Y]

    def criterion_for_boundary():
        """
        If clm boundaries are
        """
        pass


    def calc_all_cells():
        # TODO :: hent fra git på nettet.
        pass

    def merge_ts_to_one_dataset(self, grib_files):
        """
        grib_files : list of files
            typically one year?
        """
        counter = 0
        for filename in grb_files:# grb file of satellite image...
            #print(filename)
            if counter == 0:
                #print("enters 0")
                clm, cnt_cells, cnt_nans = calc_all(filename, nc_file = nc_files[0])
                self.ds = xr.Dataset({'tcc': (['latitude', 'longitude'],  tcc),
                                 'nr_nans':(['latitude', 'longitude'], cnt_nans),
                                'nr_cells':(['latitude', 'longitude'], cnt_cells)},
                                 coords={'longitude': (['longitude'], lon),
                                         'latitude': (['latitude'], lat),
                                        })

                ts = timestamp(filename)
                self.ds['time'] = ts

                # Add time as a coordinate and dimension.
                self.ds = self.ds.assign_coords(time = data.time)
                self.ds = self.ds.expand_dims(dim = 'time')
                counter += 1

            else:
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
                self.ds.merge(new_ds)

                counter += 1
        return

    def output_filename(self, var, start, stop, add_text_begin = ""):
        """
        Add test train or valid in front when this is appropriate.

        OBS: Var should be in terms of variables in the era-interim datasets.
        """
        return (LAPTOP_RESULTS_REPO+ add_text_begin+var + "_" + start + "_" +
                    stop + "_" + self.season_str + ".nc")


    def _get_season(self, season):
        """ OBS:

        Only usefull after merge ts.
        Returns the xarray dataset containing only one season.
        """
        self.data = self.data.groupby('time.season')
        for group in self.data:
            key, dataset = group
            if key == season:
                return dataset
        return

    def _write_file():
        # to a nc dump of the file
        pass

if __name__ == "__main__":
    # Generate the satelite data below here.
