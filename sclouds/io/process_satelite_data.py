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

          """
          if season is None:
              self.data = xr.open_dataset(self.filename)
              self.season_str = "all"
          else:
              self.data = xr.open_dataset(self.filename)
              # TODO it might be better to call this subset.
              self.data = get_season(season)
              self.season_str = season
          """
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
        sek = ts[12:-1]
        return np.datetime64( year+"-"+month+"-"+day+"T"+hr+":"+minuts+":"+sek )

        # Updating the variable name.
    def fix_one_sat_file(filename):
        """
        filename : str
            full absoulute path to file.

        This reads a raw satelite file.
        Add time, longitude, latitude.
        """
        data = xr.open_dataset(filename, engine="pynio")

        ts = timestamp(filename)
        data['time'] = ts

        # Add time as a coordinate and dimension.
        data = data.assign_coords(time = data.time)

        # TODO :: dimensions with coordinates
        #data = data.assign_coords(latitude = np.arange(-67.5, 67.5, 3712))
        #data = data.assign_coords(longitude = np.arange(-67.5, 67.5, 3712))

        data['xgrid_0'] = np.linspace(-67.5, 67.5, 3712)
        data['ygrid_0'] = np.linspace(-67.5, 67.5, 3712)


        data = data.expand_dims(dim = 'time' )

        # Rename dimenstions and coordinates
        data = data.rename(name_dict={'xgrid_0': 'longitude',
                                      'ygrid_0':'latitude',
                                      'CLOUDM_P30_GSV0_I207':'tcc'}, inplace=True)
        # crop to correct region
        # TODO :: import region
        data = data.sel(latitude = slice(30,60))
        data = data.sel(longitude = slice(-15, 42))

        # Check that it contains no 4's
        nr_4 = (data.tcc == 3.0).sum().values
        if nr_4 > 0:
            print( nr_4 )
        else:
            print("We are all good.")

        # Replace values --> final result cloud = 1, no_cloud = 0.
        # currently not replacing any values.
        data = data.where(data.tcc - 1 < 0.0001, 0)
        return data

    def _regrid_tcc(self):
        """
        Regrid the cloud mask to total cloud cover, which has the same spatial
         resolution as era interim data.z<

        Be aware that when you slice they take the smaller numer i.e.
        if you write 30,60 you will get the number closest to not larger than
        30 and 60, but you would wat to include the one thats a bit larger.


        NON NESESARY.
        """
        pass

    def calculate_cloud_fractions(self):
        """
        You can do this by adding the wrong coordinate system to the satelite file

        Its might be of use when writing the thesis to know how many small
         gridcells you calculate you fraction based on. If its few it not very diverse...
        """

        #self.data = self.data.sel('latitude' slice = (), "longitude" slice = ())
        pass

    def merge_ts_to_one_dataset(self, datset_one_file):
        """
        THIS NEEDS A LONG LIST OF FILES.
        MAYBEE THIS CAN BE DONE YEARLY FIRST.
        """
        # ADD coordinates to dimension without coordinates.
        self.data.merge(datset_one_file)
        pass

    def output_filename(self, var, start, stop, add_text_begin = ""):
        """
        Add test train or valid in front when this is appropriate.
        OBS: var should be in terms of variables in the era-interim datasets.
        """
        return (LAPTOP_RESULTS_REPO+ add_text_begin+var + "_" + start + "_" +
                    stop + "_" + self.season_str + ".nc")


    def _get_season(self, season):
        """ OBS: only usefull after merge ts.
        Returns the xarray dataset containing only one season.
        """
        self.data = self.data.groupby('time.season')
        for group in self.data:
            key, dataset = group
            if key == season:
                return dataset
        return

if __name__ == "__main__":
    # Generate the satelite data below here.
