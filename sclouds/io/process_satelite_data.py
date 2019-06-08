class ProcessSateliteData:
    """
    Class for processing sataelite data.

    It needs to be:
        1. Cropped to the correct regions.
        2. Replace "no cloud over ocean and land" with "no cloud".
        3. Regridd and calculate cloud fractions
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
          self.filename = glob.glob(self.path+var+"*.grb")

          if season is None:
              self.data = xr.open_dataset(self.filename)
              self.season_str = "all"
          else:
              self.data = xr.open_dataset(self.filename)
              # TODO it might be better to call this subset.
              self.data = get_season(season)
              self.season_str = season
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

    def filter_values(self):
        """
        Make all data, a cloud = 1, no cloud = 0. By replacing no cloud over
         land and ocean with no cloud.
        """
        pass

    def _regrid_tcc(self):
        """
        Regrid the cloud mask to total cloud cover, which has the same spatial
         resolution as era interim data.z<

        Be aware that when you slice they take the smaller numer i.e.
        if you write 30,60 you will get the number closest to not larger than
        30 and 60, but you would wat to include the one thats a bit larger.

        """
        pass

    def calculate_cloud_fractions(self):
        """
        You can do this by adding the wrong coordinate system to the satelite file

        Its might be of use when writing the thesis to know how many small
         gridcells you calculate you fraction based on. If its few it not very diverse...

        """
        pass

    def merge_ts_to_one_dataset(self):
        """
        THIS NEEDS A LONG LIST OF FILES.
        MAYBEE THIS CAN BE DONE YEARLY FIRST.
        """
        pass


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
