class DataLoaderAR:
    """Class for loading data to AR models used in this thesis.

    SHOULD THIS BE A FUNCTION INSTEAD.
    """

    def __init__(self, start, stop, lat_range = (30, 50), lon_range = (-15, 25)):
      self.start = start
      self.stop = stop
      self.lat_range = lat_range
      self.lon_range = lon_range

    def merge_all_data_in_xarray(self):
        """ Get all available timesteps for a certain pixel

        Returns
            xr.Dataset
        """
        # First read files, extract that pixel
        # Merge all the pixels into one dataset.
        ds = None
        return ds

    def load_pixel():
        """Loads data for one pixel, to avoid to much memory comsumption when performing the inverse."""

        data =     def merge_all_data_in_xarray(self):
        # Extract the data and structures it in a format suiatble for inverting.

        # [t2m , sp, r, qv, tcc-1, tcc-2, tcc-3 .... ]


      pass





class DataLoaderKeras:
    def ___init__(self):
        pass
