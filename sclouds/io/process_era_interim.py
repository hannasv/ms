import glob
import xarray as xr
import numpy as np
from sclouds.helpers import LAPTOP_REPO, LAPTOP_RESULTS_REPO

class ProcessEraData:
    """
    Class which is ment to process era-intetim data in a certain way.

    1. Split into seasons.
    2. Split into train-, validation- and testdata.

    3. Share a method for writing the files.

    STORE ALL THE PROCESSED DATA INTO THE : DATA_REPO
    """

    def __init__(self, var, season = None):
        self.var = var
        self.season = season
        self.filename = glob.glob(LAPTOP_REPO + var+"*.nc")[0]

        if season is None:
            self.data = xr.open_dataset(self.filename)
            self.season_str = "all"
        else:
            self.data = xr.open_dataset(self.filename)
            # TODO it might be better to call this subset.
            self.data = self.get_season(season)
            print(self.data)
            self.season_str = season
        self.data = self.crop_era_interim() # Remove all data before satelite
                                       # measurments.


    def output_filename(self, var, start, stop, add_text_begin = ""):
        """
        Add test train or valid in front when this is appropriate.
        OBS: var should be in terms of variables in the era-interim datasets.
        """
        return LAPTOP_RESULTS_REPO+ add_text_begin+var + "_" + start + "_" + stop + "_" + self.season_str + ".nc"

    def get_season(self, season):
        """
        Returns the xarray dataset containing only one season.
        """
        self.data = self.data.groupby('time.season')
        for group in xarray:
            key, dataset = group
            if key == season:
                return dataset
        return


    def split_into_train_vaild_test_data(self, train_split = None, valid_split = None, test_split= None):
        """
        Do the train, validate, split on the data.

        When doing this split it needs to be done in full years otherwise
        seasonal training will be skeewed.

        All the splits are tuples (start, stop) if None, not interrested in this.

        train_split: default 01.01.2008
            The first date in training data.
            trainingdata from 01.01.2008-31.12.2014 (In total 6yrs)
            Example input : ('2008-01-01','2014-12-31')

        valid_split:
            The first date in training data. Here 01.01.2015.
            (01.01.2015-31.12.2016) (In total 2yrs)

        test_split:
            The test date in training data. Here ??.
            (01.01.2017-31.12.2018) (In total 2yrs)

        Here split into seasons, and training ++ and then save file.
        """

        if train_split is not None:
            tr_start, tr_stop = train_split
            train = self.data.sel(time = slice(tr_start, tr_stop))
            train.to_netcdf(self.output_filename(self.var, start=tr_start, stop=tr_stop,
                                    add_text_begin = "train_") )
        if test_split is not None:
            te_start, te_stop = test_split
            test = self.data.sel(time = slice(te_start, te_stop))
            test.to_netcdf(self.output_filename(self.var, start=te_start, stop=te_stop,
                                    add_text_begin = "test_") )
        if valid_split is not None:
            va_start, va_stop = valid_split
            validate = self.data.sel(time = slice(va_start, va_stop))
            validate.to_netcdf(self.output_filename(self.var, start=va_start, stop=va_stop,
                                    add_text_begin = "valid_") )
        return



    def crop_era_interim(self):
        """
        Remove data from before 01.01.2008.
        This is the time of the first satelite observations.
        """
        return self.data.sel(time = slice('2008-01-01','2018-12-31'))


if __name__ == "__main__":
    data = ProcessEraData(var = "r")

    data.split_into_train_vaild_test_data(
    train_split = ('2008-01-01', '2014-12-31'), valid_split=None,test_split= None)
    #valid_split = ('2015-01-01','2016-12-31'),
    #test_split= ('2017-01-01','2018-12-31'))
