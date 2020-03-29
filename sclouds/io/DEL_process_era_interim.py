import glob
import xarray as xr
import numpy as np
#from sclouds.helpers import LAPTOP_REPO, LAPTOP_RESULTS_REPO

def create_ML_repo(start = "2012-01-01", stop = "2013-01-01", season = "SON",
                  era_path = '/uio/lagringshotell/geofag/students/metos/hannasv/era_interim_data/'):
    """
    Want all year put season to "".
    """
    q   = xr.open_dataset(glob.glob(era_path + "*_q_*" + season +".nc")[0]).q.values
    r   = xr.open_dataset(glob.glob(era_path + "*_r_*"+ season +".nc")[0]).r.values
    tcc = xr.open_dataset(glob.glob(era_path + "*tcc*"+ season +".nc")[0]).tcc.values
    sp  = xr.open_dataset(glob.glob(era_path + "*sp*"+ season +".nc")[0]).sp.values
    t2m = xr.open_dataset(glob.glob(era_path + "*t2m*"+ season +".nc")[0]).t2m.values
    assert np.shape(q) == np.shape(r) == np.shape(tcc) == np.shape(sp) == np.shape(t2m)

    nbr_times, nbr_lats, nbr_lon = np.shape(q)
    #print(np.shape(q[0]))
    #print(nbr_times, nbr_lats, nbr_lon)
    train = []
    true  = tcc
    for i in range(nbr_times):
        one_timestep = np.array([ q[i], r[i], tcc[i], t2m[i] ])
        #print(one_timestep.shape)
        train.append(one_timestep)
    return np.array(train), true

# crop_nc_file(fil, fil[:-3]+"_cropped.nc")
def create_seasonal_files(infile):
    W = xr.open_dataset(infile)

    for i, season in enumerate(W.groupby('time.season')):
        key, dataset = season
        # TODO make this variable more general.
        outfile = infile[:-3]+"_{}.nc".format(key)
        dataset.to_netcdf(path = outfile)
        #cropped.to_netcdf(path = outfile)
    return

def crop_nc_file_to_MS_experiments(infile, outfile):
    """
    Original nc files contain data from 1979 to end of 2018.

    Humidities are only kept for surface plev = 1000.
    time is from 2012 to end of 2018
    longitude -15, 29.25
    latitude 30, 55

    Resolution is 0.75 degrees.

    """
    Q =  xr.open_dataset(infile)

    variable = fil.split('_')[-2]
    if 'r' == variable or 'q' == variable:
        #print('enters for filename {}'.format( infile ))
        cropped = Q.sel(level = 1000,
                          longitude = slice(-15, 29.25),
                          latitude = slice(55.5, 30),
                          time = slice('2012-01-01', '2018-12-31'))
    else:
        cropped = Q.sel(longitude = slice(-15, 29.25),
                        latitude = slice(55.5, 30),
                        time = slice('2012-01-01', '2018-12-31'))
    #print(cropped)
    cropped.to_netcdf(path = outfile)
    return


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
    #data = ProcessEraData(var = "r")
    #data.split_into_train_vaild_test_data(
    #train_split = ('2008-01-01', '2014-12-31'), valid_split=None,test_split= None)
    #valid_split = ('2015-01-01','2016-12-31'),
    #test_split= ('2017-01-01','2018-12-31'))

    for fil in files:
        outfile = fil[:-3] + "_MS_region.nc"
        crop_nc_file_to_MS_experiments(fil, outfile)
    #create_ML_repo(start = "2012-01-01", stop = "2013-01-01", season = "SON",
    #                  era_path = '/uio/lagringshotell/geofag/students/metos/hannasv/era_interim_data/')
