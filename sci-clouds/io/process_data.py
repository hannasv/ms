import sci-clouds.helpers import DATA_REPO, VARIABLES


class ProcessData:
    """
    Class which is ment to process data in a certain way.

    1. Split into seasons.
    2. Split into train-, validation- and testdata.

    3. Share a method for writing the files.

    STORE ALL THE PROCESSED DATA INTO THE : DATA_REPO
    """

    def __init__():
        pass

    def split_seasons_and_write_files(xarray):
        xarray = xarray.groupby('time.season')
        for group in xarray:
            key, dataset = group
            # TODO write file, use key in filename and dataset.dump to netcdf
            # To save the files.

    def split_into_train_vaild_test_data():
        """
        Do the train, validate, split on the data.
        """
        pass

    def regrid_tcc():
        """
        Regrid the cloud mask to total cloud cover, which has the same spatial
         resolution as era interim data.z<
        """
        pass


    def crop_era_interim():
        """
        Crop the timeseries of Era-Interim data to be the same size as satelite
        data and save to
        """
