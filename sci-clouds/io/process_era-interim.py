import sci-clouds.helpers import DATA_REPO, VARIABLES


class ProcessEraData:
    """
    Class which is ment to process era-intetim data in a certain way.

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

    def split_into_train_vaild_test_data(train_split, valid_split, test_split, last):
        """
        Do the train, validate, split on the data.

        train_split: default 01.01.2008
            The first date in training data.
            trainingdata from 01.01.2008-31.12.2014 (In total 6yrs)

        valid_split:
            The first date in training data. Here 01.01.2015.
            (01.01.2015-31.12.2016) (In total 2yrs)

        test_split:
            The test date in training data. Here ??.
            (01.01.2017-31.12.2018) (In total 2yrs)

        last: Default 31.12.2018.
            Should be updated to 31.08.2019 when the dataset is available.

        """
        pass



    def crop_era_interim():
        """
        Crop the timeseries of Era-Interim data to be the same size as satelite
        data and save to
        """
