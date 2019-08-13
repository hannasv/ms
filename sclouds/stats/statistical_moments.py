import os
import glob

import xarray as xr
import numpy as np

from sclouds.helpers import (LAPTOP_RESULTS_REPO, LAPTOP_REPO,
                                VARIABLES, PRESSURE_LEVELS, LAT, LON)

class Stats:
    """
    Class which calculates the statistical properties of a dataset.

    Statistical properties:
        1. mean
        2. median
        3. MAD
        4. min
        5. max
        6. Std (not variance) because it is used in normalization.
        7. 10 and 90 percentile.
    """
    STATS = ['mean', 'median', 'period', 'season', 'variable',
             'p_lev', 'MAD', 'STD', 'min', 'max', "q10", 'q90']

    # TODO ::
        # 1 ) Decide where to store the data and what format you want to store it in.

    def __init__(self, var = None, season = None, p_lev = 1000):
        """
        var : str
            if not specified; retrieve data from all variables.
        """
        # TODO: This function could be able to take a list of variables and seasons
        self.var = var
        self.season = season
        self.empty_results()
        self.p_lev = p_lev

        if var == "t2m":
            fil_var = "temperature"

        self.filename = glob.glob(LAPTOP_REPO + "*" + fil_var + "*.nc")[0]
        self.longitude = np.arange(-15, 42 + 0.75, 0.75)
        self.latitute = np.arange(30, 60 + 0.75, 0.75)

        if var != "r" or var != 'q':
            self.dataset = xr.open_dataset(self.filename)
            p_lev = 1000
        else:
            self.dataset = xr.open_dataset(self.filename).sel(level=p_lev)
            print(self.dataset)


    def compute_one(self, var, start, stop, season):
        """
        Computes statistics for one variable and one season.
        """
        self.results['mean'].append(data.mean()[var].to_pandas() )
        medi = data.median()[var].to_pandas()
        self.results['median'].append(  medi   )
        self.results['MAD'].append( MAD( data, medi  ) )
        self.results['period'].append( start+"-"+stop   )
        self.results['season'].append( season )
        self.results['variable'].append( var  )
        self.results['STD'].append( data.std()[var].to_pandas()  )
        self.results['min'].append( data.min()[var].to_pandas()   )
        self.results['max'].append( data.max()[var].to_pandas()  )
        self.results['q10'].append(  data.quantile(q=0.1)[var].to_pandas()   )
        self.results['q90'].append(  data.quantile(q=0.9)[var].to_pandas()   )
        self.results['p_lev'].append( self.p_lev )
        return

    #def _calculate_for_all_cells(self,):
        #for lat in self.latitude:
            #for lon in self.longitude:
                # hent ut en celle 
                #self.compute_one(var, start, stop, season)

    def output_filename(self, var, start, stop, add_text_begin = ""):
        """
        Add test train or valid in front when this is appropriate.
        OBS: var should be in terms of variables in the era-interim datasets.
        """
        return (LAPTOP_RESULTS_REPO + add_text_begin + var + "_" + start + "_" + stop
                    + "_" + self.season_str + "_stats.csv")

    def _write_file(self):
        """
        Creates a csv file containing the result.
        TODO : store it in lagringshotellet.
        """
        df = pd.DataFrame(self.results)
        df.to_csv(path_or_buf = self.output_filename(), index=False, sep =",")
        return

    def MAD(data, median):
        """
        Copy WIKI: In statistics, the median absolute deviation (MAD) is a robust
        measure of the variability of a univariate sample of quantitative data.

        data : array-like
        median : float
            This is already computed in the script. Makes more scence to send it
            as input and not calculate it again.
        """
        return (data[var] - median).median()[var].to_pandas()

    def empty_results(self):
        """ Empty the results dictionary everytime you finish one variable
        otherwise all the old information will be dumped to the file. """
        self.results = {}
        for str in self.STATS:
            self.results[str] = []


if __name__ == "__main__":
    # Generate the satelite data below here.
    #for var in VARIABLES:
        #for p_lev in PRESSURE_LEVELS:
    st = Stats("t2m")
