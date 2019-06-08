import xarray as xa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import netCDF4 as n

from sclouds.helpers import DATA_REPO, FIGURE_REPO, VARIABLES, PRESSURE_LEVELS

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

    # Double check this
    # WHAT YOU WANT THE STAT TO CONTAIN

    # TODO ::
        # 1 ) Decide where to store the data and what format you want to store it in.

    def __init__(var = None, season = None):
        """
        var : str
            if not specified; retrieve data from all variables.
        """
        # TODO: This function could be able to take a list of variables and seasons
        self.var = VARIABLES
        self.season = season
        empty_results()

         if p_lev is None:
            self.datasets = [xa.open_dataset(fil) for fil in self.files]
            p_lev = 3
        else:
            self.datasets = [xa.open_dataset(fil).isel(level=p_lev) for fil in self.files]

    def compute_stat_one(var, start, stop, season):
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
        self.results['p_lev'].append(  data.level  )

        return


    def _write_file():
        """
        Creates a csv file containing the result.
        TODO : store it in lagringshotellet.
        """
        df = pd.DataFrame(self.results)
        df.to_csv(path_or_buf = "./stats/" + s + "_" + cz + "_" +
        param +"_p_lev"+ str(p_lev) + ".csv",index=False, sep =",")
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

    def empty_results():
        """ Empty the results dictionary everytime you finish one variable
        otherwise all the old information will be dumped to the file. """
        self.results = {}
        for str in STATS:
            self.results[str] = []
