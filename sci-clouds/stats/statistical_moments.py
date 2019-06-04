import xarray as xa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import netCDF4 as n

from pycloudprediction.helpers import DATA_REPO, FIGURE_REPO, VARIABLES, PRESSURE_LEVELS

class Stats:
    """
    Class which calculates the statistical properties of a dataset.

    Statistical properties:
        1. mean
        2. median
        3. MAD
        4. min
        5. max
        6. std or is it more interrestingly to look at the variance.
        7. 10 and 90 percentile.
    """

    STATS = ['mean', 'median', 'period', 'season', 'variable',
             'p_lev', 'MAD', 'STD', 'min', 'max', "q10", 'q90']

    # Double check this
    # WHAT YOU WANT THE STAT TO CONTAIN

    # TODO ::
        # 1 ) Decide where to store the data and what format you want to store it in.

    def __init__(var = None):
        """
        var : str
            if not specified; retrieve data from all variables.
        """

        self.results = {}


    def MAD(data, median):
        """
        Copy WIKI: In statistics, the median absolute deviation (MAD) is a robust
        measure of the variability of a univariate sample of quantitative data.

        data : array-like
        median : float
            This is already computed in the script. Makes more scence to send it
            as input and not calculate it again.
        """
        return np.median(data - median)

    def empty_results():
        """ Empty the results dictionary everytime you finish one variable
        otherwise all the old information will be dumped to the file. """
        self.results = {}
