import os
import glob

import xarray as xr
import numpy as np

from sclouds.helpers import (LAPTOP_RESULTS_REPO, LAPTOP_REPO,
                                VARIABLES, PRESSURE_LEVELS, LAT, LON)

class LinPred:
    """
    Exploring linear predictability.

    Class used to calculate the linear predictive power of a certain variable
    pp :: predictive power a(q90 - a10)

    q90, and q10 an be read from Results Stats folder.
    """
    def __init__(self, var = None, season = None, p_lev = 1000):
        """
        var : str
            if not specified; retrieve data from all variables.
        """
        pass

    def 
