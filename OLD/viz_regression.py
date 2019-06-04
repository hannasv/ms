import seaborn as sns
import numpy as np
import pandas as pd

class Viz:

    LONGITUDE = np.arange(-15, 42+0.25, 0.25) #" Doesn't include the last one "
    LATITUDE = {"Boreal": np.arange(57, 70+0.25, 0.25),
                "Temperate": np.arange(46,57, 0.25),
                "Mediterranean": np.arange(30, 46, 0.25)}

    def __init__(self, filename):
        self.filename = filename


    def read_and_restructure():
        pass

    def heatmap():
        pass
