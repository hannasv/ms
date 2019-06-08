import xarray as xa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import netCDF4 as n
from sklearn import linear_model
import pandas as pd

class OLS:
    """
    THIS IS A COPY FORM OLD FILES REWRITE THIS

    Ordinary least squares regression, for each cell.
    Save a, b in order to use them in vizualisations --> Better to do this is

    All regression is done with one variable against total cloud cover.
    """
    PATH =  "//uio/lagringshotell/geofag/students/metos/hannasv/dataERA5/"

    def __init__(self, var = "t2m", season = "JJA"):
        _empty_results()
        self.par = par
        # TODO this should be imported from somewhere.
        self.longitude = np.arange(-15, 42+0.75, 0.75) #" Doesn't include the last one "
        self.latitude =  np.arange(30, 60+0.75, 0.75)
        self.season = season
        return

    def perform_reg_one_cell(self, season = "JJA", var = "t2m", lat = 0, lon = 0):
        """ TODO:
            params = ["T", "RH", "qv", "P"]  """
        #def reg(season = "JJA", climatezone = "Boreal", par1 = "T", par2 = "TCC", lat = 0, lon = 0):

        tcc_files = glob.glob(self.PATH+season+"*"+climatezone+"*"+par2+".nc")
        y = xa.open_dataset(fil).sel(latitude = lat, longitude = lon).to_array()[0]

        variable_files = glob.glob(self.PATH+season+"*"+climatezone+"*"+par1+".nc")
        x = xa.open_dataset(fil).sel(latitude = lat, longitude = lon).to_array()[0]

        y= y.reshape(len(y), 1)
        X = x.reshape(len(x), 1)

        model = linear_model.LinearRegression()
        model.fit(X,y)
        return (model.coef_[0][0], model.intercept_[0])

    def regression(self):
        """
        Perform the regression over all cells for all seasons.
        Write file and empty_results everytime we write a files???
        """
        for v in vars:
            for lon in self.longitude:
                for lat in self.latitude:
                    #print(lat,lon)
                    a,b = self.perform_reg_one_cell(season = self.season,
                                            var = self.var, lat = lat, lon = lon)
                    #print(a,b)
                    self.results["a"].append(a)
                    self.results["b"].append(b)
                    self.results["lat"].append(lat)
                    self.results["lon"].append(lon)
            # Doesnt return anything saves all result to the contstructur
            self.write_file()
            empty_results()

        return

    def _empty_results():
        self.results = dict()
        self.results["a"]   = []
        self.results["b"]   = []
        self.results["lat"] = []
        self.results["lon"] = []
        return


    def write_file(self):
        # self.results
        df = pd.DataFrame(self.results)
        output_filename = "blabla"
        df.to_csv(path_or_buf = "./Results/" + self.season + "_" + self.climatezone + "_" +
                    self.par + ".csv",index=False, sep =",")
        return

class PredictivePower:
    """
    Class used to calculate the linear predictive power of a certain variable
    pp :: predictive power a(q90 - a10)

    q90, and q10 an be read from Results Stats folder.

    """
    def __init__():
        pass



if __name__ == "__main__":
    SimpleLinearRegression().regression()
