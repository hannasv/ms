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
    """
    PATH =  "//uio/lagringshotell/geofag/students/metos/hannasv/dataERA5/"

    def __init__(self, season = "JJA",  par = "T"):
        self.results = {"a":[], "b":[]}# "lat":[], "lon":[]}
        self.par = par

        self.longitude = np.arange(-15, 42+0.25, 0.25) #" Doesn't include the last one "
        self.latitude =  np.arange(30, 67+0.25, 0.25) #
        self.season = season


    def perform_reg_one_cell(self, season = "JJA", climatezone = "Boreal", par1 = "T", par2 = "TCC", lat = 0, lon = 0):
        """ TODO: params = ["T", "RH", "qv", "P"]  """
        #def reg(season = "JJA", climatezone = "Boreal", par1 = "T", par2 = "TCC", lat = 0, lon = 0):

        tcc_files = glob.glob(self.PATH+season+"*"+climatezone+"*"+par2+".nc")
        # Add all these together --> Use concat
        for counter, fil in enumerate(tcc_files):
            if counter == 0:
                y = xa.open_dataset(fil).sel(latitude = lat, longitude = lon).to_array()[0]
            else:
                new = xa.open_dataset(fil).sel(latitude = lat, longitude = lon).to_array()[0]
                y = np.concatenate((y, new), axis=None)

        y = y.reshape(len(y), 1)
        print("shape y" + str(y.shape))
        # TODO change this to loop over all params
        variable_files = glob.glob(self.PATH+season+"*"+climatezone+"*"+par1+".nc")
        # Add all these together --> Use concat
        for counter, fil in enumerate(variable_files):
            if counter == 0:
                #x = xa.open_dataset(fil1).sel(latitude = lat, longitude = lon).to_array()[0]
                x = xa.open_dataset(fil).sel(latitude = lat, longitude = lon).to_array()[0]
            else:
                new = xa.open_dataset(fil).sel(latitude = lat, longitude = lon).to_array()[0]
                x = np.concatenate((x, new), axis=None)

        X = x.reshape(len(x), 1)

        print(" Managed to add files .. ")

        model = linear_model.LinearRegression()
        model.fit(X,y)
        return (model.coef_[0][0], model.intercept_[0])

    def regression(self):
        """
        TODO update for four parameters when it woorks"""
        # loop over lat and long
        for lon in self.longitude:
            for lat in self.latitude[self.climatezone]:
                #print(lat,lon)
                a,b = self.perform_reg_one_cell(season = self.season, climatezone = self.climatezone, par1 = self.par, par2 = "TCC", lat = lat, lon = lon)
                #print(a,b)
                self.results["a"].append(a)
                self.results["b"].append(b)

                #self.results["lat"].append(lat)
                #self.results["lon"].append(lon)
        # Doesnt return anything saves all result to the contstructur
        self.write_file()
        return

    def write_file(self):
        # self.results
        df = pd.DataFrame(self.results)

        df.to_csv(path_or_buf = "./Results/" + self.season + "_" + self.climatezone + "_" + self.par + ".csv",index=False, sep =",")
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
