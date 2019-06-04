import numpy as np
import glob
from sklearn import linear_model
import xarray as xa
import pandas as pd

class SimpleLinearRegression():

    """
    Class for doing simple regression in each gridcell in one area.

    # longitude is equal for all files -15.0 -14.75 -14.5 -14.25 ... 41.5 41.75 42.0
    # Boreal latitude : 75.0 74.75 74.5 74.25 ... 57.5 57.25 57.0
    # Temperate : 56.75 56.5 56.25 56.0 ... 46.5 46.25 46.0
    # Mediterranean : 45.75 45.5 45.25 45.0 ... 30.5 30.25 30.0

    Saving years 11-18 to be tested on

    """
    #PATH =  "//uio/lagringshotell/geofag/students/metos/hannasv/dataERA5/"
    PATH = "./../../../lagringshotell/"
    def __init__(self, season = "JJA", climatezone = "Boreal", par = "T"):
        print("Denne blir brukt")
        self.results = {"a":[], "b":[]}# "lat":[], "lon":[]}
        self.par = par

        self.longitude = np.arange(-15, 42+0.25, 0.25) #" Doesn't include the last one "
        self.latitude = {"Boreal": np.arange(57, 70+0.25, 0.25),
                        "Temperate": np.arange(46,57, 0.25),
                        "Mediterranean": np.arange(30, 46, 0.25)}
        self.season = season
        self.climatezone = climatezone


    def perform_reg_one_cell(self, season = "JJA", climatezone = "Boreal", par1 = "T", par2 = "TCC", lat = 0, lon = 0):
        print(par1)
        """ Performs regression in one cell """
        variable_files = glob.glob(self.PATH+season+"*"+climatezone+"*"+par1+".nc")
        print(variable_files)
        if par1 == "T" or par1 == "P":
            for counter, fil in enumerate(variable_files[:-1]):
                if counter == 0:
                    x = xa.open_dataset(fil).sel(latitude = lat, longitude = lon).to_array()[0]
                else:
                    new = xa.open_dataset(fil).sel(latitude = lat, longitude = lon).to_array()[0]
                    x = np.concatenate((x, new), axis=None)

        else:
            """
            Need to choose the correct pressure level
            """
            for counter, fil in enumerate(variable_files[:-1]):
                if counter == 0:
                    x = xa.open_dataset(fil).sel(latitude = lat, longitude = lon, level = 1000).to_array()[0]

                else:
                    print(np.shape(x))
                    new = xa.open_dataset(fil).sel(latitude = lat, longitude = lon, level = 1000).to_array()[0]
                    x = np.concatenate((x, new), axis=None)

        X = x.reshape(len(x), 1)
        print("Finished reading X ")

        tcc_files = glob.glob(self.PATH+season+"*"+climatezone+"*"+"TCC"+".nc")
        #y = np.ones(10)
        for counter, fil in enumerate(tcc_files[:-1]):
            if counter == 0:
                y = xa.open_dataset(fil).sel(latitude = lat, longitude = lon).to_array()[0]
            else:
                new = xa.open_dataset(fil).sel(latitude = lat, longitude = lon).to_array()[0]
                #new = new.reshapelen(new), 1)
                print("On the second one for y ")
                y = np.concatenate((y, new), axis=None)
        y = y.reshape(len(y), 1)

        model = linear_model.LinearRegression()
        print(" Starts regression on " + self.season + "_" + self.climatezone + "_" + self.par )
        model.fit(X,y)
        return (model.coef_[0][0], model.intercept_[0])

    def regression(self):
        for lon in self.longitude:
            for lat in self.latitude[self.climatezone]:
                #print(lat,lon)
                a,b = self.perform_reg_one_cell(season = self.season, climatezone = self.climatezone, par1 = self.par, par2 = "TCC", lat = lat, lon = lon)
                #print(a,b)
                self.results["a"].append(a)
                self.results["b"].append(b)
            self.write_file()

        return

    def write_file(self):
        # self.results
        df = pd.DataFrame(self.results)
        df.to_csv(path_or_buf = "./Results/" + self.season + "_" + self.climatezone + "_" + self.par + ".csv",index=False, sep =",")
        return

if __name__ == "__main__":

    a = SimpleLinearRegression(season = "MAM", climatezone = "Boreal", par = "qv")
    a.regression()
