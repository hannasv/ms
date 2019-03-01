import numpy as np
import pandas as pd
import glob

class ExploringStatisticalStationarity:

    PATH =  "//uio/lagringshotell/geofag/students/metos/hannasv/dataERA5/"
    #PATH = "./../../../lagringshotell/dataERA5/"
    SEASONS = ["DJF", "MAM", "JJA", "SON"]
    CLIMATE_ZONES = ["Boreal", "Temperate", "Mediterranean"]
    VARIABLES = ["qv", "P", "rh", "T"]

    def __init__(self, season = "JJA", climatezone = "Boreal", par = "T"):
        self.all_year_results = {"min":[], "max":[], "std":[], "mean":[]}
        # "lat":[], "lon":[]}
        self.periode_defined_results = {}
        self.par = par

        self.longitude = np.arange(-15, 42+0.25, 0.25) #" Doesn't include the last one "
        self.latitude = {"Boreal": np.arange(57, 70+0.25, 0.25),
                        "Temperate": np.arange(46,57, 0.25),
                        "Mediterranean": np.arange(30, 46, 0.25)}

        self.season = season
        self.climatezone = climatezone
        self.counter = 0


    def calc_stats_all_years(self, seasonal = False, season = "JJA", climatezone = "Boreal", par = "T"):
        # Read all files
        if not seasonal:
            variable_files = glob.glob(self.PATH+season+"*"+climatezone+"*"+par+".nc")

            if par == "T" or par == "P":
                for counter, fil in enumerate(variable_files):
                    if counter == 0:
                        x = xa.open_dataset(fil).sel(latitude = lat, longitude = lon).to_array()[0]
                    else:
                        new = xa.open_dataset(fil).sel(latitude = lat, longitude = lon).to_array()[0]
                        x = np.concatenate((x, new), axis=None)

            else:
                for counter, fil in enumerate(variable_files):
                    if counter == 0:
                        x = xa.open_dataset(fil).sel(latitude = lat, longitude = lon, level = 1000).to_array()[0]

                    else:
                        print(np.shape(x))
                        new = xa.open_dataset(fil).sel(latitude = lat, longitude = lon, level = 1000).to_array()[0]
                        x = np.concatenate((x, new), axis=None)

                X = x.reshape(len(x), 1)
        else:
            if par == "T" or par == "P":
                fil = glob.glob(self.PATH+season+"*"+climatezone+"*"+par+".nc")
                x = xa.open_dataset(fil).sel(latitude = lat, longitude = lon).to_array()[0]
            else:
                x = xa.open_dataset(fil).sel(latitude = lat, longitude = lon, level = 1000).to_array()[0]


        self.all_year_results['min'] = np.min(X)
        self.all_year_results['max'] = np.max(X)
        self.all_year_results['std'] = np.std(X)
        self.all_year_results['mean'] = np.mean(X)
        # write to file
        self.write_file("Statistics_all_years")






        def write_file(self, folder):
            # self.results
            df = pd.DataFrame(self.all_year_results)
            df.to_csv(path_or_buf = "./" +folder + "/" + self.season + "_" + self.climatezone + "_" + self.par + ".csv",index=False, sep =",")
            return


if __name__ == "__main__":

    a = ExploringStatisticalStationarity(season = "MAM", climatezone = "Boreal", par = "qv")
    a.calc_stats_all_years()
