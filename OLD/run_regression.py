from regression_ms_new import SimpleLinearRegression
import numpy as np
import multiprocessing

"""
class Worker(multiprocessing.Process):

    def __init__(self,season = "JJA", climate_zone = "Boreal", par = "qv"):
        self.season = season
        self.climate_zone = climate_zone
        self.par = par

    def run(self):
        r = SimpleLinearRegression(self.season, self.climate_zone, self.par)
        r.regression()
"""

def calc(season = "JJA", climate_zone = "Boreal", par = "qv"):
    r = SimpleLinearRegression(season, climate_zone, par)
    r.regression()


if __name__ == "__main__":
    print("Number of cpu : ", multiprocessing.cpu_count())

    SEASONS = ["DJF", "MAM", "JJA", "SON"]
    CLIMATE_ZONES = ["Boreal", "Temperate", "Mediterranean"]
    VARIABLES = ["P", "T", "qv", "rh"] #

    LONGITUDE = np.arange(-15, 42+0.25, 0.25) #" Doesn't include the last one "
    LATITUDE = {"Boreal": np.arange(57, 70+0.25, 0.25),
                "Temperate": np.arange(46,57, 0.25),
                "Mediterranean": np.arange(30, 46, 0.25)}

    jobs = []
    counter = 0
    for season in SEASONS:
        for climate_zone in CLIMATE_ZONES:
            for variable in VARIABLES:
                print("counter " + str(counter))
                p = multiprocessing.Process(target = calc, args = (season, climate_zone, variable,))
                jobs.append(p)
                p.start()
                counter += 1

    for j in jobs:
        j.join()
