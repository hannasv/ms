import os
import glob

import xarray as xr
import numpy as np

data_m = '/home/hannasv/Desktop/lagrings/ERA5_monthly/'

class DataLoaderAR:
    """ lOADING ON PIXEL OF DATA TO AR MODELS
    
    start: str
        on the form year_month
    stop: str
        year_month
    lat: float
        latitude value
    lon: float 
        logitude
        
    Methods
    -------------
    load_data
    
    """

    def __init__(self, start, stop, lat, lon, num_ts = 0, bias = True, validation_split = 0.0):
        self.start = start
        self.stop = stop
        self.lat = lat
        self.lon = lon
        self.num_ts = num_ts 
        self.bias = bias
        self.validation_split = validation_split
        
        self.files = None
        self.X = None
        self.y = None
        
        self.load_data_AR()
        
    def set_files(self):
        """ Get files in requested period.
        Insert test on start > stop. 

        Returns:
            List of files to be read into 

        """
        y, m = self.start.split('_')
        start_y = int(y)
        start_m = int(m)

        y, m = self.stop.split('_')
        stop_y = int(y)
        stop_m = int(m)

        years = np.arange(start_y, stop_y+1)
        months = np.arange(1, 13)  
        search_str = [self.start, self.stop]

        for y in years:
            for m in months:
                m = "%2.2d" %m
                tmp = '{}_{}'.format(y, m)
                if tmp > self.start and tmp < self.stop:
                    search_str.append(tmp)   

        storage = '/home/hannasv/Desktop/lagrings/ERA5_monthly/'

        files = []
        for ss in search_str:
            tmp_files = glob.glob(os.path.join( storage, '*{}*.nc'.format(ss)  ))  
            files += tmp_files
        self.files = files
        return files
        
    # Loader of AR 
    def load_data_AR(self):
        """ Prepares data for 

        start, stop : str
            year_month. Limited to montly values.

        lat, lon : float
            coordinate information

        bias : bool 
            Default 1

        validation_split : float
            Number between 0 and 1. 
            Intended to make it similar to keras input data. Will it make it easier to train networks though...?
        """
        if self.validation_split > 0:
            raise NotImplementedError('Comming soon ... Make sure that you actually need this?')

        # Search for list of files in the range start stop.
        # Returns list of strings you can merge to a dataset 
        print("Load data .... ")
        files = self.set_files()
        datasets = [ xr.open_dataset(fil).sel(latitude = self.lat, longitude = self.lon) for fil in self.files]
        data = xr.merge(datasets) 

        n     = len(data.time.values)
        #n_lat = len(data.latitude.values)
        #n_lon = len(data.longitude.values)

        q   = data.q.values
        t2m = data.t2m.values
        r   = data.r.values
        sp  = data.sp.values

        tcc = data.tcc.values

        if self.bias:
            num_vars = 4+1+self.num_ts
        else:
            raise NotImplementedError('Coming soon')
            num_vars = 4 + self.num_ts

        num_samples = n - self.num_ts
        X = np.zeros((num_samples, num_vars+1))

        X[:, 0] = 1
        X[:, 1] = t2m[:n-self.num_ts]
        X[:, 2] = r[:n-self.num_ts]
        X[:, 3] = sp[:n-self.num_ts]
        X[:, 4] = q[:n-self.num_ts]

        y = tcc[:n-self.num_ts, np.newaxis]

        for i in range(1, self.num_ts+1):
            print("Adds previos ts of tcc ...")
            if num_ts - i != 0:
                X[:, 4+i] = tcc[i:num_ts-i]
            else:
                X[:, 4+i] = tcc[i:]

        X[:, -1] = tcc[:n-self.num_ts]
        # Now drop rows contaning nans.
        no_nans =  X[~np.isnan(X.any(axis=1))]
        self.X = no_nans[:, :-1]
        self.y = no_nans[:, -1, np.newaxis]
        return self.X, self.y