""" Explenation of the content of this file.
"""

import sys
import os
import glob

import numpy as np
import xarray as xr

from timeit import default_timer as timer


from utils import (mean_squared_error, r2_score, fit_pixel, predict_pixel,
                     accumulated_squared_error,
                     sigmoid, inverse_sigmoid)

from utils import (dataset_to_numpy, dataset_to_numpy_order,
#from sclouds.ml.regression.utils import (mean_squared_error, r2_score, fit_pixel, predict_pixel,
#                     accumulated_squared_error,
#                     sigmoid, inverse_sigmoid)
#
#from sclouds.ml.regression.utils import (dataset_to_numpy, dataset_to_numpy_order,

                    dataset_to_numpy_order_traditional_ar,
                              dataset_to_numpy_grid_order,
                              dataset_to_numpy_grid,
                              get_xarray_dataset_for_period,
                              get_list_of_files_excluding_period,
                              get_list_of_files,
                              get_list_of_files_excluding_period_traditional_model,
                              get_list_of_files_traditional_model)


sys.path.insert(0,'/uio/hume/student-u89/hannasv/MS/sclouds/')
from helpers import (merge, get_list_of_variables_in_ds,
                             get_pixel_from_ds, path_input, path_ar_results)

from model import Model
#sys.path.insert(0,'/uio/hume/student-u89/hannasv/MS/sclouds/')
#from sclouds.helpers import (merge, get_list_of_variables_in_ds,
                             get_pixel_from_ds, path_input, path_ar_results)

#from sclouds.ml.regression.model import Model

base = '/uio/lagringshotell/geofag/students/metos/hannasv/results/stats/2014-01-01_2018-12-31/' #'2014-01-01_2018-12-31/

class ParallellModel(Model):

    def save(self):
        """ Saves model configuration, evaluation, transformation into a file
        named by the current time. Repo : /home/hanna/lagrings/results/ar/
        """
        filename      = '/uio/lagringshotell/geofag/students/metos/hannasv/results/ar/MODEL_{}_{}.nc'.format(
                                np.min(self.longitude), np.datetime64('now'))
        path_ar_results = '/uio/lagringshotell/geofag/students/metos/hannasv/results/ar/'
        #os.path.join(path_ar_results, )
        print('Stores file {}'.format(filename))
        config_dict   = self.get_configuration()
        weights_dict  = self.get_weights()

        eval_dict     = self.get_evaluation()

        # Merges dictionaries together
        config_dict.update(weights_dict)
        config_dict.update(eval_dict)

        ds = xr.Dataset(config_dict,
                         coords={'longitude': (['longitude'], self.longitude),
                                 'latitude': (['latitude'], self.latitude),
                                })
        ds.to_netcdf(filename)
        return

def get_train_test(test_start, test_stop, model = 'ar'):
    #print('gets train test data for ar model')
    if type == 'ar':
        files_train = get_list_of_files_excluding_period(test_start, test_stop)
        files_test = get_list_of_files(test_start, test_stop)

    else:
        files_train = get_list_of_files_excluding_period_traditional_model(test_start, test_stop)
        files_test = get_list_of_files_traditional_model(test_start, test_stop)

    #print('Detected {} train files and {} test files. Merging might take a while .... '.format(len(files_train), len(files_test)))
    train_dataset = merge(files_train)
    #print('finished merging train')
    test_dataset = merge(files_test)
    return train_dataset, test_dataset

from multiprocessing import Process, Queue

def config_model(start, stop, test_start, test_stop,
                 train_dataset, test_dataset, order, transform,
                 sigmoid, latitude, longitude,
                 type):
<<<<<<< HEAD

    print('Starting {}'.format(longitude))

    m = ParallellModel(start = start, stop = stop,
                       test_start = test_start, test_stop = test_stop,
                       train_dataset = train_dataset, test_dataset = test_dataset,
                       order = order, transform = transform,
                       sigmoid = sigmoid, latitude = latitude, longitude = longitude,
                       type = type)#.fit().save()

    print('Finished initialized ')
    m.fit()

    print('passed fit')
    m.save()
    return


if __name__ == '__main__':
    start = None
    stop  = None

    print('start new run.')

    latitudes = np.arange(30,  50+0.25, step = 0.25)
    longitudes = np.arange(-15, 25+0.25, step = 0.25)

    #test_start = '2014-01-01'
    #test_stop  = '2018-12-31'
    test_start = '2012-01-01'
    test_stop  = '2012-01-31'
    
    order = 1
    sig = False
    trans = True
    bias = True

    timer_start = timer()

    #test_start = '2014-01-01'
    #test_stop  = '2018-12-31'

    type = 'traditional'
    train_dataset, test_dataset = get_train_test(test_start, test_stop,
                                                 model = type)

    sig = False
    trans = True
    bias = True

    num_threads = 15
    n_lon = len(longitudes)
    parts = round(n_lon/num_threads)

    proces = []

    counter = 0
    longitudes = np.arange(num_threads)
    for lon in np.array_split(longitudes, num_threads):
        tr_data_sel =  train_dataset.sel(longitude = slice(min(lon), max(lon))).copy()
        te_data_sel =  test_dataset.sel(longitude = slice(min(lon), max(lon))).copy()

        p = Process(target=config_model, args=(start, stop, test_start,
                           test_stop, tr_data_sel, te_data_sel, order, transform,
                            sigmoid, latitudes, lon, type))
        
        p.start()
        print('starts thread {}'.format(counter))
        counter+=1
        proces.append(p)

    print('Started all .. ')

    # for pro in proces:
    #    print('Enters join .. ')
    #    pro.join()
    for lon in np.array_split(longitudes, num_threads):
        p = Process(target=config_model, args=(start, stop, test_start,
                           test_stop, train_dataset, test_dataset, order, transform,
                            sigmoid, latitudes, lon, type)).start()
        proces.append(p)

    #for t in threads:
        #t.join()
    """
    m = Model(start = start, stop = stop,
                 test_start = test_start, test_stop = test_stop,
                 train_dataset = train_dataset, test_dataset = test_dataset,
                 order = 1,                 transform = trans,
                 sigmoid = sig, latitude = None, longitude = None,
                 type = type)
    coeff = m.fit()
    m.save()
    print(m.get_configuration())
    """
