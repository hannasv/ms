
import os
import glob
import json

import numpy as np

#import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras import losses, optimizers
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization

from kerastuner import HyperModel


# PATH to these needs to be updated on the server
from sclouds.helpers  import (path_input, path_convlstm_results, get_lon_array, get_lat_array)
from sclouds.io.utils import (dataset_to_numpy_grid_keras, get_xarray_dataset_for_period,
                              train_test_split_keras)

# Custom properties made for keras.
from sclouds.ml.ConvLSTM.hyper_convlstm import HyperConvLSTM
from sclouds.ml.ConvLSTM.utils import r2_keras, keras_custom_loss_function

# Packages from keras tuner.
from kerastuner.tuners import RandomSearch
from kerastuner import HyperParameters

"""
ConvLSTM 3x3-256-2
ConvLSTM filterxfilter-num_hidden_states-num_layers


MSE loss and adam optimizer, sier det er pga det er reelle tall,
observasjoner vil jo aldri kunne bli komplekse tall.

Varianter i Air Quality Forecast:
ConvLSTM 1x1-256-2
ConvLSTM 3x3-256-2
ConvLSTM 5x5-256-2
ConvLSTM 3x3-256-3
ConvLSTM 5x5-256-3

Cell further away doesn't provide usefull information, neither does deeper models.

"""


def generate_folder_name(filter_dim, num_hidden_states, num_layers):
        return 'ConvLSTM_{}x{}-{}-{}'.format(filter_dim, filter_dim,
                                         num_hidden_states, num_layers)


hp = HyperParameters()
hypermodel = HyperConvLSTM(num_hidden_layers = 2, seq_length= 4)

""" Create several hyper models settings and

    Num hidden layers :
     1, 2, 3?

    Focus on sequence length:
        seq_length = 4 hourse
        24 would be one day
        week : 24*7
"""



tuner = RandomSearch(
        hypermodel,
        objective='mean_squared_error',
        max_trials=10,
        allow_new_entries = True,
        directory=path_convlstm_results, # determines where the reslts should be stored.
        project_name='test_hyperparameters')


print(tuner.search_space_summary())


data = get_xarray_dataset_for_period(start = '2012-01-01', stop = '2012-01-31')
X_train, y_train, X_test, y_test = train_test_split_keras(data, seq_length = 4, val_split=0.2)

tuner.search(X_train, y_train,
             epochs=2,
             validation_data=(X_test, y_test))

models = tuner.get_best_models(num_models=2)

""" Store the history for plotting later"""
import pandas as pd

# convert the history.history dict to a pandas DataFrame:
hist_df = pd.DataFrame(history.history)

# TODO: produce a file generator

# or save to csv:
hist_csv_file = 'history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)
